import os
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List
import torch
import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configure logger
logger.remove()
logger.add(
    "logs/speaker_verification_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

app = FastAPI()

SAMPLE_RATE = 16000

@dataclass
class Exemplar:
    """A single speaker exemplar with embedding and confidence score"""
    embedding: torch.Tensor
    confidence: float  # ASR confidence score

@dataclass
class SpeakerProfile:
    """
    Stores speaker embedding profile and verification state.
    
    The mean_embedding is automatically maintained as the average of all current exemplars.
    It is recalculated whenever exemplars are updated (when a new high-confidence embedding
    makes it into the top-K).
    """
    embeddings: list  # List of embeddings collected during enrollment
    mean_embedding: Optional[torch.Tensor] = None  # Average of current exemplars (auto-updated)
    exemplars: list = None  # Top-K exemplars with confidence scores
    is_enrolled: bool = False
    num_speeches: int = 0
    total_verifications: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    # Pending embedding from latest utterance (for commit-based enrollment)
    pending_embedding: Optional[torch.Tensor] = None
    pending_confidence: Optional[float] = None  # ASR confidence for pending embedding
    
    def __post_init__(self):
        if self.exemplars is None:
            self.exemplars = []

# Global state
sessions: Dict[str, SpeakerProfile] = defaultdict(lambda: SpeakerProfile(embeddings=[]))
models: List[torch.nn.Module] = []  # Multiple model instances for parallel processing
device = None

# Configuration
NUM_SPEECHES_FOR_ENROLLMENT = int(os.getenv("SPEAKER_VERIFICATION_ENROLLMENT_SPEECHES", "3"))

# Three-threshold verification:
# - similarity < THRESHOLD1: NOT passed
# - THRESHOLD1 <= similarity < THRESHOLD2: uncertain (needs LLM confirmation)
# - THRESHOLD2 <= similarity < THRESHOLD3: PASSED, NO UPDATE
# - similarity >= THRESHOLD3: PASSED + UPDATE
SIMILARITY_THRESHOLD1 = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD1", "0.2"))
SIMILARITY_THRESHOLD2 = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD2", "0.3"))
SIMILARITY_THRESHOLD3 = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD3", "0.5"))
MIN_AUDIO_DURATION_MS = float(os.getenv("SPEAKER_VERIFICATION_MIN_AUDIO_MS", "300"))

# Top-K Exemplar Configuration
MAX_EXEMPLARS = int(os.getenv("SPEAKER_VERIFICATION_MAX_EXEMPLARS", "3"))
CENTROID_WEIGHT = float(os.getenv("SPEAKER_VERIFICATION_CENTROID_WEIGHT", "0.6"))
MAX_EXEMPLAR_WEIGHT = float(os.getenv("SPEAKER_VERIFICATION_MAX_EXEMPLAR_WEIGHT", "0.4"))

# Embedding Model Configuration
# Options: "titanet" (NVIDIA TitaNet-Large via NeMo) or "redimnet" (ReDimNet via torch.hub)
EMBEDDING_MODEL = os.getenv("SPEAKER_EMBEDDING_MODEL", "titanet").lower()
# ReDimNet model size: b0, b1, b2, b3, b5, b6, M (default: M for best quality)
REDIMNET_MODEL_NAME = os.getenv("REDIMNET_MODEL_NAME", "b6")

# Multi-worker Configuration
NUM_ENROLLMENT_WORKERS = int(os.getenv("SPEAKER_VERIFICATION_ENROLLMENT_WORKERS", "2"))
NUM_VERIFICATION_WORKERS = int(os.getenv("SPEAKER_VERIFICATION_VERIFICATION_WORKERS", "3"))

# Global queues for request distribution
enrollment_queue: Optional[asyncio.Queue] = None
verification_queue: Optional[asyncio.Queue] = None

def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio from raw PCM bytes (16-bit mono) and convert to float32 [-1, 1]."""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    return torch.from_numpy(audio_float).unsqueeze(0)  # (1, T)

def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings (GPU-accelerated if available)."""
    emb1_norm = emb1 / (torch.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (torch.norm(emb2) + 1e-8)
    return torch.dot(emb1_norm.flatten(), emb2_norm.flatten()).item()

def compute_hybrid_similarity(embedding: torch.Tensor, exemplars: List[Exemplar]) -> tuple[float, float, float]:
    """
    Compute hybrid similarity score using top-K exemplars.
    
    Returns:
        (hybrid_score, centroid_similarity, max_exemplar_similarity)
    
    Formula: score = CENTROID_WEIGHT * cosine(centroid, emb) + MAX_EXEMPLAR_WEIGHT * max(cosine(exemplar_i, emb))
    """
    if not exemplars:
        return 0.0, 0.0, 0.0
    
    # Compute centroid from all exemplars
    exemplar_embeddings = torch.stack([ex.embedding for ex in exemplars])
    centroid = torch.mean(exemplar_embeddings, dim=0)
    
    # Centroid similarity
    centroid_sim = compute_cosine_similarity(centroid, embedding)
    
    # Max exemplar similarity
    max_exemplar_sim = max(
        compute_cosine_similarity(ex.embedding, embedding)
        for ex in exemplars
    )
    
    # Hybrid score
    hybrid_score = CENTROID_WEIGHT * centroid_sim + MAX_EXEMPLAR_WEIGHT * max_exemplar_sim
    
    return hybrid_score, centroid_sim, max_exemplar_sim

def update_exemplars_and_mean(profile: SpeakerProfile, new_embedding: torch.Tensor, new_confidence: float) -> bool:
    """
    Update exemplar list with new embedding based on confidence.
    Maintains top-K exemplars with highest confidence scores.
    When exemplars are updated, also recalculates mean_embedding.
    
    Returns:
        True if exemplar was added/replaced, False otherwise
    """
    exemplars = profile.exemplars
    
    if len(exemplars) < MAX_EXEMPLARS:
        # Room for more exemplars
        exemplars.append(Exemplar(
            embedding=new_embedding,
            confidence=new_confidence,
        ))
        profile.mean_embedding = _recalculate_mean_embedding(exemplars)
        return True
    
    # Find exemplar with lowest confidence
    min_confidence_idx = min(range(len(exemplars)), key=lambda i: exemplars[i].confidence)
    min_confidence = exemplars[min_confidence_idx].confidence
    
    if new_confidence > min_confidence:
        exemplars[min_confidence_idx] = Exemplar(
            embedding=new_embedding,
            confidence=new_confidence,
        )
        profile.mean_embedding = _recalculate_mean_embedding(exemplars)
        return True
    
    return False

def _recalculate_mean_embedding(exemplars: List[Exemplar]) -> torch.Tensor:
    """
    Recalculate mean embedding from current exemplars.
    Returns the mean of all exemplar embeddings.
    """
    if not exemplars:
        return None
    
    exemplar_embeddings = torch.stack([ex.embedding for ex in exemplars])
    return torch.mean(exemplar_embeddings, dim=0)

def extract_embedding(audio_bytes: bytes, model: torch.nn.Module, keep_on_gpu: bool = False) -> Optional[torch.Tensor]:
    """
    Extract speaker embedding from audio bytes using a specific model instance.
    
    Args:
        audio_bytes: Raw PCM audio bytes (16-bit mono)
        model: The model instance to use for extraction
        keep_on_gpu: If True, keep embedding on GPU for similarity computation
    """
    global device
    
    try:
        waveform = load_audio_from_bytes(audio_bytes, SAMPLE_RATE)
        audio_length = waveform.shape[-1]
        
        # Move to device
        audio_input = waveform.to(device, dtype=torch.float32)
        
        with torch.no_grad():
            if EMBEDDING_MODEL == "redimnet":
                # ReDimNet: input [N, T], output [N, embedding_dim]
                device_type = "cuda" if device.type == "cuda" else "cpu"
                precision = torch.float16 if device_type == "cuda" else torch.float32
                with torch.autocast(device_type=device_type, dtype=precision):
                    embedding = model(audio_input)
            else:
                # TitaNet (NeMo): requires input_signal_length
                audio_length_tensor = torch.tensor([audio_length], device=device)
                _, embedding = model.forward(
                    input_signal=audio_input,
                    input_signal_length=audio_length_tensor
                )
        
        # Squeeze to get single embedding vector
        embedding = embedding.squeeze(0)
        
        if not keep_on_gpu:
            embedding = embedding.cpu()
        
        return embedding
    
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}", exc_info=True)
        return None

async def enrollment_worker(worker_id: int, model: torch.nn.Module):
    """Worker for enrollment requests - processes one request at a time."""
    global enrollment_queue
    
    logger.info(f"Enrollment worker {worker_id} started")
    
    while True:
        try:
            # Wait for a request from the queue
            data = await enrollment_queue.get()
            
            session_id = data["session_id"]
            future = data["future"]
            audio_bytes = data["audio_bytes"]
            asr_confidence = data.get("asr_confidence", 0.85)
            
            try:
                logger.debug(f"[Worker-E{worker_id}][{session_id}] Processing enrollment request")
                
                # Extract embedding using this worker's model
                embedding = extract_embedding(audio_bytes, model, keep_on_gpu=False)
                
                if embedding is None:
                    future.set_result({
                        "status_code": 500,
                        "content": {"error": "Failed to extract embedding"}
                    })
                    continue
                
                profile = sessions[session_id]
                
                if profile.is_enrolled:
                    future.set_result({
                        "status_code": 200,
                        "content": {
                            "enrolled": True,
                            "message": "Already enrolled, use /verify endpoint"
                        }
                    })
                    continue
                
                # Store as pending embedding with ASR confidence (will be committed later)
                profile.pending_embedding = embedding
                profile.pending_confidence = asr_confidence
                logger.debug(f"[Worker-E{worker_id}][{session_id}] Stored pending embedding (conf={asr_confidence:.3f})")
                
                future.set_result({
                    "status_code": 200,
                    "content": {
                        "enrolled": False,
                        "enrollment_progress": f"{len(profile.embeddings)}/{NUM_SPEECHES_FOR_ENROLLMENT}",
                        "message": "Embedding stored, waiting for commit",
                        "pending": True
                    }
                })
            
            except Exception as e:
                logger.error(f"[Worker-E{worker_id}] Error processing enrollment for {session_id}: {e}")
                if not future.done():
                    future.set_exception(e)
        
        except Exception as e:
            logger.error(f"[Worker-E{worker_id}] Error in enrollment worker: {e}", exc_info=True)

def _compute_verification_result(similarity: float, allow_update: bool, audio_duration_ms: float = 1000.0) -> tuple[bool | str, bool]:
    """
    Three-threshold verification logic with duration-based threshold adjustment.
    Returns (is_speaker, should_update)
    
    Thresholds:
    - < THRESHOLD1: NOT passed
    - [THRESHOLD1, THRESHOLD2): uncertain (needs LLM confirmation)
    - [THRESHOLD2, THRESHOLD3): PASSED, NO UPDATE
    - >= THRESHOLD3: PASSED + UPDATE
    
    For audio < 1 second: thresholds are lowered by up to 20% (linear scaling)
    - At 300ms: thresholds * 0.7
    - At 1000ms: thresholds * 1.0
    """
    # Adjust thresholds for short audio (< 1 second)
    threshold_scale = 1.0 
    if audio_duration_ms < 1000:
        # Linear scaling: 0.7 at MIN_AUDIO_DURATION_MS -> 1.0 at 1000
        threshold_scale = 0.7 + (0.3 * (audio_duration_ms - MIN_AUDIO_DURATION_MS) / (1000 - MIN_AUDIO_DURATION_MS))
    
    threshold2 = SIMILARITY_THRESHOLD2 * threshold_scale
    
    if similarity >= SIMILARITY_THRESHOLD3:
        return True, allow_update  # PASSED + UPDATE
    elif similarity >= threshold2:
        return "uncertain_high", False  # needs to see if this is a presence check
    elif similarity >= SIMILARITY_THRESHOLD1:
        return "uncertain_low", False  # needs to see if this is a presence check
    else:
        return False, False  # NOT passed

def _initialize_exemplars_from_embeddings(profile: SpeakerProfile):
    """Initialize exemplars from existing embeddings (for backward compatibility)."""
    if profile.exemplars or not profile.embeddings:
        return
    
    # Convert existing embeddings to exemplars
    # Handle both tuple format (embedding, confidence) and legacy tensor-only format
    for item in profile.embeddings[:MAX_EXEMPLARS]:
        if isinstance(item, tuple):
            emb, conf = item
        else:
            emb, conf = item, 0.85  # Legacy format fallback
        profile.exemplars.append(Exemplar(
            embedding=emb,
            confidence=conf,
        ))
    
    # Calculate mean_embedding from exemplars
    profile.mean_embedding = _recalculate_mean_embedding(profile.exemplars)
    
    exemplar_confs = [f"{ex.confidence:.2f}" for ex in profile.exemplars]
    logger.info(f"Initialized {len(profile.exemplars)} exemplars + mean_embedding from enrollment embeddings (confs={exemplar_confs})")

async def verification_worker(worker_id: int, model: torch.nn.Module):
    """Worker for verification requests - processes one request at a time."""
    global verification_queue
    
    logger.info(f"Verification worker {worker_id} started")
    
    while True:
        try:
            # Wait for a request from the queue
            data = await verification_queue.get()
            
            session_id = data["session_id"]
            future = data["future"]
            audio_bytes = data["audio_bytes"]
            allow_update = data.get("allow_update", True)
            audio_duration_ms = data.get("audio_duration_ms", 1000.0)
            
            try:
                logger.debug(f"[Worker-V{worker_id}][{session_id}] Processing verification request")
                
                # Extract embedding using this worker's model - keep on GPU for fast similarity
                embedding = extract_embedding(audio_bytes, model, keep_on_gpu=True)
                
                if embedding is None:
                    future.set_result({
                        "status_code": 500,
                        "content": {"error": "Failed to extract embedding"}
                    })
                    continue
                
                profile = sessions.get(session_id)
                if not profile or not profile.is_enrolled:
                    future.set_result({
                        "status_code": 400,
                        "content": {"error": "Speaker not enrolled yet, use /enroll endpoint first"}
                    })
                    continue
                
                # Initialize exemplars if needed (backward compatibility)
                _initialize_exemplars_from_embeddings(profile)
                
                # Store embedding for potential presence check (reuse instead of re-extracting)
                profile.pending_embedding = embedding
                
                # Compute hybrid similarity using top-K exemplars
                similarity, centroid_sim, max_exemplar_sim = compute_hybrid_similarity(embedding, profile.exemplars)
                
                # Log threshold adjustment for short audio
                if audio_duration_ms < 1000.0:
                    threshold_scale = 0.7 + (0.3 * (audio_duration_ms - MIN_AUDIO_DURATION_MS) / (1000 - MIN_AUDIO_DURATION_MS))
                    logger.debug(
                        f"[Worker-V{worker_id}][{session_id}] Short audio ({audio_duration_ms:.0f}ms) - "
                        f"adjusting thresholds by {threshold_scale:.2f}x"
                    )
                
                logger.debug(f"[Worker-V{worker_id}][{session_id}] Hybrid sim={similarity:.4f} (centroid={centroid_sim:.4f}, max_ex={max_exemplar_sim:.4f})")
                
                is_speaker, should_update = _compute_verification_result(similarity, allow_update, audio_duration_ms)
                
                # Update stats
                profile.total_verifications += 1
                if is_speaker is True:
                    profile.successful_verifications += 1
                    logger.info(f"[Worker-V{worker_id}][{session_id}] ✅ Verified (sim={similarity:.4f}, dur={audio_duration_ms:.0f}ms)")
                elif is_speaker == "uncertain_low":
                    logger.info(f"[Worker-V{worker_id}][{session_id}] ⚠️ Uncertain low (sim={similarity:.4f}, dur={audio_duration_ms:.0f}ms)")
                elif is_speaker == "uncertain_high":
                    logger.info(f"[Worker-V{worker_id}][{session_id}] ⚠️ Uncertain high (sim={similarity:.4f}, dur={audio_duration_ms:.0f}ms)")
                else:
                    profile.failed_verifications += 1
                    logger.warning(f"[Worker-V{worker_id}][{session_id}] ❌ Failed (sim={similarity:.4f}, dur={audio_duration_ms:.0f}ms)")
                
                future.set_result({
                    "status_code": 200,
                    "content": {
                        "is_speaker": is_speaker,
                        "similarity": round(similarity, 4),
                        "centroid_similarity": round(centroid_sim, 4),
                        "max_exemplar_similarity": round(max_exemplar_sim, 4),
                        "has_room": len(profile.exemplars) < MAX_EXEMPLARS,
                    }
                })
            
            except Exception as e:
                logger.error(f"[Worker-V{worker_id}] Error processing verification for {session_id}: {e}")
                if not future.done():
                    future.set_exception(e)
        
        except Exception as e:
            logger.error(f"[Worker-V{worker_id}] Error in verification worker: {e}", exc_info=True)

def _load_single_model(device) -> torch.nn.Module:
    """Load and return a single model instance."""
    if EMBEDDING_MODEL == "redimnet":
        # Load ReDimNet from torch.hub (https://github.com/IDRnD/redimnet)
        model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name=REDIMNET_MODEL_NAME,
            train_type="ft_lm",  # Fine-tuned on mixed datasets for best quality
            dataset="vox2"
        )
        model = model.to(device)
        model.eval()
    else:
        # Load TitaNet model from NVIDIA NeMo
        import nemo.collections.asr as nemo_asr
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel("WARNING")
        
        model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        model = model.to(device)
        model.eval()
    
    return model

@app.on_event("startup")
async def startup_event():
    """Load multiple speaker verification models and start worker pool on startup"""
    global models, device, enrollment_queue, verification_queue
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Calculate total models needed
    total_workers = NUM_ENROLLMENT_WORKERS + NUM_VERIFICATION_WORKERS
    
    logger.info(f"Loading {total_workers} model instances ({EMBEDDING_MODEL})...")
    
    for i in range(total_workers):
        model = _load_single_model(device)
        models.append(model)
        logger.info(f"✅ Model {i+1}/{total_workers} loaded")
    
    logger.info(
        f"Config: model={EMBEDDING_MODEL}, enrollment_speeches={NUM_SPEECHES_FOR_ENROLLMENT}, "
        f"enrollment_workers={NUM_ENROLLMENT_WORKERS}, verification_workers={NUM_VERIFICATION_WORKERS}, "
        f"threshold1={SIMILARITY_THRESHOLD1}, threshold2={SIMILARITY_THRESHOLD2}, threshold3={SIMILARITY_THRESHOLD3}, "
        f"max_exemplars={MAX_EXEMPLARS}, "
        f"centroid_weight={CENTROID_WEIGHT}, max_exemplar_weight={MAX_EXEMPLAR_WEIGHT}"
    )
    
    # Initialize queues
    enrollment_queue = asyncio.Queue()
    verification_queue = asyncio.Queue()
    
    # Start enrollment workers
    model_idx = 0
    for i in range(NUM_ENROLLMENT_WORKERS):
        asyncio.create_task(enrollment_worker(worker_id=i, model=models[model_idx]))
        model_idx += 1
    
    # Start verification workers
    for i in range(NUM_VERIFICATION_WORKERS):
        asyncio.create_task(verification_worker(worker_id=i, model=models[model_idx]))
        model_idx += 1
    
    logger.info(
        f"🔥 Worker pool started: {NUM_ENROLLMENT_WORKERS} enrollment + {NUM_VERIFICATION_WORKERS} verification workers"
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down speaker verification service...")
    sessions.clear()

def _check_audio_duration(audio_bytes: bytes, min_duration_ms: float) -> tuple[bool, float]:
    """Check if audio meets minimum duration. Returns (is_valid, duration_ms)."""
    duration_ms = (len(audio_bytes) / 2 / SAMPLE_RATE) * 1000
    return duration_ms >= min_duration_ms, duration_ms

async def _queue_and_wait(queue: asyncio.Queue, data: dict, timeout: float = 10.0) -> JSONResponse:
    """Queue request and wait for result - shared logic."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    data["future"] = future
    
    await queue.put(data)
    
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
        return JSONResponse(
            status_code=result["status_code"],
            content=result["content"]
        )
    except asyncio.TimeoutError:
        logger.error(f"[{data['session_id']}] Request timed out")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"error": "Request timed out"}
        )
    except Exception as e:
        logger.error(f"[{data['session_id']}] Request failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(e)}
        )

@app.post("/enroll/{session_id}")
async def enroll_speaker(session_id: str, request: Request):
    """Enroll a speaker by collecting embeddings from the first N speeches."""
    audio_bytes = await request.body()
    
    if not audio_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Empty audio data"}
        )
    
    # Get ASR confidence from header (default to 0.85 if not provided)
    asr_confidence = float(request.headers.get("X-ASR-Confidence", "0.85"))
    
    # Check if already enrolled (avoid unnecessary work)
    profile = sessions.get(session_id)
    if profile and profile.is_enrolled:
        logger.debug(f"[{session_id}] Already enrolled")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "enrolled": True,
                "message": "Already enrolled, use /verify endpoint"
            }
        )
    
    # Check minimum duration
    is_valid, duration_ms = _check_audio_duration(audio_bytes, MIN_AUDIO_DURATION_MS)
    if not is_valid:
        logger.debug(f"[{session_id}] Audio too short: {duration_ms:.0f}ms")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "enrolled": False,
                "enrollment_progress": f"{len(sessions[session_id].embeddings)}/{NUM_SPEECHES_FOR_ENROLLMENT}",
                "message": f"Audio too short ({duration_ms:.0f}ms), skipping"
            }
        )
    
    return await _queue_and_wait(enrollment_queue, {
        "session_id": session_id,
        "audio_bytes": audio_bytes,
        "asr_confidence": asr_confidence
    })

def _get_enrollment_progress(profile: SpeakerProfile) -> str:
    """Get enrollment progress string."""
    return f"{len(profile.embeddings)}/{NUM_SPEECHES_FOR_ENROLLMENT}"

@app.post("/commit-enrollment/{session_id}")
async def commit_enrollment(session_id: str):
    """
    Commit the pending embedding to enrollment collection.
    Called when the gateway confirms the speech segment is final.
    """
    global device
    
    profile = sessions.get(session_id)
    
    if not profile:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Session not found"}
        )
    
    if profile.is_enrolled:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"enrolled": True, "message": "Already enrolled"}
        )
    
    if profile.pending_embedding is None:
        logger.debug(f"[{session_id}] No pending embedding (progress: {_get_enrollment_progress(profile)})")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "No pending embedding to commit",
                "enrollment_progress": _get_enrollment_progress(profile)
            }
        )
    
    # Commit pending embedding with confidence as tuple (embedding, confidence)
    confidence = profile.pending_confidence if profile.pending_confidence is not None else 0.85
    profile.embeddings.append((profile.pending_embedding, confidence))
    profile.num_speeches += 1
    profile.pending_embedding = None
    profile.pending_confidence = None
    
    progress = _get_enrollment_progress(profile)
    
    # Check if enrollment complete
    if len(profile.embeddings) >= NUM_SPEECHES_FOR_ENROLLMENT:
        # Initialize exemplars from enrollment embeddings using stored ASR confidence
        profile.exemplars = []
        for item in profile.embeddings[:MAX_EXEMPLARS]:
            # Handle both tuple format (embedding, confidence) and legacy tensor-only format
            if isinstance(item, tuple):
                emb, conf = item
            else:
                emb, conf = item, 0.85  # Legacy format fallback
            profile.exemplars.append(Exemplar(
                embedding=emb.to(device),
                confidence=conf,
            ))
        
        # Calculate mean_embedding from exemplars
        profile.mean_embedding = _recalculate_mean_embedding(profile.exemplars)
        profile.is_enrolled = True
        
        exemplar_confs = [f"{ex.confidence:.2f}" for ex in profile.exemplars]
        logger.info(f"[{session_id}] 🎭 Speaker enrollment complete! Initialized {len(profile.exemplars)} exemplars (confs={exemplar_confs})")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "enrolled": True,
                "enrollment_progress": progress,
                "num_exemplars": len(profile.exemplars),
                "message": "Enrollment complete"
            }
        )
    
    logger.info(f"[{session_id}] 🎭 Enrollment progress: {progress} (conf={confidence:.3f})")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "enrolled": False,
            "enrollment_progress": progress,
            "message": "Enrollment in progress"
        }
    )

@app.post("/verify/{session_id}")
async def verify_speaker(session_id: str, request: Request):
    """
    Verify if audio matches the enrolled speaker profile.
    """
    audio_bytes = await request.body()
    
    if not audio_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Empty audio data"}
        )
    
    profile = sessions.get(session_id)
    if not profile or not profile.is_enrolled:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Speaker not enrolled yet, use /enroll endpoint first"}
        )
    
    # Check minimum duration
    is_valid, duration_ms = _check_audio_duration(audio_bytes, MIN_AUDIO_DURATION_MS)
    if not is_valid:
        logger.debug(f"[{session_id}] Audio too short: {duration_ms:.0f}ms")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "is_speaker": True,
                "similarity": SIMILARITY_THRESHOLD2 + 0.01, # just above threshold2 to pass verification
                "updated": False,
                "message": "Audio too short, skipping verification"
            }
        )
    
    # Get allow_update and audio_duration_ms from headers
    allow_update = request.headers.get("X-Allow-Update", "true").lower() == "true"
    audio_duration_ms = float(request.headers.get("X-Audio-Duration-MS", str(duration_ms)))
    
    return await _queue_and_wait(verification_queue, {
        "session_id": session_id,
        "audio_bytes": audio_bytes,
        "allow_update": allow_update,
        "audio_duration_ms": audio_duration_ms
    })

def _cleanup_profile(profile: SpeakerProfile):
    """Release all tensor references in a speaker profile."""
    profile.embeddings.clear()
    profile.exemplars.clear()
    profile.mean_embedding = None
    profile.pending_embedding = None
    profile.pending_confidence = None

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its speaker profile"""
    if session_id in sessions:
        profile = sessions[session_id]
        logger.info(
            f"[{session_id}] 🗑️ Deleting session | "
            f"enrolled={profile.is_enrolled}, "
            f"verifications={profile.total_verifications} "
            f"(✅{profile.successful_verifications}/❌{profile.failed_verifications})"
        )
        _cleanup_profile(profile)
        del sessions[session_id]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Session deleted"}
        )
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Session not found"}
    )

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get enrollment and verification status for a session"""
    profile = sessions.get(session_id)
    
    if not profile:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Session not found"}
        )
    
    # Get exemplar info
    exemplar_info = None
    if profile.exemplars:
        exemplar_info = {
            "count": len(profile.exemplars),
            "confidences": [round(ex.confidence, 3) for ex in profile.exemplars],
            "min_confidence": round(min(ex.confidence for ex in profile.exemplars), 3),
            "max_confidence": round(max(ex.confidence for ex in profile.exemplars), 3)
        }
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "session_id": session_id,
            "is_enrolled": profile.is_enrolled,
            "enrollment_progress": f"{len(profile.embeddings)}/{NUM_SPEECHES_FOR_ENROLLMENT}",
            "num_speeches": profile.num_speeches,
            "total_verifications": profile.total_verifications,
            "successful_verifications": profile.successful_verifications,
            "failed_verifications": profile.failed_verifications,
            "exemplars": exemplar_info
        }
    )

@app.post("/update-embedding/{session_id}")
async def update_embedding(session_id: str, request: Request):
    """
    Update speaker embedding using the pending embedding from last verification.
    Called when verification passes or LLM confirms uncertain verification.
    
    Body (JSON):
        confidence: float - ASR confidence score for the utterance
        force_replace: bool - If True, replace the lowest-confidence exemplar (for presence checks)
    """
    global device
    
    profile = sessions.get(session_id)
    
    if not profile or not profile.is_enrolled:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Speaker not enrolled yet"}
        )
    
    if profile.pending_embedding is None:
        logger.debug(f"[{session_id}] No pending embedding for update")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"updated": False, "message": "No pending embedding"}
        )
    
    # Get parameters from request body
    try:
        body = await request.json()
        confidence = body.get("confidence", profile.pending_confidence or 0.9)
        force_replace = body.get("force_replace", False)
    except:
        confidence = profile.pending_confidence or 0.9
        force_replace = False
    
    # Initialize exemplars if needed
    _initialize_exemplars_from_embeddings(profile)
    
    # Ensure embedding is on correct device
    embedding = profile.pending_embedding
    if device.type == "cuda" and not embedding.is_cuda:
        embedding = embedding.to(device)
    
    # Force add/replace (for presence checks) - bypasses confidence comparison
    if force_replace:
        capped_conf = min(confidence, 0.9)
        if len(profile.exemplars) < MAX_EXEMPLARS:
            profile.exemplars.append(Exemplar(embedding=embedding, confidence=capped_conf))
            profile.mean_embedding = _recalculate_mean_embedding(profile.exemplars)
            logger.info(f"[{session_id}] ➕ Added exemplar (conf={capped_conf:.3f}, count={len(profile.exemplars)})")
        else:
            min_idx = min(range(len(profile.exemplars)), key=lambda i: profile.exemplars[i].confidence)
            old_conf = profile.exemplars[min_idx].confidence
            profile.exemplars[min_idx] = Exemplar(embedding=embedding, confidence=capped_conf)
            profile.mean_embedding = _recalculate_mean_embedding(profile.exemplars)
            logger.info(f"[{session_id}] 🔄 Force replaced exemplar[{min_idx}] (old_conf={old_conf:.3f} → new_conf={capped_conf:.3f})")
        was_updated = True
    else:
        was_updated = update_exemplars_and_mean(profile, embedding, confidence)
    
    if was_updated:
        exemplar_confidences = [ex.confidence for ex in profile.exemplars]
        logger.info(
            f"[{session_id}] 📊 Exemplar + mean_embedding updated (conf={confidence:.3f}, "
            f"exemplars={len(profile.exemplars)}, confs={[f'{c:.2f}' for c in exemplar_confidences]})"
        )
    else:
        logger.debug(
            f"[{session_id}] Exemplar not updated (conf={confidence:.3f} too low, "
            f"min_current={min(ex.confidence for ex in profile.exemplars):.3f})"
        )
    
    profile.pending_embedding = None
    profile.pending_confidence = None
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "updated": was_updated,
            "confidence": confidence,
            "num_exemplars": len(profile.exemplars),
            "force_replaced": force_replace
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "embedding_model": EMBEDDING_MODEL,
        "active_sessions": len(sessions),
        "enrollment_workers": NUM_ENROLLMENT_WORKERS,
        "verification_workers": NUM_VERIFICATION_WORKERS,
        "enrollment_queue_size": enrollment_queue.qsize() if enrollment_queue else 0,
        "verification_queue_size": verification_queue.qsize() if verification_queue else 0
    }

if __name__ == "__main__":
    import uvicorn
    url = os.getenv("SPEAKER_VERIFICATION_WORKER_URL", "http://localhost:8092")
    if "localhost" in url:
        port = url.split(":")[-1]
        uvicorn.run(app, host="0.0.0.0", port=int(port))