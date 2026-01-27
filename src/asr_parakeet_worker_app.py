# Parakeet ASR Worker App - Multi-instance for parallel batch processing

import logging
import os
import asyncio
import base64
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf, open_dict
import numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel
from nemo.collections.asr.models import EncDecRNNTBPEModel
from typing import List, Any, Optional

# --- Load Environment Variables ---
load_dotenv()

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()

# --- Model & Device Configuration ---
MODEL_PATH = os.getenv("ASR_MODEL_PATH", "nvidia/parakeet-tdt-0.6b-v2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
NUM_MODEL_INSTANCES = int(os.getenv("ASR_PARAKEET_NUM_WORKERS", "2"))

# --- Global Resources ---
model_pool: Queue = Queue()  # Pool of model instances
model_executor: Optional[ThreadPoolExecutor] = None


# --- Request Body Model ---
class BatchTranscriptionRequest(BaseModel):
    audio_batch: List[str]  # List of base64 encoded audio bytes


# ======================================================================================
#  CORE TRANSCRIPTION FUNCTION
# ======================================================================================
def _transcribe_with_model(batch_audio_b64: List[str]) -> List[Optional[Any]]:
    """Process a batch using a model from the pool"""
    # Decode base64 audio
    try:
        batch_audio_np = [
            np.frombuffer(base64.b64decode(b64_audio), dtype=np.float32)
            for b64_audio in batch_audio_b64
        ]
    except Exception as e:
        logger.error(f"Could not decode audio batch: {e}")
        return [None] * len(batch_audio_b64)

    # Get model from pool
    model = model_pool.get(timeout=3)
    try:
        with torch.no_grad():
            hypotheses = model.transcribe(
                audio=batch_audio_np, 
                batch_size=len(batch_audio_np), 
                return_hypotheses=True, 
                verbose=False
            )
            formatted_results = []
            for hyp in hypotheses:
                if hyp and hyp.text:
                    word_confidences = [c.item() for c in hyp.word_confidence] if hasattr(hyp, 'word_confidence') and hyp.word_confidence else []
                    avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
                    formatted_results.append({
                        "text": hyp.text,
                        "word_confidence": word_confidences,
                        "avg_confidence": round(avg_confidence, 4),
                    })
                else:
                    formatted_results.append(None)
            return formatted_results
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return [None] * len(batch_audio_np)
    finally:
        # Return model to pool
        model_pool.put(model)


# ======================================================================================
#  FASTAPI LIFECYCLE & ENDPOINT
# ======================================================================================
@app.on_event("startup")
async def startup_event():
    global model_executor
    logger.info(f"🚀 Parakeet Worker starting up with {NUM_MODEL_INSTANCES} model instances...")
    
    try:
        for i in range(NUM_MODEL_INSTANCES):
            model = EncDecRNNTBPEModel.from_pretrained(MODEL_PATH, map_location=torch.device(DEVICE))
            model.eval().to(DEVICE)
            
            with open_dict(model.cfg.decoding) as cfg:
                cfg.compute_timestamps = True
                cfg.preserve_alignments = True
                cfg.confidence_cfg = OmegaConf.create({
                    'method_cfg': {'name': 'max_prob'}, 
                    'preserve_word_confidence': True
                })
            model.change_decoding_strategy(model.cfg.decoding)
            
            model_pool.put(model)
            logger.info(f"✅ Parakeet model instance {i+1}/{NUM_MODEL_INSTANCES} loaded on {DEVICE}.")
        
        # Warmup all instances
        logger.info("🔥 Warming up Parakeet model instances...")
        def warmup_model():
            model = model_pool.get()
            try:
                dummy_audio = [np.random.randn(SAMPLE_RATE).astype(np.float32)]
                with torch.no_grad():
                    model.transcribe(audio=dummy_audio, batch_size=1, return_hypotheses=False, verbose=False)
            finally:
                model_pool.put(model)
        
        with ThreadPoolExecutor(max_workers=NUM_MODEL_INSTANCES) as warmup_executor:
            list(warmup_executor.map(lambda _: warmup_model(), range(NUM_MODEL_INSTANCES)))
        logger.info("✅ Parakeet model instances warmed up.")
        
        # Create thread pool for handling requests
        model_executor = ThreadPoolExecutor(max_workers=NUM_MODEL_INSTANCES, thread_name_prefix="parakeet_worker")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Parakeet models: {e}", exc_info=True)
    
    logger.info("🎉 PARAKEET WORKER READY 🎉")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down model executor...")
    if model_executor:
        model_executor.shutdown(wait=True)


@app.post("/batch_transcribe")
async def transcribe_endpoint(request: BatchTranscriptionRequest):
    start_time = time.perf_counter()
    if model_pool.qsize() == 0:
        logger.error("Request received but no model instances available.")
        return Response(content="Server is not ready.", status_code=503)

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        model_executor, _transcribe_with_model, request.audio_batch
    )
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    logger.info(f"Parakeet transcription for batch of {len(request.audio_batch)} in {processing_time_ms:.2f} ms.")
    
    return results


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_pool.qsize() > 0 else "unhealthy",
        "model_path": MODEL_PATH,
        "num_instances": NUM_MODEL_INSTANCES,
        "available_instances": model_pool.qsize(),
        "device": DEVICE,
        "sample_rate": SAMPLE_RATE,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)