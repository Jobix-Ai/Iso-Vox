"""
Audio Preprocessing Module (GPU-accelerated with Model Pool)
Handles:
- Audio format conversion (mulaw_8k → PCM 16kHz) with GPU resampling
- Noise reduction (GTCRN or MPSENet - configurable via NOISE_REDUCTION_MODEL env var)
- Audio normalization

GPU Parallelism:
- Model replicas with separate CUDA streams for true parallel processing
- Queue-based distribution across model instances
- Configurable workers per model type via env vars
"""

import logging
import os
import audioop
import time
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, List, Tuple, Callable, Dict
import numpy as np
import torch
import torchaudio.transforms as T
import soundfile as sf
from dotenv import load_dotenv

# Background thread pool for non-blocking audio saves
_audio_save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio_save")

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio config
SAMPLE_RATE = 16000

# Noise reduction config
NOISE_REDUCTION_ENABLED = os.getenv("NOISE_REDUCTION_ENABLED", "true").lower() == "true"
NOISE_REDUCTION_MODEL = os.getenv("AUDIO_NOISE_REDUCTION_MODEL", "gtcrn").lower()  # "gtcrn" or "mpsenet"
GTCRN_MODEL_CHECKPOINT = os.getenv("AUDIO_GTCRN_MODEL_CHECKPOINT", "weights/gtcrn_dns3.tar")
MPSENET_MODEL_NAME = os.getenv("AUDIO_MPSENET_MODEL_PATH", "JacobLinCool/MP-SENet-DNS")  # HuggingFace model
MPSENET_SEGMENT_SIZE = int(os.getenv("AUDIO_MPSENET_SEGMENT_SIZE", "32000"))  # 2 seconds at 16kHz

# Normalization config
TARGET_RMS = float(os.getenv("TARGET_RMS", "3000.0"))
MAX_GAIN_DB = float(os.getenv("MAX_GAIN_DB", "20.0"))
MAX_GAIN_LINEAR = 10 ** (MAX_GAIN_DB / 20.0)

# Speech separation config
DEBUG_SAVE_AUDIO = os.getenv("DEBUG_SAVE_AUDIO", "true").lower() == "true"
DEBUG_AUDIO_DIR = os.getenv("DEBUG_AUDIO_DIR", "debug_audio")

# Speech check config
SPEECH_CHECK_ENABLED = os.getenv("SPEECH_CHECK_ENABLED", "true").lower() == "true"
SPEECH_CHECK_MIN_RATIO = float(os.getenv("SPEECH_CHECK_MIN_RATIO", "0.1"))  # 10% speech required
SPEECH_CHECK_VAD_THRESHOLD = float(os.getenv("SPEECH_CHECK_VAD_THRESHOLD", "0.5"))
SPEECH_CHECK_FRAME_SAMPLES = 512  # Minimum samples for VAD

# GPU parallelism config - workers per model type
AUDIO_GTCRN_WORKERS = int(os.getenv("AUDIO_GTCRN_WORKERS", "2"))
AUDIO_MPSENET_WORKERS = int(os.getenv("AUDIO_MPSENET_WORKERS", "2"))
AUDIO_SILERO_WORKERS = int(os.getenv("AUDIO_SILERO_WORKERS", "2"))


# ======================================================================================
#  GPU MODEL POOL - Enables true parallel processing with CUDA streams
# ======================================================================================

@dataclass
class ModelReplica:
    """A single model replica with its own CUDA stream."""
    model: Any
    stream: torch.cuda.Stream
    extras: Dict[str, Any]  # Additional resources (e.g., stft_window)


class GPUModelPool:
    """
    Pool of GPU model replicas for parallel processing.
    
    Each replica has its own CUDA stream, enabling true GPU parallelism.
    Work is distributed via a thread-safe queue.
    
    Usage:
        pool = GPUModelPool("noise_reduction", num_replicas=2)
        pool.initialize(model_factory, extras_factory)
        
        # In worker thread:
        with pool.acquire() as replica:
            result = process_with_model(replica.model, replica.stream)
    """
    
    def __init__(self, name: str, num_replicas: int = 2):
        self.name = name
        self.num_replicas = num_replicas
        self.replicas: List[ModelReplica] = []
        self.available: queue.Queue = queue.Queue()
        self.initialized = False
        self._lock = threading.Lock()
        
    def initialize(
        self, 
        model_factory: Callable[[], Any],
        extras_factory: Optional[Callable[[torch.cuda.Stream], Dict[str, Any]]] = None,
        device: torch.device = None
    ) -> bool:
        """
        Initialize model replicas with separate CUDA streams.
        
        Args:
            model_factory: Function that creates and returns a model instance
            extras_factory: Optional function that creates additional resources per replica
            device: Target device (defaults to cuda if available)
            
        Returns:
            True if initialization successful
        """
        if self.initialized:
            return True
            
        with self._lock:
            if self.initialized:
                return True
                
            try:
                target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                use_cuda_streams = target_device.type == "cuda"
                
                for i in range(self.num_replicas):
                    # Create CUDA stream for this replica (or None for CPU)
                    stream = torch.cuda.Stream() if use_cuda_streams else None
                    
                    # Create model instance
                    model = model_factory()
                    if model is None:
                        logger.error(f"[{self.name}] Model factory returned None for replica {i}")
                        continue
                    
                    # Create extras (e.g., pre-allocated windows)
                    extras = extras_factory(stream) if extras_factory else {}
                    
                    replica = ModelReplica(model=model, stream=stream, extras=extras)
                    self.replicas.append(replica)
                    self.available.put(i)
                    
                    logger.info(f"[{self.name}] Replica {i} initialized (stream={stream is not None})")
                
                if not self.replicas:
                    logger.error(f"[{self.name}] No replicas initialized")
                    return False
                    
                self.initialized = True
                logger.info(f"[{self.name}] Pool initialized with {len(self.replicas)} replicas")
                return True
                
            except Exception as e:
                logger.error(f"[{self.name}] Pool initialization failed: {e}")
                return False
    
    @contextmanager
    def acquire(self, timeout: float = 30.0):
        """
        Acquire a model replica for processing.
        
        Blocks until a replica is available or timeout is reached.
        Automatically releases the replica when done.
        
        Args:
            timeout: Maximum time to wait for a replica (seconds)
            
        Yields:
            ModelReplica with model, stream, and extras
            
        Raises:
            RuntimeError: If pool not initialized or timeout reached
        """
        if not self.initialized or not self.replicas:
            raise RuntimeError(f"[{self.name}] Pool not initialized")
        
        try:
            idx = self.available.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"[{self.name}] Timeout waiting for available replica")
        
        try:
            yield self.replicas[idx]
        finally:
            self.available.put(idx)
    
    def is_available(self) -> bool:
        """Check if pool is initialized and has replicas."""
        return self.initialized and len(self.replicas) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "total_replicas": len(self.replicas),
            "available_replicas": self.available.qsize()
        }


# --- Global Model Pools ---
noise_reduction_pool: Optional[GPUModelPool] = None  # GTCRN or MPSENet pool
speech_vad_pool: Optional[GPUModelPool] = None       # Silero VAD pool

# --- Legacy Global State (for backward compatibility) ---
gtcrn_model = None
mpsenet_model = None         # MPSENet noise reduction model
speech_check_vad = None      # Silero VAD for speech check
device = None
stft_window = None
resampler_8k_to_16k = None  # GPU resampler (8kHz → 16kHz)

def _save_audio_files(files: List[Tuple[str, np.ndarray, int]]):
    """Background helper to save audio files. Called by executor."""
    try:
        os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
        for filename, data, sr in files:
            sf.write(os.path.join(DEBUG_AUDIO_DIR, filename), data, sr)
        logger.debug(f"Saved debug audio: {[f[0] for f in files]}")
    except Exception as e:
        logger.warning(f"Failed to save debug audio: {e}")


# ======================================================================================
#  MODEL & RESAMPLER LOADING (with GPU Model Pools)
# ======================================================================================
def initialize_preprocessing():
    """
    Load noise reduction model and speech VAD on GPU.
    
    Uses model pools with CUDA streams for true parallel processing.
    Call once at startup.
    """
    global noise_reduction_pool, speech_vad_pool
    global gtcrn_model, mpsenet_model, device, stft_window, resampler_8k_to_16k
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Audio preprocessing device: {device}")
    logger.info(f"Worker config: GTCRN={AUDIO_GTCRN_WORKERS}, MPSENet={AUDIO_MPSENET_WORKERS}, Silero={AUDIO_SILERO_WORKERS}")
    
    # Load GPU resampler (shared, stateless)
    resampler_8k_to_16k = T.Resample(8000, SAMPLE_RATE).to(device)
    logger.info("✅ GPU resampler loaded (8k→16k)")
    
    # Load noise reduction model pool
    if not NOISE_REDUCTION_ENABLED:
        logger.info("Noise reduction disabled by config")
    else:
        logger.info(f"Noise reduction model: {NOISE_REDUCTION_MODEL}")
        
        if NOISE_REDUCTION_MODEL == "mpsenet":
            noise_reduction_pool = GPUModelPool("noise_reduction", num_replicas=AUDIO_MPSENET_WORKERS)
            _initialize_mpsenet_pool()
        else:
            noise_reduction_pool = GPUModelPool("noise_reduction", num_replicas=AUDIO_GTCRN_WORKERS)
            _initialize_gtcrn_pool()
    
    # Load speech VAD pool
    if SPEECH_CHECK_ENABLED:
        speech_vad_pool = GPUModelPool("speech_vad", num_replicas=AUDIO_SILERO_WORKERS)
        _initialize_speech_vad_pool()


def _initialize_speech_vad_pool():
    """Initialize Silero VAD pool for speech checking."""
    global speech_vad_pool, device, speech_check_vad
    
    try:
        from silero_vad import load_silero_vad
    except ImportError:
        logger.warning("silero_vad not found")
        return
    
    def vad_factory():
        model = load_silero_vad(onnx=False)
        if device.type == "cuda":
            model = model.to(device)
        return model
    
    success = speech_vad_pool.initialize(vad_factory, device=device)
    if success:
        logger.info(f"✅ Speech VAD pool initialized ({AUDIO_SILERO_WORKERS} workers)")
        # Keep legacy reference for backward compatibility
        speech_check_vad = speech_vad_pool.replicas[0].model if speech_vad_pool.replicas else None
    else:
        # Fallback to single model
        logger.warning("VAD pool init failed, falling back to single model")
        speech_check_vad = load_silero_vad(onnx=False)
        if device.type == "cuda":
            speech_check_vad = speech_check_vad.to(device)


def _initialize_gtcrn_pool():
    """Initialize GTCRN noise reduction pool."""
    global noise_reduction_pool, device, gtcrn_model, stft_window
    
    try:
        from gtcrn import GTCRN
    except ImportError:
        logger.warning("GTCRN not found. Install: pip install git+https://github.com/Xiaobin-Rong/gtcrn.git")
        return
    
    if not os.path.exists(GTCRN_MODEL_CHECKPOINT):
        logger.warning(f"GTCRN checkpoint not found: {GTCRN_MODEL_CHECKPOINT}")
        return
    
    # Load checkpoint once
    ckpt = torch.load(GTCRN_MODEL_CHECKPOINT, map_location=device, weights_only=False)
    
    def gtcrn_factory():
        model = GTCRN().to(device).eval()
        model.load_state_dict(ckpt['model'])
        return model
    
    def gtcrn_extras_factory(stream):
        # Each replica gets its own STFT window
        return {"stft_window": torch.hann_window(512).pow(0.5).to(device)}
    
    success = noise_reduction_pool.initialize(gtcrn_factory, gtcrn_extras_factory, device=device)
    if success:
        logger.info(f"✅ GTCRN pool initialized ({AUDIO_GTCRN_WORKERS} workers)")
        # Keep legacy references for backward compatibility
        gtcrn_model = noise_reduction_pool.replicas[0].model if noise_reduction_pool.replicas else None
        stft_window = noise_reduction_pool.replicas[0].extras.get("stft_window") if noise_reduction_pool.replicas else None
    else:
        # Fallback to single model
        logger.warning("GTCRN pool init failed, falling back to single model")
        _initialize_gtcrn_legacy()


def _initialize_gtcrn_legacy():
    """Legacy single-model GTCRN initialization (fallback)."""
    global gtcrn_model, device, stft_window
    
    try:
        from gtcrn import GTCRN
        gtcrn_model = GTCRN().to(device).eval()
        ckpt = torch.load(GTCRN_MODEL_CHECKPOINT, map_location=device, weights_only=False)
        gtcrn_model.load_state_dict(ckpt['model'])
        stft_window = torch.hann_window(512).pow(0.5).to(device)
        logger.info(f"✅ GTCRN loaded (legacy single model)")
    except Exception as e:
        logger.error(f"Failed to load GTCRN: {e}")
        gtcrn_model = None


def _initialize_mpsenet_pool():
    """Initialize MPSENet noise reduction pool."""
    global noise_reduction_pool, device, mpsenet_model
    
    try:
        from MPSENet import MPSENet
    except ImportError:
        logger.warning("MPSENet not found. Install: pip install MPSENet")
        return
    
    def mpsenet_factory():
        model = MPSENet.from_pretrained(MPSENET_MODEL_NAME).to(device)
        return model
    
    success = noise_reduction_pool.initialize(mpsenet_factory, device=device)
    if success:
        logger.info(f"✅ MPSENet pool initialized ({AUDIO_MPSENET_WORKERS} workers)")
        # Keep legacy reference for backward compatibility
        mpsenet_model = noise_reduction_pool.replicas[0].model if noise_reduction_pool.replicas else None
    else:
        # Fallback to single model
        logger.warning("MPSENet pool init failed, falling back to single model")
        _initialize_mpsenet_legacy()


def _initialize_mpsenet_legacy():
    """Legacy single-model MPSENet initialization (fallback)."""
    global mpsenet_model, device
    
    try:
        from MPSENet import MPSENet
        mpsenet_model = MPSENet.from_pretrained(MPSENET_MODEL_NAME).to(device)
        logger.info(f"✅ MPSENet loaded (legacy single model)")
    except Exception as e:
        logger.error(f"Failed to load MPSENet: {e}")
        mpsenet_model = None


# ======================================================================================
#  PREPROCESSING FUNCTIONS (GPU-accelerated)
# ======================================================================================

# --- Internal tensor-based functions (no CPU-GPU transfers) ---

def _convert_mulaw_8k_to_tensor_16k(audio_bytes: bytes) -> torch.Tensor:
    """Convert mulaw 8kHz bytes to float32 tensor at 16kHz on GPU."""
    global resampler_8k_to_16k, device
    
    # Decode mulaw to linear PCM (CPU - audioop is fast)
    linear_pcm_8k = audioop.ulaw2lin(audio_bytes, 2)
    audio_8k = np.frombuffer(linear_pcm_8k, dtype=np.int16)
    
    # Convert to float32 tensor and resample on GPU
    audio_tensor = torch.from_numpy(audio_8k.astype(np.float32)).to(device)
    resampled = resampler_8k_to_16k(audio_tensor.unsqueeze(0)).squeeze(0)
    return resampled


def _bytes_to_tensor(audio_bytes: bytes) -> torch.Tensor:
    """Convert PCM int16 bytes to float32 tensor on GPU."""
    global device
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return torch.from_numpy(audio_int16.astype(np.float32)).to(device)


def _tensor_to_bytes(audio_tensor: torch.Tensor) -> bytes:
    """Convert float32 tensor to PCM int16 bytes."""
    audio_clipped = torch.clamp(audio_tensor, -32768, 32767)
    return audio_clipped.cpu().to(torch.int16).numpy().tobytes()


def _normalize_audio_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize audio tensor to target RMS with gain limiting (stays on GPU)."""
    # Calculate RMS on GPU
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    
    if rms < 1.0:
        return audio_tensor
    
    # Apply gain
    gain = min(TARGET_RMS / rms.item(), MAX_GAIN_LINEAR)
    return audio_tensor * gain


def _apply_gtcrn_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply GTCRN noise reduction on tensor using model pool.
    
    Uses CUDA streams for true parallel processing across multiple clients.
    """
    global noise_reduction_pool, gtcrn_model, stft_window
    
    # Try to use pool first
    if noise_reduction_pool is not None and noise_reduction_pool.is_available():
        return _apply_gtcrn_with_pool(audio_tensor)
    
    # Fallback to legacy single model
    return _apply_gtcrn_legacy_impl(audio_tensor, gtcrn_model, stft_window)


def _apply_gtcrn_with_pool(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Apply GTCRN using model pool with CUDA stream."""
    global noise_reduction_pool
    
    with noise_reduction_pool.acquire() as replica:
        model = replica.model
        stream = replica.stream
        window = replica.extras.get("stft_window", stft_window)
        
        if stream is not None:
            # Process in dedicated CUDA stream for parallel execution
            with torch.cuda.stream(stream):
                result = _apply_gtcrn_legacy_impl(audio_tensor, model, window)
            # Synchronize this stream before returning
            stream.synchronize()
            return result
        else:
            # CPU fallback
            return _apply_gtcrn_legacy_impl(audio_tensor, model, window)


def _apply_gtcrn_legacy_impl(audio_tensor: torch.Tensor, model: Any, window: torch.Tensor) -> torch.Tensor:
    """Core GTCRN implementation (used by both pool and legacy paths)."""
    # Normalize to [-1, 1] range
    audio_float = audio_tensor / 32768.0
    
    # STFT (GTCRN parameters: n_fft=512, hop=256, win=512)
    stft_complex = torch.stft(
        audio_float,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=window,
        return_complex=True
    )
    
    # Convert to real/imag format for GTCRN [batch, freq, time, 2]
    stft_input = torch.stack([stft_complex.real, stft_complex.imag], dim=-1).unsqueeze(0)
    
    # Run GTCRN denoising
    with torch.no_grad():
        stft_output = model(stft_input)[0]
    
    # Convert back to complex and apply iSTFT
    stft_complex_output = torch.complex(stft_output[..., 0], stft_output[..., 1])
    enhanced = torch.istft(
        stft_complex_output,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=window,
        return_complex=False
    )
    
    # Scale back to int16 range
    return enhanced * 32768.0


def _apply_mpsenet_tensor(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply MPSENet noise reduction on tensor using model pool.
    
    Uses CUDA streams for true parallel processing across multiple clients.
    """
    global noise_reduction_pool, mpsenet_model
    
    # Try to use pool first
    if noise_reduction_pool is not None and noise_reduction_pool.is_available():
        return _apply_mpsenet_with_pool(audio_tensor)
    
    # Fallback to legacy single model
    return _apply_mpsenet_legacy_impl(audio_tensor, mpsenet_model)


def _apply_mpsenet_with_pool(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Apply MPSENet using model pool with CUDA stream."""
    global noise_reduction_pool
    
    with noise_reduction_pool.acquire() as replica:
        model = replica.model
        stream = replica.stream
        
        if stream is not None:
            # Process in dedicated CUDA stream for parallel execution
            with torch.cuda.stream(stream):
                result = _apply_mpsenet_legacy_impl(audio_tensor, model)
            # Synchronize this stream before returning
            stream.synchronize()
            return result
        else:
            # CPU fallback
            return _apply_mpsenet_legacy_impl(audio_tensor, model)


def _apply_mpsenet_legacy_impl(audio_tensor: torch.Tensor, model: Any) -> torch.Tensor:
    """Core MPSENet implementation (used by both pool and legacy paths)."""
    global device
    
    # Normalize to [-1, 1] range for MPSENet
    audio_float = (audio_tensor / 32768.0).cpu().numpy()
    
    # MPSENet processes numpy arrays
    with torch.no_grad():
        enhanced, sr, notation = model(audio_float, segment_size=MPSENET_SEGMENT_SIZE)
    
    # Debug save in background (non-blocking)
    # if DEBUG_SAVE_AUDIO:
    #     timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    #     _audio_save_executor.submit(_save_audio_files, [
    #         (f"{timestamp}_mpsenet_before.wav", audio_float.copy(), SAMPLE_RATE),
    #         (f"{timestamp}_mpsenet_after.wav", enhanced.copy(), sr),
    #     ])
    
    # Convert back to tensor in int16 range
    return torch.from_numpy(enhanced * 32768.0).to(device)


# --- Backward-compatible byte-based functions ---

def convert_mulaw_8k_to_pcm_16k_gpu(audio_bytes: bytes) -> bytes:
    """Convert mulaw 8kHz to PCM 16kHz using GPU resampling. Returns PCM int16 bytes."""
    if not audio_bytes:
        return audio_bytes
    
    try:
        audio_tensor = _convert_mulaw_8k_to_tensor_16k(audio_bytes)
        return _tensor_to_bytes(audio_tensor)
    except Exception as e:
        logger.error(f"Mulaw conversion failed: {e}")
        return audio_bytes


def apply_noise_reduction_gpu(audio_bytes: bytes) -> bytes:
    """Apply noise reduction using configured model (GTCRN or MPSENet)."""
    global gtcrn_model, mpsenet_model
    
    if not audio_bytes:
        return audio_bytes
    
    try:
        audio_tensor = _bytes_to_tensor(audio_bytes)

        # Use MPSENet if available
        if mpsenet_model is not None:
            enhanced_tensor = _apply_mpsenet_tensor(audio_tensor)
        # Fall back to GTCRN
        elif gtcrn_model is not None:
            enhanced_tensor = _apply_gtcrn_tensor(audio_tensor)
        else:
            return audio_bytes
        
        return _tensor_to_bytes(enhanced_tensor)
        
    except Exception as e:
        logger.error(f"Noise reduction failed: {e}")
        return audio_bytes


def check_is_speech(audio_bytes: bytes) -> bool:
    """
    Check if audio contains speech (GPU-accelerated with model pool).
    
    Args:
        audio_bytes: PCM 16kHz audio bytes
        
    Returns:
        True if audio contains speech above threshold
    """
    global speech_vad_pool, device, speech_check_vad
    
    if not SPEECH_CHECK_ENABLED:
        return True
    
    try:
        samples_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_tensor = torch.from_numpy(samples_int16.astype(np.float32) / 32768.0).to(device)
        return _check_speech_with_pool(audio_tensor)
    except Exception as e:
        logger.error(f"Speech check failed: {e}")
        return True  # Allow on error


def _check_speech_with_pool(audio_tensor: torch.Tensor) -> bool:
    """Check speech presence using VAD pool with CUDA stream."""
    global speech_vad_pool, speech_check_vad
    
    if not SPEECH_CHECK_ENABLED:
        return True
    
    num_samples = audio_tensor.shape[0]
    if num_samples < SPEECH_CHECK_FRAME_SAMPLES:
        return True
    
    # Try pool first
    if speech_vad_pool is not None and speech_vad_pool.is_available():
        try:
            with speech_vad_pool.acquire(timeout=5.0) as replica:
                vad_model = replica.model
                stream = replica.stream
                
                if stream is not None:
                    with torch.cuda.stream(stream):
                        result = _check_speech_impl(audio_tensor, vad_model)
                    stream.synchronize()
                    return result
                else:
                    return _check_speech_impl(audio_tensor, vad_model)
        except Exception as e:
            logger.warning(f"VAD pool error, falling back: {e}")
    
    # Fallback to legacy
    if speech_check_vad is not None:
        return _check_speech_impl(audio_tensor, speech_check_vad)
    
    return True


def _check_speech_impl(audio_tensor: torch.Tensor, vad_model: Any) -> bool:
    """Core speech check implementation."""
    from silero_vad import get_speech_timestamps
    
    num_samples = audio_tensor.shape[0]
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            threshold=SPEECH_CHECK_VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
            return_seconds=False,
        )
    speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
    speech_ratio = speech_samples / num_samples
    return speech_ratio >= SPEECH_CHECK_MIN_RATIO


def preprocess_audio(
    audio_bytes: bytes,
    input_encoding: str = "pcm_16k",
    apply_noise_reduction: bool = False,
    apply_normalization: bool = True
) -> bytes:
    """
    Preprocess audio with format conversion, noise reduction, and normalization.
    
    Optimized pipeline: converts to tensor once, processes on GPU, converts back once.
    
    Args:
        audio_bytes: Raw audio bytes
        input_encoding: "mulaw_8k" or "pcm_16k" (default: pcm_16k)
        apply_noise_reduction: Whether to apply noise reduction (default: False)
        apply_normalization: Whether to normalize audio (default: True)
    
    Returns:
        Preprocessed PCM 16kHz int16 audio bytes
    """
    if not audio_bytes:
        return audio_bytes
    
    try:
        # Step 1: Convert to tensor (single conversion)
        if input_encoding == "mulaw_8k":
            audio_tensor = _convert_mulaw_8k_to_tensor_16k(audio_bytes)
        else:
            audio_tensor = _bytes_to_tensor(audio_bytes)
        
        # Step 2: Noise reduction (tensor stays on GPU)
        if apply_noise_reduction:
            bg = time.time()
            if mpsenet_model is not None:
                audio_tensor = _apply_mpsenet_tensor(audio_tensor)
                logger.debug(f"MPSENet noise reduction time: {time.time() - bg} seconds")
            elif gtcrn_model is not None:
                audio_tensor = _apply_gtcrn_tensor(audio_tensor)
                logger.debug(f"GTCRN noise reduction time: {time.time() - bg} seconds")
        
        # Step 3: Normalization (tensor stays on GPU)
        if apply_normalization:
            audio_tensor = _normalize_audio_tensor(audio_tensor)
        
        # Step 4: Convert back to bytes (single conversion)
        return _tensor_to_bytes(audio_tensor)
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return audio_bytes

def tensor_to_bytes(audio_tensor: torch.Tensor) -> bytes:
    """Convert float32 audio tensor [-1, 1] to PCM int16 bytes."""
    audio_int16 = (audio_tensor.cpu().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


# ======================================================================================
#  POOL MONITORING & UTILITIES
# ======================================================================================

def get_pool_stats() -> Dict[str, Any]:
    """Get statistics for all GPU model pools."""
    global noise_reduction_pool, speech_vad_pool
    
    stats = {}
    
    if noise_reduction_pool is not None:
        stats["noise_reduction"] = noise_reduction_pool.get_stats()
    
    if speech_vad_pool is not None:
        stats["speech_vad"] = speech_vad_pool.get_stats()
    
    stats["config"] = {
        "gtcrn_workers": AUDIO_GTCRN_WORKERS,
        "mpsenet_workers": AUDIO_MPSENET_WORKERS,
        "silero_workers": AUDIO_SILERO_WORKERS,
        "noise_reduction_enabled": NOISE_REDUCTION_ENABLED,
        "noise_reduction_model": NOISE_REDUCTION_MODEL,
        "speech_check_enabled": SPEECH_CHECK_ENABLED,
    }
    
    return stats


def is_pool_healthy() -> bool:
    """Check if all enabled pools are healthy and have available replicas."""
    global noise_reduction_pool, speech_vad_pool
    
    if NOISE_REDUCTION_ENABLED:
        if noise_reduction_pool is None or not noise_reduction_pool.is_available():
            if gtcrn_model is None and mpsenet_model is None:
                return False
    
    if SPEECH_CHECK_ENABLED:
        if speech_vad_pool is None or not speech_vad_pool.is_available():
            if speech_check_vad is None:
                return False
    
    return True
