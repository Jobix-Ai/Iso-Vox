# Moonshine ASR Worker App - Multi-instance, no batching
# Lightweight ONNX-based ASR worker with parallel model instances

import os
import sys
import site
import ctypes

# Preload NVIDIA CUDA libraries for onnxruntime-gpu BEFORE any ONNX imports
def _preload_cuda_libraries():
    """Preload NVIDIA pip package libraries for onnxruntime-gpu using ctypes"""
    nvidia_base = None
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia")
        if os.path.isdir(candidate):
            nvidia_base = candidate
            break
    
    if nvidia_base is None:
        return
    
    libs_to_load = [
        ("cuda_runtime", "lib", "libcudart.so.12"),
        ("cublas", "lib", "libcublasLt.so.12"),
        ("cublas", "lib", "libcublas.so.12"),
        ("cufft", "lib", "libcufft.so.11"),
        ("curand", "lib", "libcurand.so.10"),
        ("cusolver", "lib", "libcusolver.so.11"),
        ("cusparse", "lib", "libcusparse.so.12"),
        ("cudnn", "lib", "libcudnn.so.9"),
    ]
    
    for subdir, libdir, libname in libs_to_load:
        lib_path = os.path.join(nvidia_base, subdir, libdir, libname)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

_preload_cuda_libraries()

import logging
import asyncio
import base64
import time
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

# --- Load Environment Variables ---
load_dotenv()

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()

# --- Model Configuration ---
MODEL_NAME = os.getenv("MOONSHINE_MODEL_NAME", "moonshine/base")
SAMPLE_RATE = 16000
NUM_MODEL_INSTANCES = int(os.getenv("ASR_MOONSHINE_NUM_WORKERS", "4"))

# --- Global Resources ---
model_pool: Queue = Queue()  # Pool of model instances
model_executor: Optional[ThreadPoolExecutor] = None


# --- Request Body Model ---
class BatchTranscriptionRequest(BaseModel):
    audio_batch: List[str]  # List of base64 encoded audio bytes


# ======================================================================================
#  CORE TRANSCRIPTION FUNCTION (Single audio)
# ======================================================================================
def _transcribe_single(audio_b64: str) -> Optional[Dict[str, Any]]:
    """Process a single audio sample through a Moonshine model from the pool"""
    # Get a model instance from the pool (with timeout to avoid infinite blocking)
    try:
        model, tok = model_pool.get(timeout=3)  # Wait up to 10 seconds for a model
    except Exception as e:
        logger.error(f"Timeout waiting for model instance: {e}")
        raise RuntimeError("No model instance available within timeout")
    
    try:
        # Decode base64 audio
        audio_np = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
        
        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)
        
        # Add batch dimension if needed
        if audio_np.ndim == 1:
            audio_input = audio_np[np.newaxis, :]
        else:
            audio_input = audio_np
        
        # Generate transcription
        tokens = model.generate(audio_input.astype(np.float32))
        text = tok.decode_batch(tokens)[0].strip()
        
        if text:
            return {
                "text": text,
                "word_confidence": [],
                "avg_confidence": 0.8,
            }
        return None
        
    except Exception as e:
        logger.error(f"Failed to transcribe audio sample: {e}", exc_info=True)
        return None
    finally:
        # Return model instance to the pool
        model_pool.put((model, tok))


def _transcribe_batch_parallel(batch_audio_b64: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Process a batch by distributing individual items across model instances"""
    with ThreadPoolExecutor(max_workers=min(len(batch_audio_b64), NUM_MODEL_INSTANCES)) as executor:
        results = list(executor.map(_transcribe_single, batch_audio_b64))
    return results


# ======================================================================================
#  FASTAPI LIFECYCLE & ENDPOINT
# ======================================================================================
@app.on_event("startup")
async def startup_event():
    global model_executor
    import onnxruntime
    
    logger.info(f"🚀 Moonshine Worker starting up with {NUM_MODEL_INSTANCES} model instances...")
    
    # Check for GPU support
    available_providers = onnxruntime.get_available_providers()
    use_gpu = 'CUDAExecutionProvider' in available_providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    logger.info(f"Available ONNX providers: {available_providers}")
    logger.info(f"Using providers: {providers}")
    
    try:
        # Load shared tokenizer (thread-safe for decoding)
        shared_tokenizer = load_tokenizer()
        logger.info("✅ Shared tokenizer loaded")
        
        # Load multiple model instances
        for i in range(NUM_MODEL_INSTANCES):
            model = MoonshineOnnxModel(model_name=MODEL_NAME)
            
            # Re-create sessions with GPU support if available
            if use_gpu:
                encoder_path = model.encoder._model_path
                decoder_path = model.decoder._model_path
                model.encoder = onnxruntime.InferenceSession(encoder_path, providers=providers)
                model.decoder = onnxruntime.InferenceSession(decoder_path, providers=providers)
            
            model_pool.put((model, shared_tokenizer))
            logger.info(f"✅ Moonshine model instance {i+1}/{NUM_MODEL_INSTANCES} loaded")
        
        # Warmup all instances
        logger.info("🔥 Warming up Moonshine model instances...")
        warmup_tasks = []
        for _ in range(NUM_MODEL_INSTANCES):
            dummy_audio = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.float32)
            dummy_b64 = base64.b64encode(dummy_audio.tobytes()).decode('utf-8')
            warmup_tasks.append(dummy_b64)
        
        _transcribe_batch_parallel(warmup_tasks)
        logger.info("✅ Moonshine model instances warmed up.")
        
        # Create thread pool for handling requests
        model_executor = ThreadPoolExecutor(max_workers=NUM_MODEL_INSTANCES, thread_name_prefix="moonshine_worker")
        
    except Exception as e:
        logger.error(f"❌ Failed to load Moonshine models: {e}", exc_info=True)
    
    logger.info("🎉 MOONSHINE WORKER READY 🎉")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down model executor...")
    if model_executor:
        model_executor.shutdown(wait=True)


@app.post("/transcribe")
async def transcribe_single_endpoint(request: Request):
    """Single audio transcription endpoint - no batching"""
    start_time = time.perf_counter()
    
    audio_bytes = await request.body()
    if not audio_bytes:
        return Response(status_code=204)
    
    # Convert to float32 if coming as int16
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_b64 = base64.b64encode(audio_np.astype(np.float32).tobytes()).decode('utf-8')
    
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(model_executor, _transcribe_single, audio_b64)
    except RuntimeError as e:
        logger.error(f"Model instance unavailable: {e}")
        return Response(content="All model instances are busy. Please try again.", status_code=503)
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return Response(content="Transcription failed.", status_code=500)
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    logger.info(f"Moonshine transcription processed in {processing_time_ms:.2f} ms.")
    
    if not result:
        return Response(status_code=204)
    return result


@app.post("/batch_transcribe")
async def transcribe_batch_endpoint(request: BatchTranscriptionRequest):
    """Batch endpoint - processes in parallel across model instances (for compatibility)"""
    start_time = time.perf_counter()

    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            model_executor, 
            _transcribe_batch_parallel, 
            request.audio_batch
        )
    except RuntimeError as e:
        logger.error(f"Model instances unavailable: {e}")
        return Response(content="All model instances are busy. Please try again.", status_code=503)
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}", exc_info=True)
        return Response(content="Batch transcription failed.", status_code=500)
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000
    logger.info(f"Moonshine parallel transcription for {len(request.audio_batch)} items in {processing_time_ms:.2f} ms.")
    
    return results


@app.get("/health")
async def health_check():
    available = model_pool.qsize()
    return {
        "status": "healthy" if model_executor is not None else "unhealthy",
        "model_name": MODEL_NAME,
        "num_instances": NUM_MODEL_INSTANCES,
        "available_instances": available,
        "busy_instances": NUM_MODEL_INSTANCES - available,
        "sample_rate": SAMPLE_RATE,
        "backend": "ONNX",
        "batch_inference": False,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)