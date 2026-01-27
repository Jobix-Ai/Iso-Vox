"""
Smart-Turn EOS Detection Worker with Batch Inference (Optimized)
Similar architecture to batching_worker_app.py but for end-of-speech detection
"""

import logging
import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ONNX_MODEL_PATH = "weights/smart-turn-v3.1-gpu.onnx"
MAX_BATCH_SIZE = int(os.getenv("SMART_TURN_BATCH_SIZE", "16"))  # Increased default
BATCH_TIMEOUT_S = float(os.getenv("SMART_TURN_BATCH_TIMEOUT_S", "0.010"))  # 10ms default (reduced)
NUM_PREPROCESSING_THREADS = int(os.getenv("NUM_PREPROCESSING_THREADS", "4"))

# --- Global Queue and Thread Pool ---
request_queue: Optional[asyncio.Queue[Tuple[np.ndarray, Future]]] = None
preprocessing_executor: Optional[ThreadPoolExecutor] = None

# --- Initialize Model ---
def build_session(onnx_path):
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 2  # Increased for better CPU utilization
    so.intra_op_num_threads = 2  # Added for better parallel execution
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Use GPU if available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

logger.info(f"Loading Smart-Turn ONNX model from {ONNX_MODEL_PATH}...")
feature_extractor = WhisperFeatureExtractor(chunk_length=8)
session = build_session(ONNX_MODEL_PATH)
logger.info("Smart-Turn model loaded successfully")

# --- Pre-computed Constants ---
MAX_SAMPLES = 8 * 16000  # 8 seconds at 16kHz
SAMPLE_RATE = 16000
INT16_TO_FLOAT32_FACTOR = np.float32(1.0 / 32768.0)  # Pre-computed constant

# --- Preprocessing (Optimized) ---
def truncate_audio_to_last_n_seconds_fast(audio_array: np.ndarray) -> np.ndarray:
    """Optimized: Truncate audio to last 8 seconds or pad with zeros."""
    arr_len = len(audio_array)
    if arr_len > MAX_SAMPLES:
        # Slice directly, avoid copy when possible
        return audio_array[-MAX_SAMPLES:]
    elif arr_len < MAX_SAMPLES:
        # Pre-allocate and copy (faster than np.pad for this use case)
        result = np.zeros(MAX_SAMPLES, dtype=audio_array.dtype)
        result[-arr_len:] = audio_array
        return result
    return audio_array

def preprocess_audio_batch(audio_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Optimized: Preprocess multiple audio arrays at once using batch feature extraction.
    This is significantly faster than processing each audio individually.
    """
    if not audio_arrays:
        return np.array([])
    
    # Truncate/pad all audio arrays
    processed_audios = [truncate_audio_to_last_n_seconds_fast(audio) for audio in audio_arrays]
    
    # Batch feature extraction (much faster than individual calls)
    inputs = feature_extractor(
        processed_audios,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    
    # Return as float32 batch (already in correct shape from feature_extractor)
    return inputs.input_features.astype(np.float32)

def batch_inference(batch_features: np.ndarray, batch_size: int) -> List[Dict]:
    """
    Optimized: Run batched inference on preprocessed features.
    batch_features should already be stacked and ready for inference.
    """
    if batch_features.size == 0:
        return []
    
    try:
        # Run inference (features are already preprocessed and batched)
        start_time = time.time()
        outputs = session.run(None, {"input_features": batch_features})
        inference_time = time.time() - start_time
        
        # Extract results - optimized probability extraction
        probabilities = outputs[0].squeeze()
        if batch_size == 1:
            probabilities = [probabilities.item()]
        else:
            probabilities = probabilities.tolist()
        
        # Vectorized prediction (faster than loop)
        probs_array = np.array(probabilities, dtype=np.float32)
        predictions = (probs_array > 0.5).astype(np.int32)
        
        # Build results
        results = [
            {
                "prediction": int(pred),
                "probability": float(prob)
            }
            for pred, prob in zip(predictions, probabilities)
        ]
        
        logger.info(f"Batch inference: {batch_size} samples in {inference_time*1000:.2f}ms "
                   f"({inference_time*1000/batch_size:.2f}ms per sample)")
        
        return results
    except Exception as e:
        logger.error(f"Batch inference error: {e}", exc_info=True)
        return [{"prediction": 1, "probability": 1.0, "error": str(e)} for _ in range(batch_size)]

# ======================================================================================
#  BATCHING WORKER (Optimized)
# ======================================================================================
async def batch_predict_worker(queue: asyncio.Queue):
    """
    Optimized: Continuously pulls requests from the queue, batches them, and processes predictions.
    Uses batch preprocessing for significant performance improvements.
    """
    global BATCH_TIMEOUT_S, MAX_BATCH_SIZE, preprocessing_executor
    while True:
        requests: List[np.ndarray] = []
        futures: List[Future] = []
        try:
            # Wait for the first request to start a new batch
            first_audio, first_future = await queue.get()
            requests.append(first_audio)
            futures.append(first_future)

            # Collect more requests for a short period to form a batch
            # Use perf_counter for higher precision timing
            start_time = time.perf_counter()
            while (
                len(requests) < MAX_BATCH_SIZE and
                (time.perf_counter() - start_time) < BATCH_TIMEOUT_S and
                not queue.empty()
            ):
                try:
                    audio, future = queue.get_nowait()
                    requests.append(audio)
                    futures.append(future)
                except asyncio.QueueEmpty:
                    break
            
            batch_size = len(requests)
            logger.info(f"Processing batch of size {batch_size}/{MAX_BATCH_SIZE}.")

            # Optimized: Batch preprocess all audio samples at once (much faster)
            loop = asyncio.get_running_loop()
            preprocessed_batch = await loop.run_in_executor(
                preprocessing_executor,
                preprocess_audio_batch,
                requests
            )
            
            # Run batch inference
            results = await loop.run_in_executor(
                preprocessing_executor,
                batch_inference,
                preprocessed_batch,
                batch_size
            )

            # Distribute the results back to the waiting request handlers
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            logger.error(f"Error in batching worker: {e}", exc_info=True)
            for future in futures:
                if not future.done():
                    # Return fallback result on error (finalize)
                    future.set_result({"prediction": 1, "probability": 1.0, "error": str(e)})

# ======================================================================================
#  MONITORING WORKER
# ======================================================================================
async def log_queue_size_worker(queue: asyncio.Queue):
    """Periodically logs the size of the request queue to monitor demand."""
    while True:
        await asyncio.sleep(5)
        logger.info(f"MONITOR: Current smart-turn queue size: {queue.qsize()}")

# ======================================================================================
#  FASTAPI LIFECYCLE & ENDPOINT
# ======================================================================================
app = FastAPI(title="Smart-Turn EOS Detection Worker")

@app.on_event("startup")
async def startup_event():
    global request_queue, preprocessing_executor
    logger.info(f"🚀 Smart-Turn Worker starting (Optimized)...")
    logger.info(f"   Max batch size: {MAX_BATCH_SIZE}")
    logger.info(f"   Batch timeout: {BATCH_TIMEOUT_S*1000:.1f}ms")
    logger.info(f"   Preprocessing threads: {NUM_PREPROCESSING_THREADS}")
    
    # Initialize thread pool for CPU-bound preprocessing tasks
    preprocessing_executor = ThreadPoolExecutor(
        max_workers=NUM_PREPROCESSING_THREADS,
        thread_name_prefix="preproc"
    )
    
    request_queue = asyncio.Queue()
    asyncio.create_task(batch_predict_worker(request_queue))
    asyncio.create_task(log_queue_size_worker(request_queue))
    
    logger.info("🎉 SMART-TURN WORKER READY (OPTIMIZED) 🎉")

@app.on_event("shutdown")
async def shutdown_event():
    global preprocessing_executor
    logger.info("Shutting down Smart-Turn worker...")
    if preprocessing_executor:
        preprocessing_executor.shutdown(wait=True)
        logger.info("Preprocessing executor shut down.")

@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    Optimized: Endpoint to predict end-of-speech.
    Expects raw audio bytes (int16 PCM, 16kHz) in request body.
    Returns JSON with prediction and probability.
    """
    client_id = request.headers.get("X-Client-ID", "unknown")
    
    try:
        # Read audio data from request body
        audio_bytes = await request.body()
        
        if not audio_bytes:
            logger.warning(f"[{client_id}] Received empty audio body.")
            return JSONResponse(
                status_code=400,
                content={"error": "Empty audio data"}
            )
        
        # Optimized: Convert bytes to float32 numpy array using in-place operation
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        # Use pre-computed constant and multiply instead of divide (faster)
        audio_float32 = audio_int16.astype(np.float32) * INT16_TO_FLOAT32_FACTOR
        
        if not request_queue:
            logger.error("Request received but worker is not ready (queue not initialized).")
            return Response(content="Server is not ready.", status_code=503)
        
        # Create a future for this request
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Add to queue
        await request_queue.put((audio_float32, future))
        
        # Wait for result with timeout
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
        except asyncio.TimeoutError:
            logger.error(f"[{client_id}] Request timed out waiting for prediction result.")
            return JSONResponse(
                status_code=504,
                content={"error": "Request timed out", "prediction": 1, "probability": 1.0}
            )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[{client_id}] Error in predict endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "prediction": 1, "probability": 1.0}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "smart-turn-eos"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

