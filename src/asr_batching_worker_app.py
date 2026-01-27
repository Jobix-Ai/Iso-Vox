# ASR Batching Worker App
# Routes requests to appropriate ASR workers:
# - Parakeet: batched requests for long audio
# - Moonshine: direct (unbatched) requests for short audio

import logging
import os
import asyncio
import time
import base64
import itertools
from concurrent.futures import Future
from dotenv import load_dotenv
import numpy as np
from fastapi import FastAPI, Request, Response
from typing import List, Optional, Tuple
import httpx

# --- Load Environment Variables ---
load_dotenv()

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = FastAPI()

# --- Batching Configuration ---
MAX_BATCH_SIZE = int(os.getenv("ASR_MAX_BATCH_SIZE", "64"))
BATCH_TIMEOUT_S = float(os.getenv("ASR_BATCH_TIMEOUT_S", "0.01"))  # 10ms

# --- Audio Routing Configuration ---
SAMPLE_RATE = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
SHORT_DURATION_MS = int(os.getenv("ASR_SHORT_DURATION_MS", "3000"))  # 3 seconds default

# --- Worker URLs ---
# Parakeet worker (for long audio, batched)
PARAKEET_WORKER_URL = os.getenv("ASR_PARAKEET_WORKER_URL", "http://127.0.0.1:8093")
logger.info(f"Parakeet worker URL: {PARAKEET_WORKER_URL}")

# Moonshine worker (for short audio, direct/unbatched)
MOONSHINE_WORKER_URL = os.getenv("ASR_MOONSHINE_WORKER_URL", "http://127.0.0.1:8094")
logger.info(f"Moonshine worker URL: {MOONSHINE_WORKER_URL}")

# --- Global Resources ---
parakeet_queue: Optional[asyncio.Queue[Tuple[np.ndarray, Future]]] = None
http_client: Optional[httpx.AsyncClient] = None


# ======================================================================================
#  PARAKEET BATCHING WORKER (for long audio)
# ======================================================================================
async def parakeet_batch_worker(queue: asyncio.Queue):
    """Continuously pulls requests from the queue, batches them, and sends to Parakeet worker."""
    global BATCH_TIMEOUT_S, MAX_BATCH_SIZE
    while True:
        requests: List[np.ndarray] = []
        futures: List[Future] = []
        try:
            # Wait for the first request to start a new batch
            first_audio, first_future = await queue.get()
            requests.append(first_audio)
            futures.append(first_future)

            # Collect more requests for a short period to form a batch
            start_time = time.time()
            while (
                len(requests) < MAX_BATCH_SIZE and
                (time.time() - start_time) < BATCH_TIMEOUT_S and
                not queue.empty()
            ):
                try:
                    audio, future = queue.get_nowait()
                    requests.append(audio)
                    futures.append(future)
                except asyncio.QueueEmpty:
                    break
            
            logger.info(f"[PARAKEET] Processing batch of size {len(requests)}/{MAX_BATCH_SIZE}")

            # Prepare payload for Parakeet worker
            audio_payload = [base64.b64encode(audio.tobytes()).decode('utf-8') for audio in requests]
            full_url = f"{PARAKEET_WORKER_URL}/batch_transcribe"
            
            try:
                response = await http_client.post(full_url, json={"audio_batch": audio_payload}, timeout=30.0)
                response.raise_for_status()
                results = response.json()

                # Distribute the results back to the waiting request handlers
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)
            except httpx.HTTPError as e:
                logger.error(f"[PARAKEET] HTTP request failed: {e}")
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
            except Exception as e:
                logger.error(f"[PARAKEET] Error processing response: {e}", exc_info=True)
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

        except Exception as e:
            logger.error(f"[PARAKEET] Error in batching worker: {e}", exc_info=True)
            for future in futures:
                if not future.done():
                    future.set_exception(e)


# ======================================================================================
#  MOONSHINE DIRECT REQUEST (for short audio, no batching)
# ======================================================================================
async def transcribe_moonshine_direct(audio_bytes: bytes) -> Optional[dict]:
    """Send audio directly to Moonshine worker without batching."""
    full_url = f"{MOONSHINE_WORKER_URL}/transcribe"
    
    try:
        response = await http_client.post(
            full_url,
            content=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10.0
        )
        if response.status_code == 204:
            return None
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"[MOONSHINE] HTTP request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"[MOONSHINE] Error: {e}", exc_info=True)
        raise


# ======================================================================================
#  MONITORING WORKER
# ======================================================================================
async def log_queue_size_worker(queue: asyncio.Queue):
    """Periodically logs the size of the request queue to monitor demand."""
    while True:
        await asyncio.sleep(5)
        logger.info(f"MONITOR: Parakeet queue size: {queue.qsize()}")


# ======================================================================================
#  FASTAPI LIFECYCLE & ENDPOINT
# ======================================================================================
@app.on_event("startup")
async def startup_event():
    global parakeet_queue, http_client
    logger.info("🚀 Batching Worker starting up...")
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=60.0, 
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=50)
    )
    
    # Create queue for Parakeet (batched long audio)
    parakeet_queue = asyncio.Queue()
    
    # Start Parakeet batching worker
    asyncio.create_task(parakeet_batch_worker(parakeet_queue))
    
    # Start monitoring worker
    asyncio.create_task(log_queue_size_worker(parakeet_queue))
    
    logger.info(f"🔥 Parakeet batching worker started (max_batch_size={MAX_BATCH_SIZE}, timeout={BATCH_TIMEOUT_S}s)")
    logger.info(f"📊 Short audio threshold: {SHORT_DURATION_MS}ms @ {SAMPLE_RATE}Hz")
    logger.info(f"🎯 Short audio (Moonshine, direct): {MOONSHINE_WORKER_URL}")
    logger.info(f"🎯 Long audio (Parakeet, batched): {PARAKEET_WORKER_URL}")
    logger.info("🎉 BATCHING WORKER READY 🎉")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down HTTP client...")
    if http_client:
        await http_client.aclose()


@app.post("/transcribe")
async def transcribe_endpoint(request: Request):
    client_id = request.headers.get("X-Client-ID", "unknown_client")
    
    audio_bytes = await request.body()
    
    if not audio_bytes:
        logger.warning(f"[{client_id}] Received empty audio body.")
        return Response(status_code=204)

    audio_np_single = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Calculate audio duration in milliseconds
    audio_duration_ms = (len(audio_np_single) / SAMPLE_RATE) * 1000
    
    # Route based on duration
    if audio_duration_ms < SHORT_DURATION_MS:
        # Short audio -> Moonshine (direct, no batching)
        logger.debug(f"[{client_id}] Audio duration: {audio_duration_ms:.1f}ms -> MOONSHINE (direct)")
        try:
            result = await transcribe_moonshine_direct(audio_bytes)
            if not result:
                return Response(status_code=204)
            return result
        except Exception as e:
            logger.error(f"[{client_id}] Moonshine transcription failed: {e}")
            return Response(content="Transcription failed.", status_code=500)
    else:
        # Long audio -> Parakeet (batched)
        logger.debug(f"[{client_id}] Audio duration: {audio_duration_ms:.1f}ms -> PARAKEET (batched)")
        
        if not parakeet_queue:
            logger.error("Request received but worker is not ready (queue not initialized).")
            return Response(content="Server is not ready.", status_code=503)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await parakeet_queue.put((audio_np_single, future))

        try:
            hypothesis_data = await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(f"[{client_id}] Request timed out waiting for transcription result.")
            return Response(content="Request timed out.", status_code=504)
        except Exception as e:
            logger.error(f"[{client_id}] An error occurred while waiting for transcription: {e}", exc_info=True)
            return Response(content="An unexpected error occurred.", status_code=500)

        if not hypothesis_data:
            return Response(status_code=204)
        
        return hypothesis_data


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "parakeet_queue_size": parakeet_queue.qsize() if parakeet_queue else -1,
        "short_duration_threshold_ms": SHORT_DURATION_MS,
        "parakeet_worker": PARAKEET_WORKER_URL,
        "moonshine_worker": MOONSHINE_WORKER_URL,
    }


if __name__ == "__main__":
    import uvicorn
    url = os.getenv("ASR_BATCHING_WORKER_URL", "http://localhost:8090")
    if "localhost" in url:
        port = url.split(":")[-1]
        uvicorn.run(app, host="0.0.0.0", port=int(port))
