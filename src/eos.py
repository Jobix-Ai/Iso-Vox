import logging
import os
from dotenv import load_dotenv
import time
from typing import Tuple
import httpx

load_dotenv()

logger = logging.getLogger(__name__)


class EOS_DETECTOR:
    """
    End-of-Speech (EOS) Detection with multiple strategies.
    
    Supports two methods:
    - "naive": Traditional timeout-based detection (waits for ENDPOINTING_MS of silence)
    - "smart_turn": ML-based detection using the smart-turn service
    """
    
    def __init__(self, method: str, endpointing_ms: int, vad_frame_ms: int):
        """
        Initialize EOS detector with specified method.
        
        Args:
            method: Either "naive" or "smart_turn"
            endpointing_ms: Milliseconds of silence before deciding EOS
            vad_frame_ms: Size of VAD frame in milliseconds
        """
        self.method = method.lower()
        if self.method not in ["naive", "smart_turn"]:
            raise ValueError(f"EOS method must be 'naive' or 'smart_turn', got: {self.method}")
        
        self.endpointing_ms = endpointing_ms
        self.vad_frame_ms = vad_frame_ms
        self.frames_for_endpointing = endpointing_ms // vad_frame_ms
        
        # Initialize smart_turn threshold and client from environment
        self.smart_turn_threshold = float(os.getenv("SMART_TURN_THRESHOLD", "0.5"))
        
        # Create smart_turn HTTP client if using smart_turn method
        if self.method == "smart_turn":
            smart_turn_worker_url = os.getenv("SMART_TURN_WORKER_URL", "http://127.0.0.1:8088")
            self.smart_turn_client = httpx.AsyncClient(
                base_url=smart_turn_worker_url,
                timeout=5.0,
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=20)
            )
            logger.info(f"EOS_DETECTOR initialized: method={self.method}, "
                       f"endpointing_ms={endpointing_ms}, threshold={self.smart_turn_threshold}, "
                       f"smart_turn_url={smart_turn_worker_url}")
        else:
            self.smart_turn_client = None
            logger.info(f"EOS_DETECTOR initialized: method={self.method}, "
                       f"endpointing_ms={endpointing_ms}")
    
    async def should_finalize_speech(self, audio_buffer: bytes, silent_frames_count: int, 
                                      client_id: str = "") -> Tuple[bool, str]:
        """
        Determine if speech should be finalized.
        
        Args:
            audio_buffer: Audio data in bytes (int16 PCM)
            silent_frames_count: Number of consecutive silent frames detected
            client_id: Client identifier for logging
            
        Returns:
            Tuple of (should_finalize: bool, reason: str)
        """
        if self.method == "naive":
            return self._naive_eos(silent_frames_count, client_id)
        elif self.method == "smart_turn":
            return await self._smart_turn_eos(audio_buffer, silent_frames_count, client_id)
    
    def _naive_eos(self, silent_frames_count: int, client_id: str) -> Tuple[bool, str]:
        """
        Traditional timeout-based EOS detection.
        Simply waits for ENDPOINTING_MS of silence before finalizing.
        
        Args:
            silent_frames_count: Number of consecutive silent frames
            client_id: Client identifier for logging
            
        Returns:
            Tuple of (should_finalize, reason)
        """
        if silent_frames_count >= self.frames_for_endpointing:
            silence_ms = silent_frames_count * self.vad_frame_ms
            reason = f"naive (silence: {silence_ms}ms >= {self.endpointing_ms}ms)"
            logger.debug(f"[{client_id}] EOS decision: {reason}")
            return True, reason
        
        return False, f"naive (waiting for {self.endpointing_ms}ms silence)"
    
    async def _smart_turn_eos(self, audio_buffer: bytes, silent_frames_count: int, 
                               client_id: str) -> Tuple[bool, str]:
        """
        ML-based EOS detection using smart-turn service.
        First checks if minimum silence threshold is met, then uses ML model to determine
        if the speech represents a complete conversational turn.
        
        Args:
            audio_buffer: Audio data to analyze
            silent_frames_count: Number of consecutive silent frames
            client_id: Client identifier for logging
            
        Returns:
            Tuple of (should_finalize, reason)
        """
        # First check if we've had enough silence frames
        if silent_frames_count < self.frames_for_endpointing:
            return False, f"smart_turn (insufficient silence: {silent_frames_count}/{self.frames_for_endpointing} frames)"
        
        # Now check with smart-turn ML service
        try:
            start_time = time.time()
            
            response = await self.smart_turn_client.post(
                "/predict",
                content=audio_buffer,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Client-ID": client_id
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            elapsed_ms = (time.time() - start_time) * 1000
            probability = result['probability']
            is_complete = probability >= self.smart_turn_threshold
            
            reason = (f"smart_turn (prediction={result['prediction']}, "
                     f"prob={probability:.4f}, threshold={self.smart_turn_threshold}, "
                     f"complete={is_complete}, time={elapsed_ms:.2f}ms)")
            
            logger.info(f"[{client_id}] EOS decision: {reason}")
            
            return is_complete, reason
            
        except Exception as e:
            logger.error(f"[{client_id}] Smart-Turn EOS error: {e}, falling back to naive method")
            # Fall back to naive method on error
            return self._naive_eos(silent_frames_count, client_id)
    
    async def cleanup(self):
        """
        Cleanup resources (close HTTP client if initialized).
        """
        if self.smart_turn_client is not None:
            await self.smart_turn_client.aclose()
            logger.info("Smart-Turn HTTP client closed")
