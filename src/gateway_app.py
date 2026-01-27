import os
import asyncio
import json
import base64
import wave
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Deque
from dotenv import load_dotenv
import numpy as np
import torch
from silero_vad import load_silero_vad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx
from loguru import logger

from eos import EOS_DETECTOR
from conversation_state import ConversationState
from llm import is_user_speech, is_presence_check
import audio_preprocessing_app as audio_preprocessing

load_dotenv()

logger.remove()
logger.add(
    "logs/gateway_{time:YYYY-MM-DD}.log",
    rotation="00:00", retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}:{line}</cyan> - <level>{message}</level>",
    level="DEBUG", colorize=True
)

app = FastAPI()

# Constants
SAMPLE_RATE = 16000
VAD_FRAME_MS = 32
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2
NUM_TRAILING_NON_SPEECH_FRAMES = 2
NUM_PREFIX_NON_SPEECH_FRAMES = 3
SENDER_INTERVAL_S = 0.01
EARLY_VERIFICATION_MS = 1000
HOLD_ON_INTERVAL_S = 0.16  # 160ms = 5 audio frames
ENHANCE_CHUNK_FRAMES = 62  # ~2 seconds (62 * 32ms = 1984ms)
SHORT_AUDIO_DURATION_MS = 600 # will check is_speech for short audio only

BLACKLIST = {
    "mm-hmm", "uh-huh", "mhm", "um", "uh", "ah", "hmm", "oh", "mm", "mmm", "oh", "eh", "ah",
    "mm-hmm.", "uh-huh.", "mhm.", "um.", "uh.", "ah.", "hmm.", "oh.", "mm.", "mmm.", "oh.", "eh.", "ah."
}
EN_WORDS_WHITELIST = {"hi", "hey", "hello", "bye"}
ASR_RESULT_LOWER_CONFIDENCE = "[unintelligible audio from client for ASR]"

@dataclass
class Config:
    vad_threshold: float = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.45"))
    min_endpointing_ms: int = int(os.getenv("MIN_ENDPOINTING_MS", "500"))
    max_endpointing_ms: int = int(os.getenv("MAX_ENDPOINTING_MS", "3000"))
    max_speech_ms: int = int(os.getenv("MAX_SPEECH_SEGMENT_MS", "20000"))
    min_speech_ms: int = int(os.getenv("MIN_SPEECH_DURATION_MS", "120"))
    speaker_verification: bool = os.getenv("SPEAKER_VERIFICATION_ENABLED", "true").lower() == "true"
    word_confidence_threshold: float = float(os.getenv("WORD_CONFIDENCE_THRESHOLD", "0.65"))
    short_transcript_avg_threshold: float = float(os.getenv("SHORT_TRANSCRIPT_AVG_THRESHOLD", "0.6"))
    single_word_threshold: float = float(os.getenv("SINGLE_WORD_CONFIDENCE_THRESHOLD", "0.55"))
    debug_save_audio: bool = os.getenv("DEBUG_SAVE_AUDIO", "false").lower() == "true"


config = Config()

asr_client = httpx.AsyncClient(
    base_url=os.getenv("ASR_BATCHING_WORKER_URL", "http://localhost:8090"),
    timeout=60.0, limits=httpx.Limits(max_connections=500)
)
speaker_client = httpx.AsyncClient(
    base_url=os.getenv("SPEAKER_VERIFICATION_WORKER_URL", "http://localhost:8092"),
    timeout=10.0, limits=httpx.Limits(max_connections=100)
)

cpu_executor = ThreadPoolExecutor(max_workers=max(4, os.cpu_count() * 2), thread_name_prefix="gw")


class AudioProcessor:
    def __init__(self, client_id: str, websocket: WebSocket, encoding: str = "pcm_16k"):
        self.client_id = client_id
        self.websocket = websocket
        self.encoding = encoding
        
        # VAD
        self.vad_model = load_silero_vad(onnx=False)
        self.audio_buffer = bytearray()
        self.speech_buffer = bytearray()
        self.non_speech_buffer: Deque[bytes] = deque(maxlen=NUM_PREFIX_NON_SPEECH_FRAMES)
        
        # Incremental noise reduction
        self.enhance_tasks: list[asyncio.Task] = []
        self.enhanced_bytes_submitted = 0
        self.frames_since_enhance = 0
        
        # State
        self.is_speaking = False
        self.speech_frames = 0
        self.silent_frames = 0
        self.waiting_eos = False
        
        # Timing
        self.min_speech_frames = config.min_speech_ms // VAD_FRAME_MS
        self.max_speech_frames = config.max_speech_ms // VAD_FRAME_MS
        self.min_endpointing_frames = config.min_endpointing_ms // VAD_FRAME_MS
        self.early_verification_frames = EARLY_VERIFICATION_MS // VAD_FRAME_MS
        
        # Speaker verification
        self.is_enrolled = False
        self.hold_on_task: Optional[asyncio.Task] = None
        
        # Conversation state
        self.conversation = ConversationState(
            client_id=client_id,
            base_endpointing_frames=self.min_endpointing_frames,
            max_endpointing_frames=config.max_endpointing_ms // VAD_FRAME_MS,
            vad_frame_ms=VAD_FRAME_MS
        )
        
        # EOS detector
        self.eos_detector = EOS_DETECTOR(
            method=os.getenv("EOS_METHOD", "naive"),
            endpointing_ms=config.min_endpointing_ms,
            vad_frame_ms=VAD_FRAME_MS
        )
        
        # Pending work queue
        self.pending: Deque[Tuple[bytes, bool]] = deque()
        self.sender_task = asyncio.create_task(self._sender_loop())
        
        # Debug audio saving
        self.debug_audio_buffer = bytearray() if config.debug_save_audio else None
        
        logger.info(f"[{client_id}] Session started | verification={config.speaker_verification}")

    def _log(self, level: str, msg: str):
        getattr(logger, level)(f"[{self.client_id}] {msg}")

    def _reset_state(self):
        """Reset for new utterance."""
        self.speech_buffer.clear()
        self.non_speech_buffer.clear()
        self.enhance_tasks.clear()
        self.enhanced_bytes_submitted = 0
        self.frames_since_enhance = 0
        self.is_speaking = False
        self.speech_frames = 0
        self.silent_frames = 0
        self.waiting_eos = False
        self._stop_hold_on()
        try:
            self.vad_model.reset_states()
        except:
            pass

    def _stop_hold_on(self):
        """Stop the hold_on loop if running."""
        if self.hold_on_task and not self.hold_on_task.done():
            self.hold_on_task.cancel()
        self.hold_on_task = None

    def _compute_vad(self, frame: bytes) -> bool:
        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        with torch.no_grad():
            prob = self.vad_model(tensor, SAMPLE_RATE).item()
        return prob >= config.vad_threshold

    async def process_chunk(self, chunk: bytes, agent_utterance: str = ""):
        """Process incoming audio chunk."""
        if agent_utterance:
            self.conversation.add_agent_utterance(agent_utterance)
        
        if self.encoding != "pcm_16k":
            loop = asyncio.get_running_loop()
            chunk = await loop.run_in_executor(
                cpu_executor, audio_preprocessing.preprocess_audio,
                chunk, self.encoding, False, False
            )
        
        # Accumulate audio for debug saving
        if self.debug_audio_buffer is not None:
            self.debug_audio_buffer.extend(chunk)
        
        await self._vad_process(chunk)

    async def _vad_process(self, chunk: bytes):
        """VAD state machine."""
        self.audio_buffer.extend(chunk)
        
        while len(self.audio_buffer) >= VAD_FRAME_BYTES:
            frame = bytes(self.audio_buffer[:VAD_FRAME_BYTES])
            del self.audio_buffer[:VAD_FRAME_BYTES]
            
            try:
                is_speech = self._compute_vad(frame)
            except Exception as e:
                self._log("error", f"VAD error: {e}")
                continue

            if self.is_speaking:
                self.speech_buffer.extend(frame)
                
                if is_speech:
                    self.speech_frames += 1
                    self.silent_frames = 0
                    self.frames_since_enhance += 1
                    
                    # Early verification at EARLY_VERIFICATION_MS
                    if (self.speech_frames == self.early_verification_frames and
                        config.speaker_verification and self.is_enrolled):
                        asyncio.create_task(self._early_verification())
                    
                    # Incremental noise reduction every ENHANCE_CHUNK_FRAMES
                    if self.frames_since_enhance >= ENHANCE_CHUNK_FRAMES:
                        # Capture current chunk before async task runs
                        chunk_bytes = self.frames_since_enhance * VAD_FRAME_BYTES
                        chunk = bytes(self.speech_buffer[-chunk_bytes:])
                        self.enhanced_bytes_submitted += chunk_bytes
                        self.frames_since_enhance = 0
                        self.enhance_tasks.append(asyncio.create_task(self._enhance_audio(chunk)))
                    
                    if self.speech_frames >= self.max_speech_frames:
                        self.pending.append((bytes(self.speech_buffer), True))
                        self._reset_state()
                else:
                    self.silent_frames += 1
                    endpointing_frames = self.conversation.get_current_endpointing_frames()
                    
                    if self.silent_frames >= endpointing_frames:
                        if self.speech_frames >= self.min_speech_frames:
                            if not self.waiting_eos:
                                await self._check_eos()
                        else:
                            self._reset_state()
            else:
                if is_speech:
                    self._log("info", "🏁 Speech started")
                    self.is_speaking = True
                    for f in self.non_speech_buffer:
                        self.speech_buffer.extend(f)
                    self.speech_buffer.extend(frame)
                    self.speech_frames = 1
                    self.silent_frames = 0
                    self.non_speech_buffer.clear()
                else:
                    self.non_speech_buffer.append(frame)

    async def _enhance_audio(self, audio_bytes: bytes) -> bytes:
        """Apply noise reduction."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            cpu_executor, audio_preprocessing.preprocess_audio,
            audio_bytes, "pcm_16k", True, False  # noise_reduction=True, normalization=False
        )

    async def _get_enhanced_audio(self, audio_bytes: bytes) -> bytes:
        """Get fully enhanced audio by combining pre-enhanced chunks with remaining."""
        # Await all pending enhancement tasks in order
        enhanced_segments = []
        for task in self.enhance_tasks:
            try:
                enhanced_segments.append(await task)
            except Exception as e:
                self._log("error", f"Chunk enhancement error: {e}")
        
        # Enhance remaining unprocessed portion
        remaining_bytes = len(audio_bytes) - self.enhanced_bytes_submitted
        if remaining_bytes > 0:
            remaining = audio_bytes[self.enhanced_bytes_submitted:]
            enhanced_segments.append(await self._enhance_audio(remaining))
        
        if enhanced_segments:
            return b"".join(enhanced_segments)
        
        # Fallback: no pre-enhanced segments, enhance entire audio
        return await self._enhance_audio(audio_bytes)

    async def _early_verification(self):
        """
        Early verification at 1 second:
        1. Apply noise reduction
        2. Do speaker verification on enhanced audio
        3. If passed or uncertain_high, start hold_on loop
        """
        try:
            audio = bytes(self.speech_buffer)
            audio_duration_ms = len(audio) // 32
            enhanced = await self._enhance_audio(audio)
            
            response = await speaker_client.post(
                f"/verify/{self.client_id}",
                content=enhanced,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Audio-Duration-MS": str(audio_duration_ms)
                }
            )
            if response.status_code != 200:
                return
            
            result = response.json()
            is_speaker = result.get("is_speaker", False)
            sim = result.get("similarity", 0.0)
            
            if is_speaker is True or is_speaker == "uncertain_high":
                self._log("info", f"🔍 Early verification passed (sim={sim:.3f}, dur={audio_duration_ms}ms)")
                self.hold_on_task = asyncio.create_task(self._hold_on_loop())
        except Exception as e:
            self._log("error", f"Early verification error: {e}")

    async def _hold_on_loop(self):
        """
        Send hold_on message every 160ms (5 audio frames) while is_final=False.
        Stops when speech ends or EOS is detected.
        """
        try:
            while self.is_speaking and not self.waiting_eos:
                await self.websocket.send_json({
                    "is_final": False,
                    "transcript": "[Hold on...]",
                    "avg_confidence": 0.9,
                    "audio_duration_ms": 1000
                })
                self._log("debug", "📢 Sent hold_on")
                await asyncio.sleep(HOLD_ON_INTERVAL_S)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._log("error", f"Hold on loop error: {e}")

    async def _check_eos(self):
        """Check end-of-speech and finalize if needed."""
        self.waiting_eos = True
        self._stop_hold_on()
        
        max_frames = self.conversation.get_current_max_endpointing_frames()
        if self.silent_frames >= max_frames:
            should_finalize = True
        else:
            should_finalize, _ = await self.eos_detector.should_finalize_speech(
                bytes(self.speech_buffer), self.silent_frames, self.client_id
            )
        
        if should_finalize:
            audio = self._trim_silence(bytes(self.speech_buffer))
            self.pending.append((audio, True))
            self._reset_state()
        else:
            self.waiting_eos = False

    def _trim_silence(self, audio: bytes) -> bytes:
        """Trim trailing silence from audio."""
        frames_to_keep = min(self.silent_frames, NUM_TRAILING_NON_SPEECH_FRAMES)
        frames_to_remove = max(0, self.silent_frames - frames_to_keep)
        bytes_to_remove = frames_to_remove * VAD_FRAME_BYTES
        if bytes_to_remove > 0 and bytes_to_remove < len(audio):
            return audio[:-bytes_to_remove]
        return audio

    async def _sender_loop(self):
        """Background sender loop."""
        try:
            while True:
                await asyncio.sleep(SENDER_INTERVAL_S)
                
                while self.pending:
                    audio, _ = self.pending.popleft()
                    try:
                        await self._process_final(audio)
                    except (WebSocketDisconnect, RuntimeError):
                        return
                    except Exception as e:
                        self._log("error", f"Process error: {e}")
        except asyncio.CancelledError:
            pass

    async def _process_final(self, audio_bytes: bytes):
        """
        Process final audio segment.
        
        1. Apply noise reduction (uses incrementally enhanced chunks)
        2. Check is_speech for short audio (<600ms)
        3. Transcribe and send to client
        4. Handle enrollment/verification if enabled
        """
        audio_duration_ms = len(audio_bytes) // 32
        enhanced = await self._get_enhanced_audio(audio_bytes)
        
        # Check is_speech for short audio only
        if audio_duration_ms < SHORT_AUDIO_DURATION_MS:
            loop = asyncio.get_running_loop()
            is_speech = await loop.run_in_executor(
                cpu_executor, audio_preprocessing.check_is_speech, enhanced
            )
            if not is_speech:
                self._log("debug", f"❌ Not speech (<600ms)")
                return
        
        if config.speaker_verification:
            if self.is_enrolled:
                result = await self._verify_and_transcribe(enhanced, audio_duration_ms)
            else:
                result = await self._transcribe(enhanced)
                if result and result.get("transcript"):
                    asyncio.create_task(self._enrollment_background(
                        enhanced, 
                        result["transcript"], 
                        result.get("avg_confidence", 0.85)
                    ))
        else:
            result = await self._transcribe(enhanced)
        
        if result:
            result["audio_duration_ms"] = audio_duration_ms
            await self._send_result(result)

    async def _verify_and_transcribe(self, enhanced_audio: bytes, audio_duration_ms: int) -> Optional[Dict]:
        """
        Verification phase with parallel processing:
        1. In parallel: verify + transcribe
        2. Send to client if: passed OR uncertain_high OR presence check
        3. Update embeddings based on conditions
        """
        async def verify():
            response = await speaker_client.post(
                f"/verify/{self.client_id}",
                content=enhanced_audio,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Audio-Duration-MS": str(audio_duration_ms)
                }
            )
            if response.status_code != 200:
                return True, 0.0, False  # Allow on error
            result = response.json()
            return (
                result.get("is_speaker", True),
                result.get("similarity", 0.0),
                result.get("has_room", False)
            )
        
        # Parallel: verification + transcription
        (is_speaker, similarity, has_room), result = await asyncio.gather(
            verify(),
            self._transcribe(enhanced_audio)
        )
        
        if not result or not result.get("transcript"):
            return None
        
        transcript = result["transcript"]
        avg_confidence = result.get("avg_confidence", 0.85)
        is_presence = is_presence_check(transcript)
        
        # Decision: send if passed OR uncertain_high OR presence check
        should_send = is_speaker is True or is_speaker == "uncertain_high" or is_presence

        self._log("info", f"Speaker verification: sim={similarity:.3f}, dur={audio_duration_ms}ms, is_presence={is_presence}, is_speaker={is_speaker}, transcript='{transcript}'")
        
        if not should_send:
            self._log("warning", f"❌ Blocked (sim={similarity:.3f}, dur={audio_duration_ms}ms)")
            return None
        
        # Update embedding if: is_presence=True OR is_speaker=True (not uncertain)
        if transcript == ASR_RESULT_LOWER_CONFIDENCE:
            pass
        elif is_presence:
            await self._update_embedding(confidence=avg_confidence, force_replace=True)
            self._log("info", f"🎯 Presence check: '{transcript}' (force update)")
        elif is_speaker is True and has_room:
            asyncio.create_task(self._llm_check_and_update(transcript, avg_confidence))
        
        return result

    async def _enrollment_background(self, enhanced_audio: bytes, transcript: str, avg_confidence: float):
        """
        Background enrollment task:
        1. Call LLM to decide if transcript is from user (using _is_user_speech function)
        2. If yes, add enhanced audio to speaker embeddings
        """
        try:
            # Skip enrollment for low confidence transcripts
            if transcript == ASR_RESULT_LOWER_CONFIDENCE:
                return
            
            loop = asyncio.get_running_loop()
            conv_history = self.conversation.get_conversation_history(n=5, exclude_last_user=True)
            
            # LLM check: is this from the user?
            is_user = await loop.run_in_executor(
                cpu_executor, is_user_speech, transcript, avg_confidence, conv_history
            )
            
            if is_user:
                await self._store_enrollment(enhanced_audio)
                self._log("info", f"✅ LLM approved enrollment: '{transcript}'")
            else:
                self._log("info", f"🚫 LLM rejected enrollment: '{transcript}'")
        except Exception as e:
            self._log("error", f"Enrollment error: {e}")

    async def _llm_check_and_update(self, transcript: str, avg_confidence: float):
        """
        Background LLM check during verification phase:
        1. Call LLM to decide if transcript is from user (using is_user_speech function)
        2. If yes, add to speaker embeddings
        """
        try:
            # Skip update for low confidence transcripts
            if transcript == ASR_RESULT_LOWER_CONFIDENCE:
                return
            
            loop = asyncio.get_running_loop()
            conv_history = self.conversation.get_conversation_history(n=5, exclude_last_user=True)
            
            is_user = await loop.run_in_executor(
                cpu_executor, is_user_speech, transcript, avg_confidence, conv_history
            )
            
            if is_user:
                await self._update_embedding(confidence=avg_confidence, force_replace=False)
                self._log("info", f"✅ LLM approved embedding update: '{transcript}'")
            else:
                self._log("info", f"🚫 LLM rejected embedding update: '{transcript}'")
        except Exception as e:
            self._log("error", f"LLM check error: {e}")

    async def _store_enrollment(self, audio_bytes: bytes):
        """Store embedding for enrollment."""
        try:
            response = await speaker_client.post(
                f"/enroll/{self.client_id}",
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream"}
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("enrolled"):
                self.is_enrolled = True
                self._log("info", "🎭 Enrollment complete!")
            else:
                commit_response = await speaker_client.post(f"/commit-enrollment/{self.client_id}")
                commit_result = commit_response.json()
                if commit_result.get("enrolled"):
                    self.is_enrolled = True
                    self._log("info", f"🎭 Enrollment complete! ({commit_result.get('enrollment_progress')})")
                else:
                    self._log("debug", f"📝 Enrollment: {commit_result.get('enrollment_progress')}")
        except Exception as e:
            self._log("error", f"Store enrollment failed: {e}")

    async def _update_embedding(self, confidence: float = 0.85, force_replace: bool = False):
        """Update speaker embedding."""
        try:
            response = await speaker_client.post(
                f"/update-embedding/{self.client_id}",
                json={"confidence": confidence, "force_replace": force_replace}
            )
            response.raise_for_status()
            result = response.json()
            if result.get("updated"):
                action = "force replaced" if force_replace else "updated"
                self._log("info", f"🔄 Embedding {action} (conf={confidence:.2f})")
        except Exception as e:
            self._log("error", f"Update embedding failed: {e}")

    def _check_asr_confidence(self, data: Dict) -> Tuple[bool, str]:
        """Check if transcription meets confidence thresholds."""
        confidences = data.get("word_confidence")
        if not confidences:
            return True, "no scores"
        
        words = data.get("text", "").split()
        num_words = len(words)
        avg_conf = sum(confidences) / len(confidences)
        
        if num_words == 1:
            word = words[0].strip(".,?!").lower()
            conf = confidences[0]
            if word in EN_WORDS_WHITELIST:
                if conf < config.single_word_threshold:
                    return False, f"whitelist '{word}' conf {conf:.2f}"
            elif conf < config.word_confidence_threshold:
                return False, f"single '{word}' conf {conf:.2f}"
        elif num_words <= 3:
            if avg_conf < config.short_transcript_avg_threshold:
                return False, f"short avg {avg_conf:.2f}"
        else:
            if avg_conf < config.word_confidence_threshold:
                return False, f"long avg {avg_conf:.2f}"
        
        return True, "passed"

    async def _transcribe(self, audio_bytes: bytes) -> Optional[Dict]:
        """Transcribe audio via ASR service."""
        try:
            response = await asr_client.post(
                "/transcribe",
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream", "X-Client-ID": self.client_id}
            )
            
            if response.status_code == 204:
                return None
            
            response.raise_for_status()
            data = response.json()
            
            text = data.get("text", "").strip()
            if not text or text.lower().replace(".", "") in BLACKLIST:
                return None
            
            passes, reason = self._check_asr_confidence(data)
            transcript = text if passes else ASR_RESULT_LOWER_CONFIDENCE
            
            if not passes:
                self._log("warning", f"Low confidence ({reason}): '{text}'")
            
            return {
                "is_final": True,
                "transcript": transcript,
                "avg_confidence": data.get("avg_confidence", 0.85)
            }
        except Exception as e:
            self._log("error", f"ASR error: {e}")
            return None

    async def _send_result(self, result: Dict):
        """Send result to client."""
        transcript = result.get("transcript", "")
        if transcript:
            self.conversation.add_user_utterance(transcript)
            self._log("info", f"📤 '{transcript}'")
        
        try:
            await self.websocket.send_json(result)
        except (WebSocketDisconnect, RuntimeError):
            raise

    async def _save_debug_audio(self):
        """Save accumulated audio to WAV file (non-blocking)."""
        if not self.debug_audio_buffer or len(self.debug_audio_buffer) == 0:
            return
        
        try:
            audio_data = bytes(self.debug_audio_buffer)
            filename = f"debug_audio/{self.client_id}.wav"
            
            # Clear buffer immediately to free memory
            self.debug_audio_buffer.clear()
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                cpu_executor,
                self._write_wav_file,
                filename,
                audio_data
            )
            self._log("info", f"💾 Debug audio saved: {filename}")
        except Exception as e:
            self._log("error", f"Failed to save debug audio: {e}")
    
    def _write_wav_file(self, filename: str, audio_data: bytes):
        """Write audio data to WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)  # 16kHz
            wf.writeframes(audio_data)
    
    async def finalize(self):
        """Cleanup on session end."""
        if self.sender_task:
            self.sender_task.cancel()
        self._stop_hold_on()
        
        if self.speech_buffer and self.speech_frames >= self.min_speech_frames:
            audio = self._trim_silence(bytes(self.speech_buffer))
            try:
                await self._process_final(audio)
            except:
                pass
        
        if config.speaker_verification:
            try:
                await speaker_client.delete(f"/session/{self.client_id}")
                self._log("debug", "🗑️ Speaker profile cleaned up")
            except Exception as e:
                self._log("warning", f"Speaker cleanup failed: {e}")
        
        # Save debug audio if enabled (non-blocking)
        if config.debug_save_audio:
            asyncio.create_task(self._save_debug_audio())


@app.on_event("startup")
async def startup():
    logger.info("🚀 Gateway starting...")
    audio_preprocessing.initialize_preprocessing()
    logger.info("🚀 Gateway ready")


@app.on_event("shutdown")
async def shutdown():
    await asr_client.aclose()
    await speaker_client.aclose()
    cpu_executor.shutdown(wait=True)


@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    session_id = websocket.headers.get("session_id", f"session_{os.urandom(8).hex()}")
    processor = None
    
    try:
        await websocket.accept()
        encoding = websocket.query_params.get("encoding", "pcm_16k").lower()
        processor = AudioProcessor(session_id, websocket, encoding)
        
        while True:
            message = await websocket.receive()
            
            if message.get("type") == "websocket.disconnect":
                break
            
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    audio = base64.b64decode(data.get("audio", ""))
                    await processor.process_chunk(audio, data.get("last_utterance", ""))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"[{session_id}] Parse error: {e}")
            elif "bytes" in message:
                await processor.process_chunk(message["bytes"])
    
    except WebSocketDisconnect:
        pass
    finally:
        if processor:
            await processor.finalize()
        logger.info(f"[{session_id}] Session ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("GATEWAY_PORT", "8089")))