#!/usr/bin/env python3
"""
Simple audio transcription client for STT gateway.
Sends an audio file to the gateway and saves transcripts with timing to a txt file.

Usage:
    python simulate_call.py <audio_file> [--url ws://localhost:8089]
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np
import websockets
from scipy.signal import resample

# Try to import audio libraries
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    from scipy.io import wavfile
    HAS_SCIPY_WAV = True
except ImportError:
    HAS_SCIPY_WAV = False

# Configuration
TARGET_SAMPLE_RATE = 8000
CHUNK_SIZE = 4096


@dataclass
class TranscriptSegment:
    """A single transcript segment with timing."""
    time_ms: float
    transcript: str


def linear_to_mulaw(sample: float) -> int:
    """Convert a float sample [-1, 1] to mu-law encoded byte."""
    MAX = 0x1FFF
    BIAS = 0x84
    
    sign = 0x80 if sample < 0 else 0
    if sign:
        sample = -sample
    
    sample = sample * MAX
    if sample > MAX:
        sample = MAX
    
    sample = sample + BIAS
    
    exponent = 7
    exp_mask = 0x4000
    while (int(sample) & exp_mask) == 0 and exponent > 0:
        exponent -= 1
        exp_mask >>= 1
    
    mantissa = (int(sample) >> (exponent + 3)) & 0x0F
    mulaw_byte = ~(sign | (exponent << 4) | mantissa)
    
    return mulaw_byte & 0xFF


def encode_audio_to_mulaw(audio_data: np.ndarray) -> bytes:
    """Convert float audio array to mu-law bytes."""
    mulaw_data = bytearray(len(audio_data))
    for i, sample in enumerate(audio_data):
        mulaw_data[i] = linear_to_mulaw(float(sample))
    return bytes(mulaw_data)


def load_and_prepare_audio(file_path: str) -> np.ndarray:
    """Load audio file and resample to 8kHz mono."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext != '.wav' or not HAS_SCIPY_WAV:
        if not HAS_PYDUB:
            raise ImportError(
                f"Cannot load {file_ext} files. Install pydub and ffmpeg:\n"
                "  pip install pydub\n"
                "  # Also install ffmpeg: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
            )
        return load_audio_with_pydub(file_path)
    
    try:
        sample_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        if HAS_PYDUB:
            print(f"⚠️  scipy.io.wavfile failed, falling back to pydub: {e}")
            return load_audio_with_pydub(file_path)
        raise
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Normalize to float [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Resample to 8kHz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
    
    return audio_data


def load_audio_with_pydub(file_path: str) -> np.ndarray:
    """Load audio file using pydub (supports webm, mp3, ogg, wav, etc.)."""
    audio = AudioSegment.from_file(file_path)
    
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    original_sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())
    
    if audio.sample_width == 1:
        audio_data = (samples.astype(np.float32) - 128) / 128.0
    elif audio.sample_width == 2:
        audio_data = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:
        audio_data = samples.astype(np.float32) / 2147483648.0
    else:
        audio_data = samples.astype(np.float32)
    
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    if original_sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / original_sample_rate)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
    
    return audio_data


def format_time(ms: float) -> str:
    """Format milliseconds as MM:SS.mmm"""
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:06.3f}"


async def transcribe_audio(audio_file: str, ws_url: str) -> List[TranscriptSegment]:
    """Send audio to gateway and collect transcripts with timing."""
    
    print(f"📂 Loading audio file: {audio_file}")
    audio_data = load_and_prepare_audio(audio_file)
    audio_duration_sec = len(audio_data) / TARGET_SAMPLE_RATE
    print(f"🎵 Audio loaded: {len(audio_data)} samples ({audio_duration_sec:.2f}s at {TARGET_SAMPLE_RATE}Hz)")
    
    session_id = f"transcribe-{int(time.time() * 1000)}"
    
    # Build WebSocket URL with query parameters
    full_url = (
        f"{ws_url}/ws/stt"
        f"?session_id={session_id}"
        f"&interim_results=true"
        f"&encoding=mulaw_8k"
        f"&X-Min-Endpointing-MS=500"
        f"&X-Max-Endpointing-MS=2000"
        f"&X-Max-Speech-Segment-MS=20000"
        f"&X-Min-Speech-Duration-MS=120"
        f"&X-Short-Transcript-Max-Len=3"
        f"&X-Word-Confidence-Threshold=0.85"
        f"&X-Short-Transcript-Avg-Word-Confidence-Threshold=0.70"
        f"&X-Single-Word-Confidence-Threshold=0.6"
    )
    
    segments: List[TranscriptSegment] = []
    audio_time_ms = 0.0  # Current position in audio timeline
    
    print(f"🔌 Connecting to {ws_url}...")
    
    async with websockets.connect(full_url, ping_interval=30, ping_timeout=10) as ws:
        print("✅ Connected")
        
        chunk_duration_sec = CHUNK_SIZE / TARGET_SAMPLE_RATE
        total_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        # Task to receive messages
        async def receive_messages():
            try:
                async for message in ws:
                    data = json.loads(message)
                    
                    if data.get("command") == "please_repeat_message":
                        continue
                    
                    transcript = data.get("transcript", "")
                    is_final = data.get("is_final", False)
                    
                    if is_final and transcript:
                        # Use current audio position as timestamp
                        segment = TranscriptSegment(
                            time_ms=audio_time_ms,
                            transcript=transcript
                        )
                        segments.append(segment)
                        
                        print(f"📝 [{format_time(segment.time_ms)}]: {transcript}")
                    elif not is_final and transcript:
                        # Interim result - just show progress
                        print(f"💬 (interim) {transcript}", end="\r")
                        
            except websockets.exceptions.ConnectionClosed:
                pass
        
        # Start receiver task
        receiver = asyncio.create_task(receive_messages())
        
        print(f"🎤 Sending {total_chunks} audio chunks...")
        
        # Send audio chunks
        for chunk_idx in range(0, len(audio_data), CHUNK_SIZE):
            chunk = audio_data[chunk_idx:chunk_idx + CHUNK_SIZE]
            
            mulaw_bytes = encode_audio_to_mulaw(chunk)
            audio_base64 = base64.b64encode(mulaw_bytes).decode('ascii')
            
            message = {
                "audio": audio_base64,
                "last_utterance": ""
            }
            
            await ws.send(json.dumps(message))
            audio_time_ms += len(chunk) / TARGET_SAMPLE_RATE * 1000
            
            # Simulate real-time delay
            await asyncio.sleep(chunk_duration_sec)
        
        print(f"\n⏳ Audio sent, waiting for final results...")
        
        # Wait for final results
        await asyncio.sleep(3.0)
        
        # Cancel receiver
        receiver.cancel()
        try:
            await receiver
        except asyncio.CancelledError:
            pass
    
    print(f"✅ Transcription complete: {len(segments)} segments")
    return segments


def save_transcripts(segments: List[TranscriptSegment], output_file: str):
    """Save transcript segments to a text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for segment in segments:
            line = f"[{format_time(segment.time_ms)}]: {segment.transcript}\n"
            f.write(line)
    print(f"💾 Saved {len(segments)} transcripts to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file using STT gateway",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python transcribe_audio.py audio.wav
    python transcribe_audio.py audio.mp3 --url ws://server:8089
        """
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:8089",
        help="WebSocket server URL (default: ws://localhost:8089)"
    )
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        return 1
    
    supported_formats = {'.wav'}
    if HAS_PYDUB:
        supported_formats.update({'.webm', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma'})
    
    if audio_path.suffix.lower() not in supported_formats:
        if not HAS_PYDUB:
            print(f"❌ Error: Format {audio_path.suffix} not supported. Install pydub for more formats:")
            print(f"   pip install pydub")
            return 1
    
    # Output file: same basename, .txt extension, same directory
    output_file = audio_path.with_suffix('.txt')
    
    try:
        segments = asyncio.run(transcribe_audio(args.audio_file, args.url))
        
        if segments:
            save_transcripts(segments, str(output_file))
        else:
            print("⚠️  No transcripts received")
            # Create empty file
            output_file.touch()
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
