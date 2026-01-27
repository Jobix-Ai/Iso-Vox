#!/usr/bin/env python3
"""
Concurrent WebSocket client tester for STT gateway.
Simulates multiple clients sending audio from a file.

Usage:
    python concurrent_client_test.py <audio_file> <num_clients> [--url ws://localhost:8089]
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import websockets
from scipy.signal import resample

# Try to import audio libraries - pydub for broad format support
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

# Configuration matching test_app.html
TARGET_SAMPLE_RATE = 8000
CHUNK_SIZE = 4096  # Matches the ScriptProcessor buffer size in test_app.html

# Script sentences (matching test_app.html)
SCRIPT_SENTENCES = [
    "Hi this is Emily from ABC, how are you doing today?",
    "First of all, could you confirm your date of birth?",
    "Sorry I could not catch that. What is your date of birth please?",
    "Thanks. And do you want to consider our offer?",
    "Could you confirm that you have a bank account?",
    "Our assistant will now guide you through the process of setting up your account. Now, please provide your bank account number.",
]


def linear_to_mulaw(sample: float) -> int:
    """Convert a float sample [-1, 1] to mu-law encoded byte.
    
    This matches the linearToMulaw function in test_app.html.
    """
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
    """Load audio file and resample to 8kHz mono.
    
    Supports WAV files natively, and other formats (webm, mp3, ogg, etc.) via pydub/ffmpeg.
    """
    file_ext = Path(file_path).suffix.lower()
    
    # Try pydub first for non-WAV files or if scipy wav fails
    if file_ext != '.wav' or not HAS_SCIPY_WAV:
        if not HAS_PYDUB:
            raise ImportError(
                f"Cannot load {file_ext} files. Install pydub and ffmpeg:\n"
                "  pip install pydub\n"
                "  # Also install ffmpeg: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)"
            )
        return load_audio_with_pydub(file_path)
    
    # Try scipy for WAV files
    try:
        sample_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        # Fall back to pydub if scipy fails
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
    
    # Clip to [-1, 1] range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Resample to 8kHz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / sample_rate)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
    
    return audio_data


def load_audio_with_pydub(file_path: str) -> np.ndarray:
    """Load audio file using pydub (supports webm, mp3, ogg, wav, etc.)."""
    # Load audio file
    audio = AudioSegment.from_file(file_path)
    
    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Get sample rate before resampling
    original_sample_rate = audio.frame_rate
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # Normalize based on sample width
    if audio.sample_width == 1:  # 8-bit
        audio_data = (samples.astype(np.float32) - 128) / 128.0
    elif audio.sample_width == 2:  # 16-bit
        audio_data = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        audio_data = samples.astype(np.float32) / 2147483648.0
    else:
        audio_data = samples.astype(np.float32)
    
    # Clip to [-1, 1] range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Resample to 8kHz if needed
    if original_sample_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / original_sample_rate)
        audio_data = resample(audio_data, num_samples).astype(np.float32)
    
    return audio_data


async def simulate_client(
    client_id: int,
    audio_data: np.ndarray,
    ws_url: str,
    results: Dict[int, Dict[str, Any]],
    verbose: bool = True
):
    """Simulate a single client sending audio and receiving transcripts."""
    session_id = f"test-client-{client_id}-{int(time.time() * 1000)}"
    
    # Build WebSocket URL with query parameters (matching test_app.html)
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
    
    results[client_id] = {
        "status": "connecting",
        "transcripts": [],
        "interim_count": 0,
        "final_count": 0,
        "start_time": time.time(),
        "end_time": None,
        "error": None,
        "latencies": [],  # Time from audio sent to transcript received
        "sentence_index": 0  # Track which sentence to use next
    }
    
    try:
        async with websockets.connect(full_url, ping_interval=30, ping_timeout=10) as ws:
            results[client_id]["status"] = "connected"
            if verbose:
                print(f"[Client {client_id}] ✅ Connected to {ws_url}")
            
            # Calculate chunk timing to simulate real-time audio
            # At 8kHz, each sample is 1/8000 seconds
            chunk_duration_sec = CHUNK_SIZE / TARGET_SAMPLE_RATE
            
            # Task to receive messages
            async def receive_messages():
                nonlocal last_utterance, should_send_last_utterance
                try:
                    async for message in ws:
                        recv_time = time.time()
                        data = json.loads(message)
                        
                        if data.get("command") == "please_repeat_message":
                            continue
                        
                        transcript = data.get("transcript", "")
                        is_final = data.get("is_final", False)
                        audio_duration_ms = data.get("audio_duration_ms", 0)
                        avg_confidence = data.get("avg_confidence")
                        
                        if is_final:
                            results[client_id]["final_count"] += 1
                            if transcript:
                                results[client_id]["transcripts"].append(transcript)
                                if verbose:
                                    print(f"[Client {client_id}] 📝 [FINAL] \"{transcript}\" (duration: {audio_duration_ms}ms)")
                                
                                # After final transcript, set next sentence as last_utterance
                                sentence_idx = results[client_id]["sentence_index"]
                                last_utterance = SCRIPT_SENTENCES[sentence_idx % len(SCRIPT_SENTENCES)]
                                results[client_id]["sentence_index"] += 1
                                should_send_last_utterance = True
                                if verbose:
                                    print(f"[Client {client_id}] 🔄 Will send last_utterance: \"{last_utterance}\"")
                        else:
                            results[client_id]["interim_count"] += 1
                            if verbose and transcript:
                                conf_str = f"{avg_confidence:.2f}" if avg_confidence else "N/A"
                                print(f"[Client {client_id}] 💬 [INTERIM] \"{transcript}\" (duration: {audio_duration_ms}ms, conf: {conf_str})")
                except websockets.exceptions.ConnectionClosed:
                    pass
            
            # Start receiver task
            receiver = asyncio.create_task(receive_messages())
            
            # Send audio chunks
            # Start with first sentence as last_utterance (simulating agent spoke first)
            last_utterance = SCRIPT_SENTENCES[0]
            results[client_id]["sentence_index"] = 1  # Next sentence will be index 1
            should_send_last_utterance = True
            total_chunks = (len(audio_data) + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            if verbose:
                print(f"[Client {client_id}] 🎤 Sending {total_chunks} audio chunks ({len(audio_data) / TARGET_SAMPLE_RATE:.2f}s of audio)")
                print(f"[Client {client_id}] 🔄 Initial last_utterance: \"{last_utterance}\"")
            
            for chunk_idx in range(0, len(audio_data), CHUNK_SIZE):
                chunk = audio_data[chunk_idx:chunk_idx + CHUNK_SIZE]
                
                # Encode to mu-law
                mulaw_bytes = encode_audio_to_mulaw(chunk)
                audio_base64 = base64.b64encode(mulaw_bytes).decode('ascii')
                
                # Build message (matching test_app.html format)
                message = {
                    "audio": audio_base64,
                    "last_utterance": last_utterance if should_send_last_utterance else ""
                }
                
                if should_send_last_utterance:
                    if verbose:
                        print(f"[Client {client_id}] 📤 Sending last_utterance: \"{last_utterance}\"")
                    should_send_last_utterance = False
                
                await ws.send(json.dumps(message))
                
                # Simulate real-time delay
                await asyncio.sleep(chunk_duration_sec)
            
            if verbose:
                print(f"[Client {client_id}] ⏳ Audio sent, waiting for final results...")
            
            # Wait a bit for final results
            await asyncio.sleep(3.0)
            
            # Cancel receiver and close
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
            
            results[client_id]["status"] = "completed"
            results[client_id]["end_time"] = time.time()
            
            if verbose:
                duration = results[client_id]["end_time"] - results[client_id]["start_time"]
                print(f"[Client {client_id}] ✅ Completed in {duration:.2f}s - Finals: {results[client_id]['final_count']}, Interims: {results[client_id]['interim_count']}")
            
    except websockets.exceptions.InvalidStatusCode as e:
        results[client_id]["status"] = "error"
        results[client_id]["error"] = f"Connection rejected: {e.status_code}"
        results[client_id]["end_time"] = time.time()
        print(f"[Client {client_id}] ❌ Connection rejected: {e.status_code}")
    except websockets.exceptions.ConnectionClosedError as e:
        results[client_id]["status"] = "error"
        results[client_id]["error"] = f"Connection closed: {e}"
        results[client_id]["end_time"] = time.time()
        print(f"[Client {client_id}] ❌ Connection closed: {e}")
    except Exception as e:
        results[client_id]["status"] = "error"
        results[client_id]["error"] = str(e)
        results[client_id]["end_time"] = time.time()
        print(f"[Client {client_id}] ❌ Error: {e}")


async def run_concurrent_clients(
    audio_file: str,
    num_clients: int,
    ws_url: str,
    verbose: bool = True,
    stagger_delay: float = 0.1
) -> Dict[int, Dict[str, Any]]:
    """Run multiple concurrent clients."""
    print(f"📂 Loading audio file: {audio_file}")
    audio_data = load_and_prepare_audio(audio_file)
    audio_duration = len(audio_data) / TARGET_SAMPLE_RATE
    print(f"🎵 Audio loaded: {len(audio_data)} samples ({audio_duration:.2f}s at {TARGET_SAMPLE_RATE}Hz)")
    
    results: Dict[int, Dict[str, Any]] = {}
    
    print(f"\n🚀 Starting {num_clients} concurrent clients...")
    print(f"🎯 Target: {ws_url}")
    if stagger_delay > 0:
        print(f"⏱️  Stagger delay: {stagger_delay}s between clients")
    print("-" * 60)
    
    start_time = time.time()
    
    # Create client tasks with optional stagger
    tasks = []
    for i in range(num_clients):
        task = asyncio.create_task(
            simulate_client(i, audio_data, ws_url, results, verbose)
        )
        tasks.append(task)
        if stagger_delay > 0 and i < num_clients - 1:
            await asyncio.sleep(stagger_delay)
    
    # Wait for all clients to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    completed = sum(1 for r in results.values() if r["status"] == "completed")
    errors = sum(1 for r in results.values() if r["status"] == "error")
    total_finals = sum(r["final_count"] for r in results.values())
    total_interims = sum(r["interim_count"] for r in results.values())
    
    all_transcripts = []
    for r in results.values():
        all_transcripts.extend(r["transcripts"])
    
    durations = [
        r["end_time"] - r["start_time"]
        for r in results.values()
        if r["end_time"] is not None
    ]
    
    print(f"👥 Clients: {num_clients} total, {completed} completed, {errors} errors")
    print(f"📝 Results: {total_finals} final transcripts, {total_interims} interim results")
    print(f"⏱️  Total test time: {total_time:.2f}s")
    
    if durations:
        print(f"📈 Per-client duration: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
    
    if all_transcripts:
        print(f"\n📋 All final transcripts ({len(all_transcripts)} total):")
        for i, t in enumerate(all_transcripts[:20]):  # Show first 20
            print(f"  {i+1}. \"{t}\"")
        if len(all_transcripts) > 20:
            print(f"  ... and {len(all_transcripts) - 20} more")
    
    if errors > 0:
        print(f"\n❌ Errors ({errors} total):")
        for client_id, r in results.items():
            if r["status"] == "error":
                print(f"  Client {client_id}: {r['error']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test STT gateway with concurrent WebSocket clients",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with 5 concurrent clients using default localhost
    python concurrent_client_test.py audio.wav 5

    # Test with 10 clients on a remote server
    python concurrent_client_test.py audio.wav 10 --url ws://server:8089

    # Quick test with no stagger and minimal output
    python concurrent_client_test.py audio.wav 20 --stagger 0 --quiet
        """
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV format)"
    )
    parser.add_argument(
        "num_clients",
        type=int,
        help="Number of concurrent clients to simulate"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="ws://localhost:8089",
        help="WebSocket server URL (default: ws://localhost:8089)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-message output (only show summary)"
    )
    parser.add_argument(
        "--stagger",
        type=float,
        default=0.0,
        help="Delay between starting clients in seconds (default: 0.1, use 0 for all at once)"
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
        if HAS_PYDUB:
            print(f"⚠️  Warning: Format {audio_path.suffix} may not be supported")
        else:
            print(f"❌ Error: Format {audio_path.suffix} not supported. Install pydub for more formats:")
            print(f"   pip install pydub")
            return 1
    
    # Validate num_clients
    if args.num_clients < 1:
        print(f"❌ Error: num_clients must be at least 1")
        return 1
    
    if args.num_clients > 100:
        print(f"⚠️  Warning: Testing with {args.num_clients} clients may strain the server")
    
    # Run the test
    try:
        asyncio.run(
            run_concurrent_clients(
                args.audio_file,
                args.num_clients,
                args.url,
                verbose=not args.quiet,
                stagger_delay=args.stagger
            )
        )
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 130
    
    return 0


if __name__ == "__main__":
    exit(main())
