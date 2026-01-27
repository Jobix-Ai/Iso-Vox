#!/usr/bin/env python3
"""
Convert debug audio files from raw PCM to WAV format for easier playback and analysis.
"""

import os
import sys
import glob
import wave
import numpy as np
from pathlib import Path

DEBUG_AUDIO_DIR = "debug_audio"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
CHANNELS = 1


def raw_to_wav(raw_path: str, wav_path: str) -> None:
    """Convert raw PCM file to WAV format."""
    # Read raw audio
    audio_data = np.fromfile(raw_path, dtype=np.int16)
    
    # Write WAV file
    with wave.open(wav_path, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✓ Converted: {os.path.basename(wav_path)}")


def convert_all_raw_files(directory: str = DEBUG_AUDIO_DIR) -> int:
    """Convert all .raw files in directory to .wav format."""
    raw_files = glob.glob(os.path.join(directory, "*.raw"))
    
    if not raw_files:
        print(f"No .raw files found in {directory}/")
        return 0
    
    print(f"Found {len(raw_files)} raw audio files")
    print()
    
    converted = 0
    for raw_path in sorted(raw_files):
        wav_path = raw_path.replace(".raw", ".wav")
        
        # Skip if WAV already exists and is newer
        if os.path.exists(wav_path):
            raw_mtime = os.path.getmtime(raw_path)
            wav_mtime = os.path.getmtime(wav_path)
            if wav_mtime >= raw_mtime:
                print(f"⊘ Skipped (already exists): {os.path.basename(wav_path)}")
                continue
        
        try:
            raw_to_wav(raw_path, wav_path)
            converted += 1
        except Exception as e:
            print(f"✗ Error converting {os.path.basename(raw_path)}: {e}")
    
    print()
    print(f"Converted {converted} files")
    return converted


def analyze_audio_file(raw_path: str) -> None:
    """Analyze and display information about a raw audio file."""
    if not os.path.exists(raw_path):
        print(f"Error: File not found: {raw_path}")
        return
    
    audio_data = np.fromfile(raw_path, dtype=np.int16)
    duration_s = len(audio_data) / SAMPLE_RATE
    
    # Calculate statistics
    audio_float = audio_data.astype(np.float32) / 32768.0
    rms_energy = np.sqrt(np.mean(audio_float ** 2))
    peak_amplitude = np.max(np.abs(audio_float))
    
    # Detect silence (frames with very low energy)
    frame_size = 512  # 32ms frames at 16kHz
    num_frames = len(audio_data) // frame_size
    silent_frames = 0
    
    for i in range(num_frames):
        frame = audio_float[i * frame_size:(i + 1) * frame_size]
        frame_energy = np.sqrt(np.mean(frame ** 2))
        if frame_energy < 0.01:  # Threshold for silence
            silent_frames += 1
    
    silence_ratio = silent_frames / num_frames if num_frames > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Audio File: {os.path.basename(raw_path)}")
    print(f"{'='*60}")
    print(f"Duration:        {duration_s:.2f}s ({duration_s * 1000:.0f}ms)")
    print(f"Samples:         {len(audio_data):,}")
    print(f"File Size:       {os.path.getsize(raw_path):,} bytes")
    print(f"RMS Energy:      {rms_energy:.4f}")
    print(f"Peak Amplitude:  {peak_amplitude:.4f}")
    print(f"Silence Ratio:   {silence_ratio:.1%} ({silent_frames}/{num_frames} frames)")
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help"]:
            print("Usage:")
            print("  python convert_debug_audio.py              # Convert all .raw files to .wav")
            print("  python convert_debug_audio.py <file.raw>   # Analyze a specific file")
            print()
            return
        
        # Analyze specific file
        analyze_audio_file(sys.argv[1])
        
        # Ask to convert
        wav_path = sys.argv[1].replace(".raw", ".wav")
        if not os.path.exists(wav_path):
            response = input(f"\nConvert to WAV? (y/n): ").strip().lower()
            if response == 'y':
                try:
                    raw_to_wav(sys.argv[1], wav_path)
                    print(f"Saved: {wav_path}")
                except Exception as e:
                    print(f"Error: {e}")
    else:
        # Convert all files
        if not os.path.exists(DEBUG_AUDIO_DIR):
            print(f"Error: Directory not found: {DEBUG_AUDIO_DIR}")
            print("Run the gateway with DEBUG_SAVE_AUDIO=true to generate audio files")
            return
        
        convert_all_raw_files(DEBUG_AUDIO_DIR)


if __name__ == "__main__":
    main()

