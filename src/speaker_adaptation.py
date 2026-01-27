"""
Speaker Adaptation Module
Provides real-time adaptive endpointing based on speaker disfluency patterns.
"""

import re
from loguru import logger


class SpeakerAdapter:
    """
    Adapts endpointing parameters based on speaker disfluency patterns.
    
    Tracks disfluency patterns (um, uh, hmm, etc.) and dynamically adjusts
    endpointing thresholds based on speech fluency.
    
    High disfluency ratio indicates speaker needs more time to formulate thoughts.
    Low disfluency ratio indicates fluent speaker who can be interrupted sooner.
    """
    
    def __init__(
        self,
        client_id: str,
        base_endpointing_frames: int,
        max_endpointing_frames: int,
        vad_frame_ms: int,
        enabled: bool = True,
        min_words_for_adaptation: int = 10,
    ):
        """
        Initialize speaker adaptation system.
        
        Args:
            client_id: Client identifier for logging
            base_endpointing_frames: Starting endpointing threshold in frames
            max_endpointing_frames: Maximum allowed endpointing in frames
            vad_frame_ms: Duration of each VAD frame in milliseconds
            enabled: Master switch for adaptation
            min_words_for_adaptation: Minimum words before adapting from disfluencies
        """
        self.client_id = client_id
        self.vad_frame_ms = vad_frame_ms
        self.enabled = enabled
        
        # Endpointing parameters
        self.base_endpointing_frames = base_endpointing_frames
        self.base_max_endpointing_frames = max_endpointing_frames
        self.current_endpointing_frames = base_endpointing_frames
        self.current_max_endpointing_frames = max_endpointing_frames
        
        # Disfluency tracking
        self.disfluency_words = {"um", "uh", "hmm", "mm", "hm", "er", "eh", "ah"}
        # Precompile regex pattern for faster matching
        self.disfluency_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in self.disfluency_words) + r')\b',
            re.IGNORECASE
        )
        self.total_words = 0
        self.disfluency_count = 0
        self.min_words_for_adaptation = min_words_for_adaptation
    
    def track_transcript(self, transcript: str) -> None:
        """
        Track transcription to analyze disfluency patterns.
        Optimized with regex pattern matching for faster processing.
        """
        if not self.enabled or not transcript:
            return
        
        # Split into words for counting
        words = transcript.split()
        if not words:
            return
        
        # Fast regex-based disfluency counting (much faster than iteration)
        disfluencies_in_transcript = len(self.disfluency_pattern.findall(transcript))
        
        # Update totals
        self.total_words += len(words)
        self.disfluency_count += disfluencies_in_transcript
        
        # Only log disfluencies if significant (reduces log spam)
        if disfluencies_in_transcript > 0 and self.total_words % 50 == 0:  # Log every 50 words
            logger.debug(
                f"[{self.client_id}] Disfluencies: {self.disfluency_count}/{self.total_words} words "
                f"({self.disfluency_count/self.total_words:.1%})"
            )
        
        # Attempt adaptation if we have enough samples
        self._adapt_from_disfluency_patterns()
    
    def _adapt_from_disfluency_patterns(self) -> None:
        """
        Adjust endpointing based on disfluency ratio.
        
        Note: This adaptation is NOT accumulative - it's based on the initial values
        and the current disfluency ratio. This allows the system to adjust up or down
        based on the speaker's overall pattern.
        """
        if not self.enabled or self.total_words < self.min_words_for_adaptation:
            return
        
        # Calculate disfluency ratio
        disfluency_ratio = self.disfluency_count / self.total_words if self.total_words > 0 else 0
        old_endpointing_ms = self.current_endpointing_frames * self.vad_frame_ms
        old_max_endpointing_ms = self.current_max_endpointing_frames * self.vad_frame_ms
        
        # Calculate new thresholds based on INITIAL values (not accumulative)
        if disfluency_ratio > 0.2:
            # High disfluency ratio: speaker pauses to think, needs more time
            self.current_endpointing_frames = int(self.base_endpointing_frames * 1.2)
            self.current_max_endpointing_frames = int(self.base_max_endpointing_frames * 1.3)
        elif disfluency_ratio > 0.1:
            self.current_endpointing_frames = int(self.base_endpointing_frames * 1.1)
            self.current_max_endpointing_frames = int(self.base_max_endpointing_frames * 1.2)
        else:
            self.current_endpointing_frames = self.base_endpointing_frames
            self.current_max_endpointing_frames = self.base_max_endpointing_frames
        
        new_endpointing_ms = self.current_endpointing_frames * self.vad_frame_ms
        new_max_endpointing_ms = self.current_max_endpointing_frames * self.vad_frame_ms
        
        # Only log if thresholds actually changed
        if new_endpointing_ms != old_endpointing_ms or new_max_endpointing_ms != old_max_endpointing_ms:
            logger.info(
                f"[{self.client_id}] 📊 Adapted: min={old_endpointing_ms}→{new_endpointing_ms}ms, "
                f"max={old_max_endpointing_ms}→{new_max_endpointing_ms}ms (disfluency={disfluency_ratio:.1%})"
            )
    
    def register_speech_start(self) -> None:
        """Called when speech starts. No-op kept for API compatibility."""
        pass
    
    def register_finalization(self) -> None:
        """Called when speech is finalized. No-op kept for API compatibility."""
        pass
    
    def reset_utterance_state(self) -> None:
        """
        Reset state for new utterance (but keep learned patterns).
        """
        pass  # No per-utterance state to reset for disfluency tracking
    
    def get_current_endpointing_frames(self) -> int:
        """
        Get the current adaptive minimum endpointing threshold in frames.
        
        Returns:
            Current minimum endpointing threshold in frames
        """
        return self.current_endpointing_frames
    
    def get_current_max_endpointing_frames(self) -> int:
        """
        Get the current adaptive maximum endpointing threshold in frames.
        
        Returns:
            Current maximum endpointing threshold in frames
        """
        return self.current_max_endpointing_frames
    
    def get_adaptation_stats(self) -> dict:
        """
        Get statistics about adaptations performed.
        
        Returns:
            Dictionary with adaptation statistics
        """
        disfluency_ratio = self.disfluency_count / self.total_words if self.total_words > 0 else 0
        return {
            "enabled": self.enabled,
            "current_min_endpointing_ms": self.current_endpointing_frames * self.vad_frame_ms,
            "current_max_endpointing_ms": self.current_max_endpointing_frames * self.vad_frame_ms,
            "base_endpointing_ms": self.base_endpointing_frames * self.vad_frame_ms,
            "base_max_endpointing_ms": self.base_max_endpointing_frames * self.vad_frame_ms,
            "total_words": self.total_words,
            "disfluency_count": self.disfluency_count,
            "disfluency_ratio": disfluency_ratio,
        }
    
    def log_stats(self) -> None:
        """Log current adaptation statistics (simplified)."""
        stats = self.get_adaptation_stats()
        logger.info(
            f"[{self.client_id}] Stats: endpointing={stats['base_endpointing_ms']}/{stats['base_max_endpointing_ms']}ms | "
            f"disfluency={stats['disfluency_ratio']:.1%} | words={stats['total_words']}"
        )

