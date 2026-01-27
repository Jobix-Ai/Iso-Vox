"""
Conversation State Module
Manages conversation history and provides real-time adaptive endpointing based on speaker disfluency patterns.
"""

import re
from typing import List, Dict
from loguru import logger


class ConversationState:
    """
    Manages conversation state including history and adaptive endpointing.
    
    Features:
    - Stores conversation history (agent and user utterances)
    - Tracks disfluency patterns (um, uh, hmm, etc.)
    - Dynamically adjusts endpointing thresholds based on speech fluency
    
    High disfluency ratio indicates speaker needs more time to formulate thoughts.
    Low disfluency ratio indicates fluent speaker who can be interrupted sooner.
    """
    
    def __init__(
        self,
        client_id: str,
        base_endpointing_frames: int,
        max_endpointing_frames: int,
        vad_frame_ms: int,
        disfluency_enabled: bool = True,
        min_words_for_adaptation: int = 10,
    ):
        """
        Initialize conversation state and speaker adaptation system.
        
        Args:
            client_id: Client identifier for logging
            base_endpointing_frames: Starting endpointing threshold in frames
            max_endpointing_frames: Maximum allowed endpointing in frames
            vad_frame_ms: Duration of each VAD frame in milliseconds
            disfluency_enabled: Master switch for adaptation
            min_words_for_adaptation: Minimum words before adapting from disfluencies
        """
        self.client_id = client_id
        self.vad_frame_ms = vad_frame_ms
        self.disfluency_enabled = disfluency_enabled
        
        # Conversation history
        self.conversation_history: List[Dict] = []
        
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
    
    def _adapt_from_disfluency_patterns(self) -> None:
        """
        Adjust endpointing based on disfluency ratio.
        
        Note: This adaptation is NOT accumulative - it's based on the initial values
        and the current disfluency ratio. This allows the system to adjust up or down
        based on the speaker's overall pattern.
        """
        if not self.disfluency_enabled or self.total_words < self.min_words_for_adaptation:
            return
        
        # Calculate disfluency ratio
        disfluency_ratio = self.disfluency_count / self.total_words if self.total_words > 0 else 0
        old_endpointing_frames = self.current_endpointing_frames
        old_max_endpointing_frames = self.current_max_endpointing_frames
        
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
        
        # Only log if thresholds actually changed
        if (self.current_endpointing_frames != old_endpointing_frames or 
            self.current_max_endpointing_frames != old_max_endpointing_frames):
            logger.info(
                f"[{self.client_id}] 📊 Adapted: "
                f"{old_endpointing_frames * self.vad_frame_ms}→{self.current_endpointing_frames * self.vad_frame_ms}ms | "
                f"disfluency={disfluency_ratio:.1%}"
            )
    
    def get_current_endpointing_frames(self) -> int:
        return self.current_endpointing_frames
    
    def get_current_max_endpointing_frames(self) -> int:
        return self.current_max_endpointing_frames
    
    def log_stats(self) -> None:
        """Log current adaptation statistics (simplified)."""
        disfluency_ratio = self.disfluency_count / self.total_words if self.total_words > 0 else 0
        base_endpointing_ms = self.base_endpointing_frames * self.vad_frame_ms
        base_max_endpointing_ms = self.base_max_endpointing_frames * self.vad_frame_ms
        
        logger.info(
            f"[{self.client_id}] Stats: endpointing={base_endpointing_ms}/{base_max_endpointing_ms}ms | "
            f"disfluency={disfluency_ratio:.1%} | words={self.total_words}"
        )
    
    def add_agent_utterance(self, text: str) -> None:
        """Add an agent utterance to the conversation history."""
        if text:
            text = text.replace("\n", " ")
            self.conversation_history.append({"speaker": "assistant", "text": text})
    
    def add_user_utterance(self, text: str) -> None:
        """Add a user utterance to the conversation history."""
        if not text:
            return
        
        self.conversation_history.append({"speaker": "user", "text": text})

        if self.disfluency_enabled:
            words = text.split()
            if words:    
                self.total_words += len(words)
                disfluencies_in_transcript = len(self.disfluency_pattern.findall(text))
                self.disfluency_count += disfluencies_in_transcript
                
                self._adapt_from_disfluency_patterns()
    
    def get_conversation_history(self, n: int = None, exclude_last_user: bool = False) -> List[Dict]:
        conversation_history = self.conversation_history[-n:] if n else self.conversation_history
        if exclude_last_user and conversation_history and conversation_history[-1]['speaker'] == 'user':
            conversation_history = conversation_history[:-1]
        if conversation_history:
            conversation_text = "\n".join([
                f"  {entry['speaker'].capitalize()}: {entry['text']}" 
                for entry in conversation_history
            ])
        else:
            conversation_text = "[Starting of conversation]"
        return conversation_text
    
    def clear_conversation_history(self) -> None:
        """Clear all conversation history."""
        self.conversation_history.clear()
        logger.info(f"[{self.client_id}] Conversation history cleared")

