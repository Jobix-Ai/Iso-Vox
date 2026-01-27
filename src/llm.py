import os
import re
from google import genai
from google.genai import types
from loguru import logger
from rapidfuzz.distance import Levenshtein
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

def is_user_speech(raw_text: str, avg_confidence: float, conversation_history: str) -> bool:
    """Check if transcript is from the primary user (vs background/other speaker)."""
    if not client:
        return True
    
    try:
        prompt = f"""You are an assistant making a phone call with a user.
Now you receive a new transcript recognized by an ASR model.
Decide if this transcript could be a potential answer from the primary caller speaking to you (not background noise or other speakers).
Return only "YES" or "NO".

### Conversation history:
{conversation_history}

### New transcript: "{raw_text}"
ASR confidence: {avg_confidence:.2f}
"""
        response = client.models.generate_content(
            model=gemini_model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        result = response.text.strip().upper()
        logger.info(f"LLM user check: '{raw_text}' -> {result}")
        return "YES" in result

    except Exception as e:
        logger.error(f"LLM check failed: {e}")
        return True


# Presence check detection
PRESENCE_PHRASES = [
    "are you there", "can you hear me", "you still there",
    "hello are you there", "hello can you hear me",
    "are you with me", "hello you still there",
]
PRESENCE_PHRASES_SPLIT = [p.split() for p in PRESENCE_PHRASES]


def is_presence_check(text: str, threshold: float = 0.4) -> bool:
    """Fast check if text is a presence check phrase."""
    text = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    tokens = text.split()
    
    if not tokens or len(tokens) > 7:
        return False

    for ref_tokens in PRESENCE_PHRASES_SPLIT:
        dist = Levenshtein.distance(tokens, ref_tokens)
        score = dist / max(len(tokens), len(ref_tokens))
        if score <= threshold:
            return True
    return False
