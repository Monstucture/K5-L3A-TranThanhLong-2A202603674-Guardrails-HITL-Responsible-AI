"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import asyncio
import os
import re
import textwrap
import time

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin
from google.adk.models.google_llm import Gemini

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent

BLOCKED_MESSAGE = (
    "The response was blocked due to safety concerns. "
    "Please contact support for assistance."
)


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    redacted = response

    # PII patterns to check
    PII_PATTERNS = {
        # TODO: Add regex patterns for:
        # - VN phone number: r"0\d{9,10}"
        # - Email: r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"
        # - National ID (CMND/CCCD): r"\b\d{9}\b|\b\d{12}\b"
        # - API key pattern: r"sk-[a-zA-Z0-9-]+"
        # - Password pattern: r"password\s*[:=]\s*\S+"
        "vn_phone": r"(?<!\d)(?:\+?84|0)\d{9,10}(?!\d)",
        "email": r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}",
        "national_id": r"(?<!\d)(?:\d{9}|\d{12})(?!\d)",
        "api_key": r"sk-[A-Za-z0-9_-]+",
        "password": r"password\s*(?:is|[:=])\s*\S+",   
        "db_host": r"db\.vinbank\.internal(?::\d+)?",
    }

    for name, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    # Exact protected values (e.g. "admin123") have no keyword, so match them literally
    for secret in DEMO_SECRETS:
        if secret and secret in redacted:
            issues.append("protected value: 1 found")
            redacted = redacted.replace(secret, "[REDACTED]")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Refusals, redirects and polite declines (e.g. "I can only help with banking") are SAFE.

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

# Free tier limits per model: 5 requests/minute, 20 requests/day.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash")
JUDGE_MIN_INTERVAL_S = float(os.getenv("JUDGE_MIN_INTERVAL_S", "13"))  # 60s / 5 RPM + margin
JUDGE_MAX_CALLS = int(os.getenv("JUDGE_MAX_CALLS", "10"))  # per run; keeps the 20/day cap safe

safety_judge_agent = llm_agent.LlmAgent(
    # attempts=1: no silent SDK retries, so 1 judge call = exactly 1 API request
    model=Gemini(model=JUDGE_MODEL, retry_options=types.HttpRetryOptions(attempts=1)),
    name="safety_judge",
    instruction=SAFETY_JUDGE_INSTRUCTION,
)

judge_runner = None  # Will be initialized later


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


class JudgeUnavailable(Exception):
    """Judge skipped: per-run budget used up, or Gemini rate-limited (429) / overloaded (503)."""


_judge_lock = asyncio.Lock()
_judge_last_call = 0.0
_judge_calls = 0
_judge_cache: dict = {}


def _is_quota_error(exc) -> bool:
    """True if any exception in the chain is a 429 or 503 (ADK wraps the genai error)."""
    for _ in range(10):  # bounded walk of __cause__/__context__
        if exc is None:
            return False
        if getattr(exc, "code", None) in (429, 503):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Calls are serialized, spaced JUDGE_MIN_INTERVAL_S apart, capped at
    JUDGE_MAX_CALLS per run, and cached by text.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)

    Raises:
        JudgeUnavailable: budget used up, or Gemini returned 429/503.
    """
    global _judge_last_call, _judge_calls

    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}
    if response_text in _judge_cache:
        return _judge_cache[response_text]

    async with _judge_lock:
        if _judge_calls >= JUDGE_MAX_CALLS:
            raise JudgeUnavailable(f"judge budget used ({JUDGE_MAX_CALLS} calls this run)")
        wait = JUDGE_MIN_INTERVAL_S - (time.monotonic() - _judge_last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        _judge_calls += 1
        _judge_last_call = time.monotonic()  # failed calls still count toward spacing

        prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
        try:
            verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
        except Exception as e:
            if _is_quota_error(e):
                raise JudgeUnavailable(f"Gemini rate-limited/overloaded: {str(e)[:100]}") from e
            raise

    # Only the first word decides: the reason line may contain "unsafe" or "safe".
    # Empty or unexpected output counts as UNSAFE (fail-closed).
    first_word = (verdict.strip().split() or [""])[0].strip(".:,!*").upper()
    result = {"safe": first_word == "SAFE", "verdict": verdict.strip()}
    _judge_cache[response_text] = result
    return result


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):  # judge is optional; needs GOOGLE_API_KEY
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        if self.use_llm_judge:
            _init_judge()
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0
        self.judge_skipped_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # TODO: Implement logic:
        # 1. Call content_filter(response_text)
        #    - If issues found: replace llm_response.content with redacted version
        #    - Increment self.redacted_count
        # 2. If use_llm_judge: call llm_safety_check(response_text)
        #    - If unsafe: replace llm_response.content with a safe message
        #    - Increment self.blocked_count
        # 3. Return llm_response (possibly modified)
        # Layer 1: regex redaction (PII + protected secrets)
        result = content_filter(response_text)
        text = response_text
        if not result["safe"]:
            text = result["redacted"]
            self.redacted_count += 1
            llm_response.content = types.Content(role="model", parts=[types.Part(text=text)])

        # Layer 2 (optional): judge the REDACTED text so secrets never reach the judge
        if self.use_llm_judge:
            try:
                is_safe = (await llm_safety_check(text))["safe"]
            except JudgeUnavailable as e:
                # Quota/outage: the regex layer already ran, so keep its result
                # rather than blocking every reply (free tier is only 20 calls/day).
                is_safe = True
                self.judge_skipped_count += 1
                print(f"[output_guardrail] judge skipped: {e}")
            except Exception:
                is_safe = False  # fail-closed on any other judge error
            if not is_safe:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model", parts=[types.Part(text=BLOCKED_MESSAGE)]
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
