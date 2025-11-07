"""Helper utilities for fine-grained uncertainty analysis."""

from __future__ import annotations

import re
from typing import Callable, List, Optional

ACTION_REGEN_PROMPT = """You are an ALFWorld agent helper.

Task: {task}
Context: {context}

Re-generate ONLY the next <action>...</action>
"""


_CALL_LLM: Optional[Callable[[str, float], str]] = None


def configure_call_llm(fn: Callable[[str, float], str]) -> None:
    """Configure the default LLM call hook used by helper utilities."""

    global _CALL_LLM
    _CALL_LLM = fn


def call_llm(prompt: str, *, temperature: float = 0.7) -> str:
    """Dispatch an LLM call through the configured hook."""

    if _CALL_LLM is None:
        raise RuntimeError(
            "No call_llm function configured. Use configure_call_llm(fn) first."
        )
    return _CALL_LLM(prompt, temperature)


def extract_action_only(text: str) -> str:
    """Return the text encapsulated by <action> tags, if present."""

    if not text:
        return ""

    match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def regenerate_action_samples(
    task_description: str,
    context_text: str,
    base_action: str,
    *,
    k: int = 3,
    temperature: float = 0.7,
    call_fn: Optional[Callable[[str, float], str]] = None,
) -> List[str]:
    """Generate `k` additional action candidates plus the provided baseline."""

    if not base_action:
        return []

    dispatcher = call_fn or (lambda prompt, temp: call_llm(prompt, temperature=temp))

    prompt = ACTION_REGEN_PROMPT.format(task=task_description.strip(), context=context_text.strip())
    outputs: List[str] = []

    for _ in range(k):
        raw = dispatcher(prompt, temperature)
        outputs.append(extract_action_only(raw))

    outputs.append(base_action.strip())
    return [out for out in outputs if out]


__all__ = [
    "ACTION_REGEN_PROMPT",
    "configure_call_llm",
    "extract_action_only",
    "regenerate_action_samples",
]

