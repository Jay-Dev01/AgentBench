"""Fine-grained analyzer that augments ALFWorld runs with uncertainty metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..uncertainty.quantifier import UncertaintyQuantifier
from .helpers import configure_call_llm, extract_action_only, regenerate_action_samples


LLMCall = Callable[[str, float], str]


@dataclass
class StepContext:
    """Container holding the minimal information required for analysis."""

    idx: int
    task_description: str
    context_text: str
    modules: Dict[str, Any]


@dataclass
class FineGrainedAnalyzer:
    """Perform step-level uncertainty quantification for AgentBench runs."""

    call_fn: Optional[LLMCall] = None
    threshold: float = 0.35
    k: int = 3
    temperature: float = 0.7
    enable_self_correction: bool = True
    track_history: bool = True
    quantifier: UncertaintyQuantifier = field(init=False)
    _step_summaries: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.quantifier = UncertaintyQuantifier(threshold=self.threshold)
        if self.call_fn is not None:
            configure_call_llm(self.call_fn)

    def set_call_fn(self, fn: LLMCall) -> None:
        """Assign the LLM call dispatcher post-instantiation."""

        self.call_fn = fn
        configure_call_llm(fn)

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------
    def analyze_step(
        self,
        *,
        step_idx: int,
        task_description: str,
        context_text: str,
        modules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute uncertainty and optional self-correction for a single step."""

        base_action = (modules.get("action") or "").strip()
        summary: Dict[str, Any] = {
            "step_idx": step_idx,
            "modules": modules,
        }

        if not base_action:
            summary["uncertainty"] = {
                "step": {"score": None, "status": "confident"},
                "details": [],
            }
            if self.track_history:
                self._step_summaries.append(summary)
            return summary

        samples = regenerate_action_samples(
            task_description,
            context_text,
            base_action,
            k=self.k,
            temperature=self.temperature,
            call_fn=self._dispatch_llm,
        )

        uncertainty = self.quantifier.from_samples(
            step_idx=step_idx, module="action", samples=samples
        )
        uncertainty["meta"].update({"samples": samples})

        summary["uncertainty"] = {
            "step": {
                "score": uncertainty["score"],
                "status": uncertainty["status"],
            },
            "details": [uncertainty],
        }

        if self.enable_self_correction and uncertainty["score"] > self.threshold:
            self._apply_self_correction(
                summary=summary,
                modules=modules,
                task_description=task_description,
                context_text=context_text,
                prior_action=base_action,
            )
        else:
            summary["self_correction"] = {"fired": False}

        if self.track_history:
            self._step_summaries.append(summary)

        return summary

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def aggregate_totals(
        self, summaries: Optional[Sequence[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Compute aggregate metrics across processed steps."""

        records = summaries if summaries is not None else self._step_summaries
        scores = [
            s["uncertainty"]["step"]["score"]
            for s in records
            if s.get("uncertainty") and s["uncertainty"]["step"].get("score") is not None
        ]
        if not scores:
            return {"avg_uncertainty": 0.0, "low_conf_steps": 0}

        avg_uncertainty = sum(scores) / len(scores)
        low_conf_steps = sum(1 for score in scores if score > self.threshold)
        return {
            "avg_uncertainty": avg_uncertainty,
            "low_conf_steps": low_conf_steps,
        }

    def attach_totals(
        self,
        result: Dict[str, Any],
        summaries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Add aggregate totals into an existing result mapping."""

        totals = self.aggregate_totals(summaries)
        result.setdefault("totals", {})
        result["totals"].update(totals)
        return result

    def reset(self) -> None:
        """Clear stored summaries."""

        self._step_summaries.clear()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _dispatch_llm(self, prompt: str, temperature: float) -> str:
        if self.call_fn is None:
            raise RuntimeError(
                "FineGrainedAnalyzer requires a call_fn to generate action samples."
            )
        return self.call_fn(prompt, temperature)

    def _apply_self_correction(
        self,
        *,
        summary: Dict[str, Any],
        modules: Dict[str, Any],
        task_description: str,
        context_text: str,
        prior_action: str,
    ) -> None:
        """Invoke a self-correction LLM prompt when uncertainty is high."""

        stripped_context = context_text.strip()
        correction_prompt = (
            "Your previous action had low confidence."
            f"\nTask: {task_description.strip()}"
            + (f"\nContext: {stripped_context}" if stripped_context else "")
            + "\nPrevious action: <action>{}</action>"
            "\nReflect on potential mistakes inside <reflection>...</reflection>."
            "\nThen produce a refined <action>...</action>."
        ).format(prior_action.strip())

        raw = self._dispatch_llm(correction_prompt, self.temperature)
        corrected_action = extract_action_only(raw) or prior_action
        modules["action"] = corrected_action
        summary["self_correction"] = {
            "fired": True,
            "reason": "uncertainty>threshold",
            "raw": raw,
            "updated_action": corrected_action,
        }


__all__ = ["FineGrainedAnalyzer", "StepContext"]

