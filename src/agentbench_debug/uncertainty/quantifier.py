"""Basic uncertainty quantification based on sample disagreement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


class UOut(TypedDict):
    """Output structure for uncertainty quantification."""
    step_idx: int
    module: str
    method: str
    score: float
    status: str
    meta: Dict[str, Any]


@dataclass
class UncertaintyQuantifier:
    """Quantify uncertainty based on disagreement among sampled outputs."""
    
    threshold: float = 0.35
    
    def from_samples(
        self,
        step_idx: int,
        module: str,
        samples: List[str],
    ) -> UOut:
        """
        Compute uncertainty from a list of sampled outputs.
        
        Uses entropy-based disagreement: if all samples agree, uncertainty is 0.
        If samples are maximally diverse, uncertainty approaches 1.
        """
        if not samples:
            return UOut(
                step_idx=step_idx,
                module=module,
                method="self_consistency",
                score=0.0,
                status="confident",
                meta={"k": 0, "threshold": self.threshold},
            )
        
        # Count occurrences of each unique sample
        counts = Counter(s.strip().lower() for s in samples if s.strip())
        
        if not counts:
            return UOut(
                step_idx=step_idx,
                module=module,
                method="self_consistency",
                score=0.0,
                status="confident",
                meta={"k": len(samples), "threshold": self.threshold},
            )
        
        total = sum(counts.values())
        
        # Compute disagreement score (1 - max_agreement_ratio)
        max_count = max(counts.values())
        agreement_ratio = max_count / total
        score = 1.0 - agreement_ratio
        
        status = "low_confidence" if score > self.threshold else "confident"
        
        return UOut(
            step_idx=step_idx,
            module=module,
            method="self_consistency",
            score=score,
            status=status,
            meta={
                "k": len(samples),
                "threshold": self.threshold,
                "unique_samples": len(counts),
                "max_agreement": agreement_ratio,
            },
        )
    
    def from_logprobs(
        self,
        step_idx: int,
        module: str,
        logprobs: List[float],
    ) -> UOut:
        """
        Compute uncertainty from token-level log probabilities.
        
        Uses geometric mean of token probabilities.
        """
        import math
        
        if not logprobs:
            return UOut(
                step_idx=step_idx,
                module=module,
                method="logprob",
                score=0.5,  # Default uncertainty when no logprobs
                status="uncertain",
                meta={"n_tokens": 0, "threshold": self.threshold},
            )
        
        # Geometric mean of probabilities
        avg_logprob = sum(logprobs) / len(logprobs)
        confidence = math.exp(avg_logprob)
        score = 1.0 - confidence
        
        status = "low_confidence" if score > self.threshold else "confident"
        
        return UOut(
            step_idx=step_idx,
            module=module,
            method="logprob",
            score=score,
            status=status,
            meta={
                "n_tokens": len(logprobs),
                "avg_logprob": avg_logprob,
                "confidence": confidence,
                "threshold": self.threshold,
            },
        )


__all__ = ["UncertaintyQuantifier", "UOut"]

