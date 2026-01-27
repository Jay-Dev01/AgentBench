"""Scoring system for agent evaluation with uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StepResult:
    """Result of a single step."""
    step_idx: int
    action: str
    success: bool
    confidence: float
    error: Optional[str] = None


@dataclass
class ProgressScore:
    """Score based on progress through the task."""
    completed_steps: int
    total_steps: int
    score: float  # 0-1
    
    @classmethod
    def compute(cls, completed: int, total: int) -> "ProgressScore":
        if total == 0:
            return cls(completed_steps=0, total_steps=0, score=0.0)
        return cls(
            completed_steps=completed,
            total_steps=total,
            score=completed / total,
        )


@dataclass
class CompletionScore:
    """Score based on task completion."""
    is_complete: bool
    score: float  # 0 or 1
    
    @classmethod
    def compute(cls, is_complete: bool) -> "CompletionScore":
        return cls(is_complete=is_complete, score=1.0 if is_complete else 0.0)


@dataclass
class CompositeScore:
    """Combined score with configurable weights."""
    progress: ProgressScore
    completion: CompletionScore
    score: float
    weights: Dict[str, float]
    
    @classmethod
    def compute(
        cls,
        progress: ProgressScore,
        completion: CompletionScore,
        progress_weight: float = 0.3,
        completion_weight: float = 0.7,
    ) -> "CompositeScore":
        weights = {
            "progress": progress_weight,
            "completion": completion_weight,
        }
        score = (
            progress_weight * progress.score +
            completion_weight * completion.score
        )
        return cls(
            progress=progress,
            completion=completion,
            score=score,
            weights=weights,
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result for a run."""
    progress: ProgressScore
    completion: CompletionScore
    composite: CompositeScore
    mean_confidence: float
    min_confidence: float
    uncertainty_correlation: Optional[float] = None


class Scorer:
    """
    Score agent runs with uncertainty awareness.
    """
    
    def __init__(
        self,
        progress_weight: float = 0.3,
        completion_weight: float = 0.7,
    ):
        self.progress_weight = progress_weight
        self.completion_weight = completion_weight
    
    def evaluate(
        self,
        steps: List[StepResult],
        task_completed: bool,
        expected_steps: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate a run and compute scores.
        
        Args:
            steps: List of step results
            task_completed: Whether the task was completed successfully
            expected_steps: Expected number of steps (if known)
        
        Returns:
            Complete evaluation result
        """
        # Compute progress
        successful_steps = sum(1 for s in steps if s.success)
        total_steps = expected_steps if expected_steps else len(steps)
        progress = ProgressScore.compute(successful_steps, total_steps)
        
        # Compute completion
        completion = CompletionScore.compute(task_completed)
        
        # Compute composite
        composite = CompositeScore.compute(
            progress=progress,
            completion=completion,
            progress_weight=self.progress_weight,
            completion_weight=self.completion_weight,
        )
        
        # Confidence statistics
        confidences = [s.confidence for s in steps] if steps else [0.0]
        mean_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        
        # Uncertainty-success correlation (if enough data)
        uncertainty_correlation = None
        if len(steps) >= 3:
            uncertainty_correlation = self._compute_correlation(steps)
        
        return EvaluationResult(
            progress=progress,
            completion=completion,
            composite=composite,
            mean_confidence=mean_confidence,
            min_confidence=min_confidence,
            uncertainty_correlation=uncertainty_correlation,
        )
    
    def _compute_correlation(self, steps: List[StepResult]) -> float:
        """Compute correlation between confidence and success."""
        if not steps:
            return 0.0
        
        # Simple point-biserial correlation approximation
        successes = [s.confidence for s in steps if s.success]
        failures = [s.confidence for s in steps if not s.success]
        
        if not successes or not failures:
            return 0.0
        
        mean_success = sum(successes) / len(successes)
        mean_failure = sum(failures) / len(failures)
        
        # Direction: positive if higher confidence -> more success
        return mean_success - mean_failure
    
    def compute_uncertainty_weighted_score(
        self,
        steps: List[StepResult],
        base_score: float,
    ) -> float:
        """
        Adjust score based on uncertainty.
        
        High confidence + success = bonus
        High confidence + failure = penalty
        Low confidence = neutral
        """
        if not steps:
            return base_score
        
        adjustments = []
        for step in steps:
            if step.confidence > 0.7:
                if step.success:
                    adjustments.append(0.02)  # Small bonus
                else:
                    adjustments.append(-0.05)  # Penalty for overconfidence
            elif step.confidence < 0.4:
                if not step.success:
                    adjustments.append(0.01)  # Small credit for knowing uncertainty
        
        total_adjustment = sum(adjustments)
        return max(0.0, min(1.0, base_score + total_adjustment))


__all__ = [
    "Scorer",
    "StepResult",
    "ProgressScore",
    "CompletionScore",
    "CompositeScore",
    "EvaluationResult",
]

