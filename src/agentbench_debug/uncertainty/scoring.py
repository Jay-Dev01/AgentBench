"""
Scoring System for API Orchestration Benchmark.

Implements the dual scoring metrics from API-ORCHA-Bench:
- Progress Score (S_progress): Incremental task completion across checkpoints
- Full Completion Score (S_full): Binary end-to-end task success
- Composite Score: Weighted combination with interaction term

Also provides checkpoint-based evaluation and uncertainty-weighted scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CheckpointType(Enum):
    """Types of workflow checkpoints."""
    AUTHENTICATION = "authentication"
    API_COORDINATION = "api_coordination"
    ERROR_RECOVERY = "error_recovery"
    DATA_CONSISTENCY = "data_consistency"
    UNCERTAINTY_ASSESSMENT = "uncertainty_assessment"
    FINAL_ARTIFACT = "final_artifact"


@dataclass
class Checkpoint:
    """A single checkpoint in the workflow."""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    description: str
    weight: float = 1.0              # Importance weight w_i
    completed: bool = False
    partial_score: float = 0.0       # For partial completion (0-1)
    uncertainty_at_completion: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressScore:
    """Progress score with breakdown."""
    score: float                     # S_progress = (1/n) * Σ w_i * c_i
    checkpoints_total: int
    checkpoints_completed: int
    checkpoints_partial: int
    weighted_completion: float       # Σ w_i * c_i
    total_weight: float              # Σ w_i
    per_checkpoint_scores: List[Tuple[str, float]]  # (checkpoint_id, score)


@dataclass
class FullCompletionScore:
    """Full completion score (binary)."""
    score: float                     # S_full ∈ {0, 1}
    completed: bool
    failure_reason: Optional[str]
    final_step_idx: int
    total_steps: int


@dataclass
class CompositeScore:
    """Composite score combining progress and completion."""
    score: float                     # S_composite = α * S_progress + β * S_full + γ * S_progress * S_full
    progress_score: float            # S_progress
    full_score: float                # S_full
    interaction_term: float          # S_progress * S_full
    alpha: float                     # Weight for progress
    beta: float                      # Weight for completion
    gamma: float                     # Weight for interaction
    interpretation: str              # Human-readable interpretation


@dataclass
class UncertaintyWeightedScore:
    """Score adjusted by uncertainty."""
    raw_score: float
    uncertainty_penalty: float
    adjusted_score: float
    high_uncertainty_steps: int
    mean_uncertainty: float


@dataclass
class FullEvaluationResult:
    """Complete evaluation result for a workflow run."""
    progress: ProgressScore
    completion: FullCompletionScore
    composite: CompositeScore
    uncertainty_weighted: UncertaintyWeightedScore
    checkpoints: List[Checkpoint]
    latency_ms: Optional[float]
    token_cost: Optional[int]
    robustness_score: Optional[float]
    safety_score: Optional[float]


class ScoringSystem:
    """
    Scoring system for API orchestration benchmark.
    
    Implements dual scoring metrics:
    1. Progress Score: Measures incremental completion
    2. Full Completion Score: Binary success indicator
    3. Composite Score: Balanced combination with interaction term
    """
    
    def __init__(
        self,
        alpha: float = 0.4,          # Progress weight
        beta: float = 0.3,           # Completion weight
        gamma: float = 0.3,          # Interaction weight
        uncertainty_penalty_factor: float = 0.2,
    ):
        """
        Initialize scoring system.
        
        Args:
            alpha: Weight for progress score in composite
            beta: Weight for full completion in composite
            gamma: Weight for interaction term (progress * completion)
            uncertainty_penalty_factor: Factor for uncertainty-based penalty
        
        Note: alpha + beta + gamma should equal 1.0
        """
        total = alpha + beta + gamma
        self.alpha = alpha / total  # Normalize
        self.beta = beta / total
        self.gamma = gamma / total
        self.uncertainty_penalty_factor = uncertainty_penalty_factor
        
        self._checkpoints: List[Checkpoint] = []
        self._step_uncertainties: List[float] = []
        self._total_steps = 0
        self._completed = False
        self._failure_reason: Optional[str] = None
    
    # =========================================================================
    # Checkpoint Management
    # =========================================================================
    
    def add_checkpoint(
        self,
        checkpoint_id: str,
        checkpoint_type: CheckpointType,
        description: str,
        weight: float = 1.0,
    ) -> Checkpoint:
        """
        Add a checkpoint to the evaluation.
        
        Args:
            checkpoint_id: Unique identifier
            checkpoint_type: Type of checkpoint
            description: Human-readable description
            weight: Importance weight (default 1.0)
        
        Returns:
            The created Checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            description=description,
            weight=weight,
        )
        self._checkpoints.append(checkpoint)
        return checkpoint
    
    def mark_checkpoint_completed(
        self,
        checkpoint_id: str,
        partial_score: float = 1.0,
        uncertainty: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a checkpoint as completed.
        
        Args:
            checkpoint_id: The checkpoint to mark
            partial_score: Score for partial completion (0-1, default 1.0 = full)
            uncertainty: Uncertainty score at completion time
            metadata: Additional data
        
        Returns:
            True if checkpoint was found and marked
        """
        for checkpoint in self._checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                checkpoint.completed = partial_score >= 1.0
                checkpoint.partial_score = min(1.0, max(0.0, partial_score))
                checkpoint.uncertainty_at_completion = uncertainty
                if metadata:
                    checkpoint.metadata.update(metadata)
                return True
        return False
    
    def add_step(self, uncertainty: float = 0.0) -> None:
        """Record a step with its uncertainty."""
        self._total_steps += 1
        self._step_uncertainties.append(uncertainty)
    
    def mark_workflow_complete(self, success: bool, failure_reason: Optional[str] = None) -> None:
        """Mark the workflow as complete."""
        self._completed = success
        if not success:
            self._failure_reason = failure_reason
    
    # =========================================================================
    # Score Computation
    # =========================================================================
    
    def compute_progress_score(self) -> ProgressScore:
        """
        Compute progress score.
        
        S_progress = (1/n) * Σ w_i * c_i
        
        where c_i is the partial_score (0-1) for each checkpoint.
        """
        if not self._checkpoints:
            return ProgressScore(
                score=0.0,
                checkpoints_total=0,
                checkpoints_completed=0,
                checkpoints_partial=0,
                weighted_completion=0.0,
                total_weight=0.0,
                per_checkpoint_scores=[],
            )
        
        n = len(self._checkpoints)
        weighted_sum = 0.0
        total_weight = 0.0
        completed = 0
        partial = 0
        per_checkpoint = []
        
        for cp in self._checkpoints:
            score = cp.partial_score * cp.weight
            weighted_sum += score
            total_weight += cp.weight
            per_checkpoint.append((cp.checkpoint_id, cp.partial_score))
            
            if cp.completed:
                completed += 1
            elif cp.partial_score > 0:
                partial += 1
        
        # Normalized score
        progress = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return ProgressScore(
            score=progress,
            checkpoints_total=n,
            checkpoints_completed=completed,
            checkpoints_partial=partial,
            weighted_completion=weighted_sum,
            total_weight=total_weight,
            per_checkpoint_scores=per_checkpoint,
        )
    
    def compute_full_completion_score(self) -> FullCompletionScore:
        """
        Compute full completion score (binary).
        
        S_full ∈ {0, 1}
        """
        return FullCompletionScore(
            score=1.0 if self._completed else 0.0,
            completed=self._completed,
            failure_reason=self._failure_reason,
            final_step_idx=self._total_steps - 1 if self._total_steps > 0 else 0,
            total_steps=self._total_steps,
        )
    
    def compute_composite_score(self) -> CompositeScore:
        """
        Compute composite score.
        
        S_composite = α * S_progress + β * S_full + γ * S_progress * S_full
        
        The interaction term γ * S_progress * S_full rewards high-progress completions.
        """
        progress = self.compute_progress_score()
        completion = self.compute_full_completion_score()
        
        s_prog = progress.score
        s_full = completion.score
        interaction = s_prog * s_full
        
        composite = (
            self.alpha * s_prog +
            self.beta * s_full +
            self.gamma * interaction
        )
        
        # Interpretation
        if composite >= 0.85:
            interpretation = "Excellent: High progress with successful completion"
        elif composite >= 0.70:
            interpretation = "Good: Strong progress with completion"
        elif composite >= 0.50:
            interpretation = "Moderate: Partial progress, may or may not complete"
        elif composite >= 0.30:
            interpretation = "Poor: Limited progress and/or failed completion"
        else:
            interpretation = "Failed: Minimal progress, workflow unsuccessful"
        
        return CompositeScore(
            score=composite,
            progress_score=s_prog,
            full_score=s_full,
            interaction_term=interaction,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            interpretation=interpretation,
        )
    
    def compute_uncertainty_weighted_score(
        self,
        base_score: Optional[float] = None,
        uncertainty_threshold: float = 0.5,
    ) -> UncertaintyWeightedScore:
        """
        Compute score adjusted by uncertainty.
        
        Higher uncertainty = higher penalty.
        
        Args:
            base_score: Score to adjust (defaults to composite score)
            uncertainty_threshold: Threshold for "high" uncertainty
        
        Returns:
            UncertaintyWeightedScore
        """
        if base_score is None:
            base_score = self.compute_composite_score().score
        
        if not self._step_uncertainties:
            return UncertaintyWeightedScore(
                raw_score=base_score,
                uncertainty_penalty=0.0,
                adjusted_score=base_score,
                high_uncertainty_steps=0,
                mean_uncertainty=0.0,
            )
        
        mean_unc = sum(self._step_uncertainties) / len(self._step_uncertainties)
        high_unc_steps = sum(1 for u in self._step_uncertainties if u > uncertainty_threshold)
        
        # Penalty: proportional to mean uncertainty and fraction of high-uncertainty steps
        high_unc_fraction = high_unc_steps / len(self._step_uncertainties)
        penalty = self.uncertainty_penalty_factor * (mean_unc + high_unc_fraction) / 2
        
        adjusted = base_score * (1.0 - penalty)
        
        return UncertaintyWeightedScore(
            raw_score=base_score,
            uncertainty_penalty=penalty,
            adjusted_score=max(0.0, adjusted),
            high_uncertainty_steps=high_unc_steps,
            mean_uncertainty=mean_unc,
        )
    
    def compute_full_evaluation(
        self,
        latency_ms: Optional[float] = None,
        token_cost: Optional[int] = None,
        robustness_score: Optional[float] = None,
        safety_score: Optional[float] = None,
    ) -> FullEvaluationResult:
        """
        Compute complete evaluation with all metrics.
        
        Args:
            latency_ms: Total latency in milliseconds
            token_cost: Total token cost (prompt + completion)
            robustness_score: Optional robustness evaluation
            safety_score: Optional safety evaluation
        
        Returns:
            FullEvaluationResult with all metrics
        """
        progress = self.compute_progress_score()
        completion = self.compute_full_completion_score()
        composite = self.compute_composite_score()
        uncertainty_weighted = self.compute_uncertainty_weighted_score(composite.score)
        
        return FullEvaluationResult(
            progress=progress,
            completion=completion,
            composite=composite,
            uncertainty_weighted=uncertainty_weighted,
            checkpoints=list(self._checkpoints),
            latency_ms=latency_ms,
            token_cost=token_cost,
            robustness_score=robustness_score,
            safety_score=safety_score,
        )
    
    # =========================================================================
    # Preset Checkpoint Templates
    # =========================================================================
    
    def setup_api_orchestration_checkpoints(
        self,
        n_apis: int = 3,
        include_auth: bool = True,
        include_recovery: bool = True,
    ) -> None:
        """
        Set up standard API orchestration checkpoints.
        
        Args:
            n_apis: Number of APIs in the workflow
            include_auth: Whether to include authentication checkpoint
            include_recovery: Whether to include error recovery checkpoint
        """
        if include_auth:
            self.add_checkpoint(
                "auth_success",
                CheckpointType.AUTHENTICATION,
                "Successfully authenticate with all required APIs",
                weight=1.5,  # Higher weight for critical step
            )
        
        for i in range(n_apis):
            self.add_checkpoint(
                f"api_{i}_call",
                CheckpointType.API_COORDINATION,
                f"Successfully call API {i + 1}",
                weight=1.0,
            )
        
        self.add_checkpoint(
            "data_validation",
            CheckpointType.DATA_CONSISTENCY,
            "Validate data consistency across APIs",
            weight=1.2,
        )
        
        if include_recovery:
            self.add_checkpoint(
                "error_recovery",
                CheckpointType.ERROR_RECOVERY,
                "Successfully recover from any errors",
                weight=1.3,
            )
        
        self.add_checkpoint(
            "uncertainty_check",
            CheckpointType.UNCERTAINTY_ASSESSMENT,
            "Maintain acceptable uncertainty levels",
            weight=0.8,
        )
        
        self.add_checkpoint(
            "final_artifact",
            CheckpointType.FINAL_ARTIFACT,
            "Produce final workflow output",
            weight=1.5,
        )
    
    def setup_toolemu_checkpoints(self) -> None:
        """Set up checkpoints for ToolEmu-style evaluation."""
        self.add_checkpoint(
            "tool_selection",
            CheckpointType.API_COORDINATION,
            "Select appropriate tool for task",
            weight=1.0,
        )
        
        self.add_checkpoint(
            "parameter_validation",
            CheckpointType.DATA_CONSISTENCY,
            "Provide valid parameters to tool",
            weight=1.0,
        )
        
        self.add_checkpoint(
            "safety_compliance",
            CheckpointType.ERROR_RECOVERY,
            "Avoid risky actions",
            weight=2.0,  # High weight for safety
        )
        
        self.add_checkpoint(
            "task_completion",
            CheckpointType.FINAL_ARTIFACT,
            "Complete the requested task",
            weight=1.5,
        )
    
    def setup_dbbench_checkpoints(self) -> None:
        """Set up checkpoints for DB-Bench evaluation."""
        self.add_checkpoint(
            "query_formulation",
            CheckpointType.API_COORDINATION,
            "Formulate correct SQL query",
            weight=1.2,
        )
        
        self.add_checkpoint(
            "syntax_valid",
            CheckpointType.DATA_CONSISTENCY,
            "Query has valid SQL syntax",
            weight=1.0,
        )
        
        self.add_checkpoint(
            "semantic_correct",
            CheckpointType.DATA_CONSISTENCY,
            "Query produces correct results",
            weight=1.5,
        )
        
        self.add_checkpoint(
            "result_interpretation",
            CheckpointType.FINAL_ARTIFACT,
            "Correctly interpret and present results",
            weight=1.0,
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def reset(self) -> None:
        """Reset all scoring state."""
        self._checkpoints = []
        self._step_uncertainties = []
        self._total_steps = 0
        self._completed = False
        self._failure_reason = None
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get a summary of checkpoint status."""
        completed = [cp.checkpoint_id for cp in self._checkpoints if cp.completed]
        partial = [cp.checkpoint_id for cp in self._checkpoints if 0 < cp.partial_score < 1]
        pending = [cp.checkpoint_id for cp in self._checkpoints if cp.partial_score == 0]
        
        return {
            "total": len(self._checkpoints),
            "completed": completed,
            "partial": partial,
            "pending": pending,
            "completion_rate": len(completed) / len(self._checkpoints) if self._checkpoints else 0,
        }
    
    def export_scores(self) -> Dict[str, Any]:
        """Export all scores as a dictionary."""
        result = self.compute_full_evaluation()
        
        return {
            "progress_score": result.progress.score,
            "full_completion_score": result.completion.score,
            "composite_score": result.composite.score,
            "uncertainty_adjusted_score": result.uncertainty_weighted.adjusted_score,
            "interpretation": result.composite.interpretation,
            "checkpoints_completed": result.progress.checkpoints_completed,
            "checkpoints_total": result.progress.checkpoints_total,
            "total_steps": result.completion.total_steps,
            "workflow_completed": result.completion.completed,
            "failure_reason": result.completion.failure_reason,
            "mean_uncertainty": result.uncertainty_weighted.mean_uncertainty,
        }


__all__ = [
    "ScoringSystem",
    "Checkpoint",
    "CheckpointType",
    "ProgressScore",
    "FullCompletionScore",
    "CompositeScore",
    "UncertaintyWeightedScore",
    "FullEvaluationResult",
]

