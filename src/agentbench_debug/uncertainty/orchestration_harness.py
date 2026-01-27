"""Unified orchestration harness for uncertainty estimation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .calibration import CalibrationMetrics, CalibrationResult
from .confidence_extractor import ConfidenceExtractor
from .error_taxonomy import ErrorTaxonomy, ErrorRecord
from .hierarchical import HierarchicalUncertainty
from .scoring import Scorer, EvaluationResult, StepResult


@dataclass
class WorkflowStep:
    """Record of a single workflow step."""
    step_idx: int
    action: str
    action_type: str
    observation: str
    confidence: float
    confidence_source: str
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "action": self.action,
            "action_type": self.action_type,
            "observation": self.observation[:500] if self.observation else "",
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkflowResult:
    """Complete result of a workflow run."""
    task_id: str
    task_type: str
    success: bool
    steps: List[WorkflowStep]
    evaluation: EvaluationResult
    trajectory_metrics: Dict[str, Any]
    calibration: Optional[CalibrationResult] = None
    errors: List[ErrorRecord] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "n_steps": len(self.steps),
            "evaluation": {
                "progress_score": self.evaluation.progress.score,
                "completion_score": self.evaluation.completion.score,
                "composite_score": self.evaluation.composite.score,
                "mean_confidence": self.evaluation.mean_confidence,
                "min_confidence": self.evaluation.min_confidence,
            },
            "trajectory": self.trajectory_metrics,
            "steps": [s.to_dict() for s in self.steps],
            "errors": [e.to_dict() for e in self.errors],
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class OrchestrationHarness:
    """
    Unified harness for running workflows with uncertainty tracking.
    
    Integrates:
    - Confidence extraction from API responses
    - Hierarchical uncertainty propagation
    - Error taxonomy
    - Calibration metrics
    - Scoring
    """
    
    def __init__(
        self,
        task_id: str = "",
        task_type: str = "unknown",
        uncertainty_threshold: float = 0.5,
        default_confidence: float = 0.70,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.uncertainty_threshold = uncertainty_threshold
        
        # Components
        self.confidence_extractor = ConfidenceExtractor(default_confidence)
        self.hierarchical = HierarchicalUncertainty(
            uncertainty_threshold=uncertainty_threshold
        )
        self.error_taxonomy = ErrorTaxonomy()
        self.scorer = Scorer()
        self.calibration = CalibrationMetrics()
        
        # State
        self.steps: List[WorkflowStep] = []
        self.start_time = datetime.now()
        self._step_counter = 0
    
    def start_workflow(self, task_id: str, task_type: str = "unknown") -> None:
        """Start a new workflow run."""
        self.task_id = task_id
        self.task_type = task_type
        self.steps = []
        self.hierarchical.reset()
        self.error_taxonomy.reset()
        self.start_time = datetime.now()
        self._step_counter = 0
    
    def record_step(
        self,
        action: str,
        action_type: str,
        observation: str,
        success: bool = True,
        error: Optional[str] = None,
        raw_api_response: Optional[Dict[str, Any]] = None,
        confidence_override: Optional[float] = None,
    ) -> WorkflowStep:
        """
        Record a workflow step.
        
        Args:
            action: The action taken
            action_type: Type of action (e.g., "query", "shell_command")
            observation: Environment observation/feedback
            success: Whether the step succeeded
            error: Error message if any
            raw_api_response: Raw API response for confidence extraction
            confidence_override: Override extracted confidence
        
        Returns:
            The recorded WorkflowStep
        """
        step_idx = self._step_counter
        self._step_counter += 1
        
        # Extract confidence
        if confidence_override is not None:
            confidence = confidence_override
            confidence_source = "override"
        else:
            confidence, confidence_source = self.confidence_extractor.extract(
                raw_api_response, action
            )
        
        # Record in hierarchical tracker
        self.hierarchical.add_step(
            step_idx=step_idx,
            action=action,
            action_type=action_type,
            observation=observation,
            confidence=confidence,
            is_error=not success,
        )
        
        # Record error if any
        if error:
            self.error_taxonomy.record_error(error, step_idx)
        
        # Create step record
        step = WorkflowStep(
            step_idx=step_idx,
            action=action,
            action_type=action_type,
            observation=observation,
            confidence=confidence,
            confidence_source=confidence_source,
            success=success,
            error=error,
            raw_response=raw_api_response,
        )
        
        self.steps.append(step)
        return step
    
    def finish_workflow(
        self,
        task_completed: bool,
    ) -> WorkflowResult:
        """
        Finish the workflow and compute all metrics.
        
        Args:
            task_completed: Whether the overall task was completed successfully
        
        Returns:
            Complete WorkflowResult with all metrics
        """
        end_time = datetime.now()
        
        # Convert to StepResults for scoring
        step_results = [
            StepResult(
                step_idx=s.step_idx,
                action=s.action,
                success=s.success,
                confidence=s.confidence,
                error=s.error,
            )
            for s in self.steps
        ]
        
        # Compute evaluation
        evaluation = self.scorer.evaluate(
            steps=step_results,
            task_completed=task_completed,
        )
        
        # Get trajectory metrics
        trajectory_metrics = self.hierarchical.compute_trajectory_uncertainty()
        
        # Compute calibration if enough data
        calibration_result = None
        if len(self.steps) >= 10:
            confidences = [s.confidence for s in self.steps]
            outcomes = [s.success for s in self.steps]
            calibration_result = self.calibration.compute(confidences, outcomes)
        
        return WorkflowResult(
            task_id=self.task_id,
            task_type=self.task_type,
            success=task_completed,
            steps=self.steps,
            evaluation=evaluation,
            trajectory_metrics=trajectory_metrics,
            calibration=calibration_result,
            errors=self.error_taxonomy.errors,
            start_time=self.start_time,
            end_time=end_time,
        )
    
    def get_current_uncertainty(self) -> Dict[str, Any]:
        """Get current uncertainty metrics without finishing."""
        return self.hierarchical.compute_trajectory_uncertainty()
    
    def get_confidence_history(self) -> List[float]:
        """Get list of confidence values for all steps."""
        return [s.confidence for s in self.steps]
    
    def save_result(
        self,
        result: WorkflowResult,
        output_dir: str,
        filename: str = "uncertainty_result.json",
    ) -> Path:
        """Save workflow result to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return filepath


__all__ = [
    "OrchestrationHarness",
    "WorkflowStep",
    "WorkflowResult",
]

