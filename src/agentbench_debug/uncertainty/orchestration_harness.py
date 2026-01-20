"""
API Orchestration Evaluation Harness.

Integrates all uncertainty estimation components:
- Hierarchical uncertainty propagation
- Calibration metrics
- Error taxonomy
- Scoring system
- Mitigation strategies

Provides a unified interface for running uncertainty-aware evaluations
on AgentBench tasks (ToolEmu, DBBench, OS, etc.).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hierarchical import (
    HierarchicalUncertaintyPropagator,
    HierarchicalUncertaintyResult,
    ActionUncertainty,
    ObservationUncertainty,
)
from .calibration import CalibrationMetrics, CalibrationResult
from .error_taxonomy import ErrorTaxonomy, ErrorCategory, ErrorInstance, ErrorSummary
from .scoring import (
    ScoringSystem,
    CheckpointType,
    FullEvaluationResult,
)
from .mitigation import (
    DecisionLayer,
    MitigationStrategy,
)
from .confidence_extractor import ConfidenceExtractor, extract_confidence


@dataclass
class StepRecord:
    """Record of a single step in the workflow."""
    step_idx: int
    action_name: str
    action_type: str
    
    # Inputs
    input_messages: List[Dict[str, str]]
    tools_available: List[str]
    
    # Outputs
    response: Any
    tool_calls: List[Dict[str, Any]]
    
    # Uncertainty
    action_uncertainty: Optional[ActionUncertainty]
    observation_uncertainty: Optional[ObservationUncertainty]
    
    # Errors
    error_occurred: bool
    error_instance: Optional[ErrorInstance]
    
    # Mitigation
    mitigation_applied: Optional[str]
    mitigation_success: bool
    
    # Timing
    latency_ms: float
    timestamp: datetime


@dataclass
class WorkflowRun:
    """Complete record of a workflow run."""
    run_id: str
    task_name: str
    task_type: str  # toolemu, dbbench, os, etc.
    start_time: datetime
    end_time: Optional[datetime]
    
    # Steps
    steps: List[StepRecord]
    
    # Results
    success: bool
    failure_reason: Optional[str]
    
    # Uncertainty analysis
    hierarchical_uncertainty: Optional[HierarchicalUncertaintyResult]
    calibration: Optional[CalibrationResult]
    
    # Error analysis
    error_summary: Optional[ErrorSummary]
    
    # Scoring
    evaluation: Optional[FullEvaluationResult]
    
    # Resource usage
    total_latency_ms: float
    total_tokens: int
    
    # Raw trajectory for debugging
    raw_trajectory: List[Dict[str, Any]]


@dataclass
class EvaluationConfig:
    """Configuration for evaluation harness."""
    # Uncertainty settings
    uncertainty_threshold: float = 0.35
    enable_hierarchical: bool = True
    enable_calibration: bool = True
    
    # Scoring settings
    progress_weight: float = 0.4
    completion_weight: float = 0.3
    interaction_weight: float = 0.3
    
    # Mitigation settings
    enable_mitigation: bool = True
    max_retries: int = 3
    
    # Output settings
    save_trajectories: bool = True
    output_dir: str = "outputs/uncertainty_eval"


class OrchestrationHarness:
    """
    Unified harness for uncertainty-aware API orchestration evaluation.
    
    Integrates all components and provides a streamlined interface
    for running evaluations on AgentBench tasks.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize orchestration harness.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        
        # Core components
        self.uncertainty = HierarchicalUncertaintyPropagator(
            uncertainty_threshold=self.config.uncertainty_threshold
        )
        self.calibration = CalibrationMetrics()
        self.errors = ErrorTaxonomy()
        self.scoring = ScoringSystem(
            alpha=self.config.progress_weight,
            beta=self.config.completion_weight,
            gamma=self.config.interaction_weight,
        )
        self.mitigation = DecisionLayer()
        
        # Confidence extractor for real API responses
        self.confidence_extractor = ConfidenceExtractor()
        
        # Run state
        self._current_run: Optional[WorkflowRun] = None
        self._step_records: List[StepRecord] = []
        self._run_counter = 0
        
        # Collected runs for aggregation
        self._completed_runs: List[WorkflowRun] = []
    
    # =========================================================================
    # Run Lifecycle
    # =========================================================================
    
    def start_run(
        self,
        task_name: str,
        task_type: str,
        checkpoints: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Start a new workflow run.
        
        Args:
            task_name: Name of the task
            task_type: Type (toolemu, dbbench, os, etc.)
            checkpoints: Optional list of checkpoint definitions
        
        Returns:
            Run ID
        """
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}_{int(time.time())}"
        
        # Reset components
        self.uncertainty.reset()
        self.errors.reset()
        self.scoring.reset()
        
        # Set up checkpoints based on task type
        if checkpoints:
            for cp in checkpoints:
                self.scoring.add_checkpoint(
                    checkpoint_id=cp.get("id", f"cp_{len(self.scoring._checkpoints)}"),
                    checkpoint_type=CheckpointType[cp.get("type", "FINAL_ARTIFACT").upper()],
                    description=cp.get("description", ""),
                    weight=cp.get("weight", 1.0),
                )
        else:
            # Default checkpoints based on task type
            if task_type == "toolemu":
                self.scoring.setup_toolemu_checkpoints()
            elif task_type == "dbbench":
                self.scoring.setup_dbbench_checkpoints()
            else:
                self.scoring.setup_api_orchestration_checkpoints()
        
        self._current_run = WorkflowRun(
            run_id=run_id,
            task_name=task_name,
            task_type=task_type,
            start_time=datetime.now(),
            end_time=None,
            steps=[],
            success=False,
            failure_reason=None,
            hierarchical_uncertainty=None,
            calibration=None,
            error_summary=None,
            evaluation=None,
            total_latency_ms=0.0,
            total_tokens=0,
            raw_trajectory=[],
        )
        
        self._step_records = []
        
        return run_id
    
    def record_step(
        self,
        action_name: str,
        action_type: str,
        input_messages: List[Dict[str, str]],
        tools_available: List[str],
        response: Any,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        raw_api_response: Optional[Dict[str, Any]] = None,
        api_type: str = "auto",
        logprobs: Optional[List[float]] = None,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> StepRecord:
        """
        Record a single step in the workflow.
        
        Args:
            action_name: Name of the action/tool
            action_type: Type of action (auth, query, etc.)
            input_messages: Input messages to agent
            tools_available: Tools available at this step
            response: Agent response (content string or structured response)
            tool_calls: Tool calls made
            confidence: Explicit confidence score (0-1). If None, extracted from raw_api_response.
            raw_api_response: Raw LLM API response for automatic confidence extraction
            api_type: API type for extraction ("openai", "gemini", "anthropic", "auto")
            logprobs: Optional log probabilities (used if raw_api_response not provided)
            latency_ms: Step latency
            tokens_used: Tokens consumed
        
        Returns:
            StepRecord
        """
        if not self._current_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        step_idx = len(self._step_records)
        
        # Extract confidence from raw API response if not explicitly provided
        if confidence is None:
            if raw_api_response is not None:
                # Use the confidence extractor for real API responses
                signals = self.confidence_extractor.extract(
                    raw_api_response,
                    api_type=api_type,
                    content=response if isinstance(response, str) else None,
                )
                confidence = signals.confidence
            elif logprobs is not None:
                # Compute from logprobs
                import math
                mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
                confidence = min(1.0, math.exp(mean_logprob))
            else:
                # Default confidence
                confidence = 0.7
        
        # Compute action uncertainty
        action_unc = self.uncertainty.compute_action_uncertainty(
            action_name=action_name,
            action_type=action_type,
            confidence=confidence,
        )
        
        # Compute observation uncertainty (from response)
        obs_unc = self.uncertainty.compute_observation_uncertainty(
            response=response,
        )
        
        # Check for errors
        error_occurred = False
        error_instance = None
        
        if isinstance(response, dict):
            if response.get("error") or response.get("status") == "error":
                error_occurred = True
                error_instance = self.errors.log_error(
                    step_idx=step_idx,
                    action_name=action_name,
                    message=str(response.get("error", response.get("message", "Unknown error"))),
                    http_status=response.get("status_code"),
                    raw_response=json.dumps(response)[:500],
                    uncertainty=action_unc.weighted_uncertainty,
                )
        
        self.errors.mark_step()
        
        # Update scoring
        self.scoring.add_step(action_unc.weighted_uncertainty)
        
        # Apply mitigation if needed
        mitigation_applied = None
        mitigation_success = False
        
        if error_occurred and self.config.enable_mitigation:
            strategies = self.mitigation.select_strategy(
                uncertainty=action_unc.weighted_uncertainty,
                error_type=error_instance.error_type if error_instance else None,
                error_category=error_instance.category.value if error_instance else None,
            )
            
            if strategies:
                mitigation_applied = strategies[0].value
                # Mitigation would be executed by calling layer
                # For now, just record the recommendation
        
        # Create step record
        record = StepRecord(
            step_idx=step_idx,
            action_name=action_name,
            action_type=action_type,
            input_messages=input_messages,
            tools_available=tools_available,
            response=response,
            tool_calls=tool_calls or [],
            action_uncertainty=action_unc,
            observation_uncertainty=obs_unc,
            error_occurred=error_occurred,
            error_instance=error_instance,
            mitigation_applied=mitigation_applied,
            mitigation_success=mitigation_success,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
        )
        
        self._step_records.append(record)
        self._current_run.steps = self._step_records
        self._current_run.total_latency_ms += latency_ms
        self._current_run.total_tokens += tokens_used
        
        # Store raw trajectory
        self._current_run.raw_trajectory.append({
            "step": step_idx,
            "action": action_name,
            "response_preview": str(response)[:200] if response else None,
            "uncertainty": action_unc.weighted_uncertainty,
            "error": error_occurred,
        })
        
        self.uncertainty.next_step()
        
        return record
    
    def mark_checkpoint(
        self,
        checkpoint_id: str,
        score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a checkpoint as completed."""
        if not self._current_run:
            return
        
        current_uncertainty = 0.0
        if self._step_records:
            last_step = self._step_records[-1]
            if last_step.action_uncertainty:
                current_uncertainty = last_step.action_uncertainty.weighted_uncertainty
        
        self.scoring.mark_checkpoint_completed(
            checkpoint_id=checkpoint_id,
            partial_score=score,
            uncertainty=current_uncertainty,
            metadata=metadata,
        )
    
    def end_run(
        self,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> WorkflowRun:
        """
        End the current run and compute final metrics.
        
        Args:
            success: Whether the workflow succeeded
            failure_reason: Reason for failure if applicable
        
        Returns:
            Complete WorkflowRun record
        """
        if not self._current_run:
            raise RuntimeError("No active run.")
        
        self._current_run.end_time = datetime.now()
        self._current_run.success = success
        self._current_run.failure_reason = failure_reason
        
        # Mark workflow completion in scoring
        self.scoring.mark_workflow_complete(success, failure_reason)
        
        # Compute hierarchical uncertainty
        if self.config.enable_hierarchical:
            self._current_run.hierarchical_uncertainty = self.uncertainty.analyze_complete(
                trajectory_id=self._current_run.run_id
            )
        
        # Compute calibration
        if self.config.enable_calibration and self._step_records:
            # Add predictions for calibration
            for record in self._step_records:
                if record.action_uncertainty:
                    confidence = record.action_uncertainty.confidence
                    correct = not record.error_occurred
                    self.calibration.add_prediction(confidence, correct)
            
            self._current_run.calibration = self.calibration.compute_all()
        
        # Compute error summary
        self._current_run.error_summary = self.errors.compute_summary()
        
        # Compute final evaluation
        self._current_run.evaluation = self.scoring.compute_full_evaluation(
            latency_ms=self._current_run.total_latency_ms,
            token_cost=self._current_run.total_tokens,
        )
        
        # Store completed run
        completed = self._current_run
        self._completed_runs.append(completed)
        self._current_run = None
        
        return completed
    
    # =========================================================================
    # Analysis and Aggregation
    # =========================================================================
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics across all completed runs.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self._completed_runs:
            return {"error": "No completed runs"}
        
        n_runs = len(self._completed_runs)
        successful = sum(1 for r in self._completed_runs if r.success)
        
        # Aggregate scores
        progress_scores = []
        completion_scores = []
        composite_scores = []
        uncertainty_scores = []
        
        for run in self._completed_runs:
            if run.evaluation:
                progress_scores.append(run.evaluation.progress.score)
                completion_scores.append(run.evaluation.completion.score)
                composite_scores.append(run.evaluation.composite.score)
                uncertainty_scores.append(run.evaluation.uncertainty_weighted.adjusted_score)
        
        # Aggregate errors
        total_errors = sum(r.error_summary.total_errors if r.error_summary else 0 
                          for r in self._completed_runs)
        
        # Aggregate calibration
        ece_values = []
        brier_values = []
        
        for run in self._completed_runs:
            if run.calibration:
                ece_values.append(run.calibration.ece)
                brier_values.append(run.calibration.brier_score)
        
        return {
            "total_runs": n_runs,
            "success_rate": successful / n_runs if n_runs > 0 else 0.0,
            "scores": {
                "mean_progress": sum(progress_scores) / len(progress_scores) if progress_scores else 0,
                "mean_completion": sum(completion_scores) / len(completion_scores) if completion_scores else 0,
                "mean_composite": sum(composite_scores) / len(composite_scores) if composite_scores else 0,
                "mean_uncertainty_adjusted": sum(uncertainty_scores) / len(uncertainty_scores) if uncertainty_scores else 0,
            },
            "calibration": {
                "mean_ece": sum(ece_values) / len(ece_values) if ece_values else 0,
                "mean_brier": sum(brier_values) / len(brier_values) if brier_values else 0,
            },
            "errors": {
                "total": total_errors,
                "mean_per_run": total_errors / n_runs if n_runs > 0 else 0,
            },
            "latency": {
                "mean_ms": sum(r.total_latency_ms for r in self._completed_runs) / n_runs,
            },
            "tokens": {
                "total": sum(r.total_tokens for r in self._completed_runs),
                "mean_per_run": sum(r.total_tokens for r in self._completed_runs) / n_runs,
            },
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get detailed error analysis across all runs."""
        if not self._completed_runs:
            return {"error": "No completed runs"}
        
        # Aggregate errors by category
        by_category: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        
        for run in self._completed_runs:
            if run.error_summary:
                for cat, count in run.error_summary.errors_by_category.items():
                    by_category[cat] = by_category.get(cat, 0) + count
                for typ, count in run.error_summary.errors_by_type.items():
                    by_type[typ] = by_type.get(typ, 0) + count
        
        # Find most common errors
        top_errors = sorted(by_type.items(), key=lambda x: -x[1])[:10]
        
        return {
            "errors_by_category": by_category,
            "errors_by_type": by_type,
            "top_10_errors": top_errors,
            "category_distribution": {
                k: v / sum(by_category.values()) if by_category else 0
                for k, v in by_category.items()
            },
        }
    
    def get_uncertainty_analysis(self) -> Dict[str, Any]:
        """Get uncertainty analysis across all runs."""
        if not self._completed_runs:
            return {"error": "No completed runs"}
        
        # Collect trajectory-level data
        trajectory_uncertainties = []
        critical_steps_counts = []
        trends = {"increasing": 0, "decreasing": 0, "stable": 0}
        
        for run in self._completed_runs:
            if run.hierarchical_uncertainty:
                hu = run.hierarchical_uncertainty
                trajectory_uncertainties.append(hu.trajectory_level.cumulative_uncertainty)
                critical_steps_counts.append(len(hu.trajectory_level.critical_steps))
                trends[hu.trajectory_level.uncertainty_trend] += 1
        
        return {
            "mean_trajectory_uncertainty": (
                sum(trajectory_uncertainties) / len(trajectory_uncertainties)
                if trajectory_uncertainties else 0
            ),
            "mean_critical_steps": (
                sum(critical_steps_counts) / len(critical_steps_counts)
                if critical_steps_counts else 0
            ),
            "trend_distribution": trends,
            "aggregated_confidence": (
                1.0 - sum(trajectory_uncertainties) / len(trajectory_uncertainties)
                if trajectory_uncertainties else 1.0
            ),
        }
    
    # =========================================================================
    # Export and Persistence
    # =========================================================================
    
    def export_run(self, run: WorkflowRun) -> Dict[str, Any]:
        """Export a run to a serializable dictionary."""
        return {
            "run_id": run.run_id,
            "task_name": run.task_name,
            "task_type": run.task_type,
            "start_time": run.start_time.isoformat(),
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "success": run.success,
            "failure_reason": run.failure_reason,
            "steps": len(run.steps),
            "total_latency_ms": run.total_latency_ms,
            "total_tokens": run.total_tokens,
            "scores": {
                "progress": run.evaluation.progress.score if run.evaluation else None,
                "completion": run.evaluation.completion.score if run.evaluation else None,
                "composite": run.evaluation.composite.score if run.evaluation else None,
            },
            "calibration": {
                "ece": run.calibration.ece if run.calibration else None,
                "brier": run.calibration.brier_score if run.calibration else None,
            },
            "errors": {
                "total": run.error_summary.total_errors if run.error_summary else 0,
                "by_category": run.error_summary.errors_by_category if run.error_summary else {},
            },
            "uncertainty": {
                "cumulative": (
                    run.hierarchical_uncertainty.trajectory_level.cumulative_uncertainty
                    if run.hierarchical_uncertainty else None
                ),
                "final_confidence": (
                    run.hierarchical_uncertainty.trajectory_level.final_confidence
                    if run.hierarchical_uncertainty else None
                ),
            },
            "trajectory": run.raw_trajectory,
        }
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save all results to disk.
        
        Args:
            output_path: Output directory (uses config default if not specified)
        
        Returns:
            Path where results were saved
        """
        output_dir = Path(output_path or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        metrics_path = output_dir / f"aggregate_metrics_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "metrics": self.get_aggregate_metrics(),
                "error_analysis": self.get_error_analysis(),
                "uncertainty_analysis": self.get_uncertainty_analysis(),
            }, f, indent=2)
        
        # Save individual runs
        runs_path = output_dir / f"runs_{timestamp}.jsonl"
        with open(runs_path, "w") as f:
            for run in self._completed_runs:
                f.write(json.dumps(self.export_run(run)) + "\n")
        
        # Save mitigation stats
        if self.config.enable_mitigation:
            mitigation_path = output_dir / f"mitigation_stats_{timestamp}.json"
            with open(mitigation_path, "w") as f:
                json.dump({
                    "strategy_stats": self.mitigation.get_strategy_stats(),
                    "decision_history": self.mitigation.get_decision_history()[-100:],
                }, f, indent=2)
        
        return str(output_dir)
    
    def reset(self) -> None:
        """Reset all state."""
        self._current_run = None
        self._step_records = []
        self._completed_runs = []
        self._run_counter = 0
        
        self.uncertainty.reset()
        self.calibration.reset()
        self.errors.reset()
        self.scoring.reset()
        self.mitigation.reset()


# Convenience function for quick evaluation
def evaluate_trajectory(
    task_name: str,
    task_type: str,
    steps: List[Dict[str, Any]],
    success: bool,
    config: Optional[EvaluationConfig] = None,
) -> WorkflowRun:
    """
    Quick evaluation of a pre-recorded trajectory.
    
    Args:
        task_name: Task name
        task_type: Task type
        steps: List of step dictionaries with action, response, etc.
        success: Whether workflow succeeded
        config: Optional configuration
    
    Returns:
        Completed WorkflowRun
    """
    harness = OrchestrationHarness(config)
    harness.start_run(task_name, task_type)
    
    for step in steps:
        harness.record_step(
            action_name=step.get("action", "unknown"),
            action_type=step.get("action_type", "default"),
            input_messages=step.get("messages", []),
            tools_available=step.get("tools", []),
            response=step.get("response"),
            tool_calls=step.get("tool_calls"),
            confidence=step.get("confidence", 1.0),
            latency_ms=step.get("latency_ms", 0),
            tokens_used=step.get("tokens", 0),
        )
        
        # Mark checkpoints if specified
        if step.get("checkpoint"):
            harness.mark_checkpoint(step["checkpoint"], step.get("checkpoint_score", 1.0))
    
    return harness.end_run(success)


__all__ = [
    "OrchestrationHarness",
    "EvaluationConfig",
    "WorkflowRun",
    "StepRecord",
    "evaluate_trajectory",
]

