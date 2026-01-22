"""
AgentBench FC Pipeline Integration for Uncertainty Estimation.

This module provides hooks to integrate uncertainty estimation into the
AgentBench FC (Function Calling) evaluation pipeline.

Supported Tasks:
    - alfworld (AF): Household tasks in simulated environment
    - dbbench (DB): Database SQL query tasks
    - os (OS): OS interaction tasks
    - kg (KG): Knowledge graph question answering
    - webshop (WS): Web shopping navigation

Usage:
    # Option 1: Wrap agent before running
    from agentbench_debug.uncertainty import create_uncertainty_agent
    
    wrapped_agent = create_uncertainty_agent(agent, task_type="alfworld")
    result = task.run(wrapped_agent)
    uncertainty_report = wrapped_agent.get_uncertainty_report()
    
    # Option 2: Use callback-based tracking
    from agentbench_debug.uncertainty import UncertaintyCallback
    
    callback = UncertaintyCallback()
    # Pass to task runner
    result = run_with_callback(agent, task, callback)
    report = callback.get_report()
    
    # Option 3: Post-hoc analysis of saved runs
    from agentbench_debug.uncertainty import analyze_saved_runs
    
    report = analyze_saved_runs("outputs/runs.jsonl")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .agent_wrapper import UncertaintyAwareAgent, UncertaintyTracker
from .confidence_extractor import ConfidenceExtractor, extract_confidence
from .orchestration_harness import OrchestrationHarness, EvaluationConfig
from .hierarchical import HierarchicalUncertaintyPropagator


@dataclass
class UncertaintyReport:
    """Complete uncertainty analysis report."""
    task_name: str
    task_type: str
    timestamp: datetime
    
    # Overall metrics
    success: bool
    total_steps: int
    mean_confidence: float
    min_confidence: float
    
    # Trajectory analysis
    trajectory_uncertainty: float
    final_confidence: float
    uncertainty_trend: str
    high_uncertainty_steps: List[int]
    
    # Calibration (if available)
    ece: Optional[float]
    brier_score: Optional[float]
    
    # Scores
    composite_score: Optional[float]
    progress_score: Optional[float]
    
    # Per-step details
    step_confidences: List[float]
    step_actions: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "total_steps": self.total_steps,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "trajectory_uncertainty": self.trajectory_uncertainty,
            "final_confidence": self.final_confidence,
            "uncertainty_trend": self.uncertainty_trend,
            "high_uncertainty_steps": self.high_uncertainty_steps,
            "ece": self.ece,
            "brier_score": self.brier_score,
            "composite_score": self.composite_score,
            "progress_score": self.progress_score,
            "step_confidences": self.step_confidences,
            "step_actions": self.step_actions,
            "recommendations": self.recommendations,
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary."""
        print(f"\n{'='*60}")
        print(f"Uncertainty Report: {self.task_name}")
        print(f"{'='*60}")
        print(f"Task Type: {self.task_type}")
        print(f"Success: {'Yes' if self.success else 'No'}")
        print(f"Steps: {self.total_steps}")
        print()
        print("CONFIDENCE METRICS:")
        print(f"  Mean Confidence: {self.mean_confidence:.3f}")
        print(f"  Min Confidence: {self.min_confidence:.3f}")
        print(f"  Final Confidence: {self.final_confidence:.3f}")
        print(f"  Uncertainty Trend: {self.uncertainty_trend}")
        print()
        if self.high_uncertainty_steps:
            print(f"  [!] High Uncertainty Steps: {self.high_uncertainty_steps}")
        if self.ece is not None:
            print(f"\nCALIBRATION:")
            ece_quality = "Good" if self.ece < 0.1 else "Fair" if self.ece < 0.2 else "Poor"
            print(f"  ECE: {self.ece:.4f} ({ece_quality})")
        if self.composite_score is not None:
            print(f"\nSCORES:")
            print(f"  Composite: {self.composite_score:.3f}")
            print(f"  Progress: {self.progress_score:.3f}")
        if self.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"  • {rec}")
        print(f"{'='*60}\n")


def create_uncertainty_agent(
    agent: Any,
    task_type: str = "auto",
    api_type: str = "auto",
    uncertainty_threshold: float = 0.35,
) -> UncertaintyAwareAgent:
    """
    Create an uncertainty-aware wrapped agent.
    
    Args:
        agent: Original AgentBench agent
        task_type: Task type for checkpoint setup ("toolemu", "dbbench", etc.)
        api_type: LLM API type ("openai", "gemini", "auto")
        uncertainty_threshold: Threshold for flagging high uncertainty
    
    Returns:
        Wrapped agent with uncertainty tracking
    
    Example:
        agent = load_agent("my-gemini")
        wrapped = create_uncertainty_agent(agent, task_type="toolemu")
        
        # Run normally
        result = task.run(wrapped)
        
        # Get uncertainty analysis
        report = wrapped.get_uncertainty_report()
    """
    wrapped = UncertaintyAwareAgent(
        agent=agent,
        api_type=api_type,
        request_logprobs=True,
        uncertainty_threshold=uncertainty_threshold,
    )
    
    # Store task type for later report generation
    wrapped._task_type = task_type
    
    # Add report generation method
    def get_uncertainty_report(task_name: str = "unknown", success: bool = True) -> UncertaintyReport:
        records = wrapped.get_inference_records()
        analysis = wrapped.get_uncertainty_analysis() or {}
        
        confidences = [r.confidence for r in records]
        actions = [r.tool_calls[0]["name"] if r.tool_calls else "respond" for r in records]
        
        # Generate recommendations
        recommendations = _generate_recommendations(
            mean_confidence=sum(confidences) / len(confidences) if confidences else 0.5,
            min_confidence=min(confidences) if confidences else 0.5,
            trend=analysis.get("trend", "stable"),
            high_uncertainty_steps=analysis.get("critical_steps", []),
        )
        
        return UncertaintyReport(
            task_name=task_name,
            task_type=wrapped._task_type,
            timestamp=datetime.now(),
            success=success,
            total_steps=len(records),
            mean_confidence=sum(confidences) / len(confidences) if confidences else 0.5,
            min_confidence=min(confidences) if confidences else 0.5,
            trajectory_uncertainty=analysis.get("trajectory_uncertainty", 0.0),
            final_confidence=analysis.get("final_confidence", 0.5),
            uncertainty_trend=analysis.get("trend", "stable"),
            high_uncertainty_steps=analysis.get("critical_steps", []),
            ece=None,
            brier_score=None,
            composite_score=None,
            progress_score=None,
            step_confidences=confidences,
            step_actions=actions,
            recommendations=recommendations,
        )
    
    wrapped.get_uncertainty_report = get_uncertainty_report
    
    return wrapped


class UncertaintyCallback:
    """
    Callback-based uncertainty tracking for pipeline integration.
    
    Can be used as a callback/hook in custom evaluation loops.
    """
    
    def __init__(
        self,
        api_type: str = "auto",
        uncertainty_threshold: float = 0.35,
    ):
        """Initialize callback."""
        self._tracker = UncertaintyTracker(
            api_type=api_type,
            uncertainty_threshold=uncertainty_threshold,
        )
        self._api_type = api_type
        self._threshold = uncertainty_threshold
        
        self._task_name: Optional[str] = None
        self._task_type: str = "unknown"
        self._actions: List[str] = []
        self._start_time: Optional[datetime] = None
    
    def on_task_start(self, task_name: str, task_type: str = "unknown") -> None:
        """Called when a task starts."""
        self._task_name = task_name
        self._task_type = task_type
        self._actions = []
        self._start_time = datetime.now()
        self._tracker.reset()
    
    def on_step(
        self,
        content: str,
        action_name: str = "action",
        action_type: str = "default",
        raw_response: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Called after each agent step.
        
        Args:
            content: Agent response content
            action_name: Name of action taken
            action_type: Type of action
            raw_response: Optional raw API response
        
        Returns:
            Extracted confidence score
        """
        self._actions.append(action_name)
        
        confidence = self._tracker.record_response(
            content=content,
            raw_response=raw_response,
            action_name=action_name,
            action_type=action_type,
        )
        
        return confidence
    
    def on_task_end(self, success: bool) -> UncertaintyReport:
        """
        Called when task ends. Returns uncertainty report.
        
        Args:
            success: Whether task succeeded
        
        Returns:
            Complete uncertainty report
        """
        analysis = self._tracker.get_analysis()
        confidences = self._tracker.get_confidence_history()
        
        recommendations = _generate_recommendations(
            mean_confidence=analysis.get("mean_confidence", 0.5),
            min_confidence=analysis.get("min_confidence", 0.5),
            trend=analysis.get("trend", "stable"),
            high_uncertainty_steps=analysis.get("critical_steps", []),
        )
        
        return UncertaintyReport(
            task_name=self._task_name or "unknown",
            task_type=self._task_type,
            timestamp=datetime.now(),
            success=success,
            total_steps=analysis.get("n_steps", 0),
            mean_confidence=analysis.get("mean_confidence", 0.5),
            min_confidence=analysis.get("min_confidence", 0.5),
            trajectory_uncertainty=analysis.get("trajectory_uncertainty", 0.0),
            final_confidence=analysis.get("final_confidence", 0.5),
            uncertainty_trend=analysis.get("trend", "stable"),
            high_uncertainty_steps=analysis.get("critical_steps", []),
            ece=None,
            brier_score=None,
            composite_score=None,
            progress_score=None,
            step_confidences=confidences,
            step_actions=self._actions,
            recommendations=recommendations,
        )
    
    def is_high_uncertainty(self) -> bool:
        """Check if current state has high uncertainty."""
        return self._tracker.is_high_uncertainty()
    
    def get_current_confidence(self) -> float:
        """Get most recent confidence score."""
        history = self._tracker.get_confidence_history()
        return history[-1] if history else 0.5


def analyze_saved_runs(
    input_path: Union[str, Path],
    task_type: str = "auto",
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Any]:
    """
    Analyze saved runs for uncertainty estimation.
    
    Args:
        input_path: Path to runs.jsonl or directory
        task_type: Task type (auto-detected if not specified)
        config: Optional evaluation configuration
    
    Returns:
        Analysis results dictionary
    """
    from .orchestration_harness import OrchestrationHarness
    
    path = Path(input_path)
    config = config or EvaluationConfig()
    harness = OrchestrationHarness(config)
    
    # Load runs
    runs = []
    if path.is_file():
        with open(path) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
    elif path.is_dir():
        for jsonl in path.rglob("*.jsonl"):
            with open(jsonl) as f:
                for line in f:
                    if line.strip():
                        runs.append(json.loads(line))
    
    # Detect task type
    if task_type == "auto" and runs:
        task_type = _detect_task_type(runs[0], path)
    
    # Process each run
    for run in runs:
        _process_run(harness, run, task_type)
    
    # Get aggregate results
    return {
        "metrics": harness.get_aggregate_metrics(),
        "error_analysis": harness.get_error_analysis(),
        "uncertainty_analysis": harness.get_uncertainty_analysis(),
        "n_runs": len(runs),
        "task_type": task_type,
    }


def _process_run(
    harness: OrchestrationHarness,
    run: Dict[str, Any],
    task_type: str,
) -> None:
    """Process a single run through the harness."""
    task_name = run.get("task", run.get("id", "unknown"))
    success = run.get("success", run.get("output", {}).get("success", False))
    
    # Extract history/steps
    history = run.get("history", run.get("output", {}).get("history", []))
    
    if not history:
        return
    
    harness.start_run(task_name, task_type)
    
    for i, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        
        # Extract step info
        action = item.get("tool", item.get("action", item.get("command", f"step_{i}")))
        response = item.get("result", item.get("response", item.get("output")))
        
        # Estimate confidence from response
        confidence = _estimate_confidence_from_history(item, i, len(history))
        
        harness.record_step(
            action_name=action,
            action_type=_infer_action_type(action),
            input_messages=item.get("messages", []),
            tools_available=item.get("tools", []),
            response=response,
            confidence=confidence,
        )
    
    harness.end_run(success)


def _detect_task_type(run: Dict[str, Any], path: Path) -> str:
    """Auto-detect task type from path and content."""
    path_str = str(path).lower()
    
    # AgentBench FC Core Tasks (priority order)
    if "alfworld" in path_str:
        return "alfworld"
    if "dbbench" in path_str:
        return "dbbench"
    if "os_interaction" in path_str or "os-std" in path_str:
        return "os"
    if "knowledgegraph" in path_str or "kg-std" in path_str:
        return "kg"
    if "webshop" in path_str:
        return "webshop"
    
    # Check content for AgentBench FC task signatures
    run_str = str(run).lower()
    
    # ALFWorld - household tasks
    if "take_action" in run_str or "alfworld" in run_str:
        return "alfworld"
    
    # DBBench - SQL queries
    if "execute_sql" in run_str or "commit_final_answer" in run_str:
        return "dbbench"
    
    # OS Interaction - bash commands
    if "bash_action" in run_str or "finish_action" in run_str:
        return "os"
    
    # Knowledge Graph - SPARQL queries
    if "sparql" in run_str or "freebase" in run_str:
        return "kg"
    
    # WebShop - e-commerce navigation
    if "search_action" in run_str or "click_action" in run_str:
        return "webshop"
    
    return "unknown"


def _infer_action_type(action: str) -> str:
    """Infer action type from name for AgentBench FC tasks."""
    action_lower = action.lower()
    
    # AgentBench FC specific actions
    # ALFWorld
    if "take_action" in action_lower:
        return "environment_action"
    
    # DBBench
    if "execute_sql" in action_lower:
        return "query"
    if "commit_final_answer" in action_lower:
        return "submit"
    
    # OS Interaction
    if "bash_action" in action_lower:
        return "shell_command"
    if "finish_action" in action_lower:
        return "complete"
    if "answer_action" in action_lower:
        return "submit"
    
    # WebShop
    if "search_action" in action_lower:
        return "search"
    if "click_action" in action_lower:
        return "navigation"
    
    # Generic patterns
    if any(kw in action_lower for kw in ["auth", "login", "token"]):
        return "auth"
    if any(kw in action_lower for kw in ["delete", "remove"]):
        return "delete"
    if any(kw in action_lower for kw in ["create", "write", "insert", "update"]):
        return "write"
    if any(kw in action_lower for kw in ["get", "read", "query", "search", "list"]):
        return "query"
    if any(kw in action_lower for kw in ["validate", "check"]):
        return "validate"
    if any(kw in action_lower for kw in ["submit", "finish", "complete", "answer"]):
        return "submit"
    
    return "default"


def _estimate_confidence_from_history(
    item: Dict[str, Any],
    step_idx: int,
    total_steps: int,
) -> float:
    """Estimate confidence from historical run data."""
    # Check if confidence is already recorded
    if "confidence" in item:
        return item["confidence"]
    
    # Check for error indicators
    response = item.get("result", item.get("response", {}))
    
    if isinstance(response, dict):
        if response.get("error") or response.get("status") == "error":
            return 0.4  # Lower confidence for errors
    
    if isinstance(response, str):
        response_lower = response.lower()
        if "error" in response_lower or "failed" in response_lower:
            return 0.5
    
    # Base confidence with slight decay over long trajectories
    base = 0.8
    decay = 0.02 * (step_idx / max(total_steps, 1))
    
    return max(0.5, base - decay)


def _generate_recommendations(
    mean_confidence: float,
    min_confidence: float,
    trend: str,
    high_uncertainty_steps: List[int],
) -> List[str]:
    """Generate recommendations based on uncertainty analysis."""
    recs = []
    
    if mean_confidence < 0.5:
        recs.append("Consider reducing task complexity or providing more context")
    elif mean_confidence < 0.7:
        recs.append("Agent shows moderate uncertainty - validate critical outputs")
    
    if min_confidence < 0.3:
        recs.append(f"Very low confidence detected - review steps with high uncertainty")
    
    if trend == "increasing":
        recs.append("Uncertainty increasing over time - consider early stopping mechanism")
    
    if len(high_uncertainty_steps) > 3:
        recs.append(f"Multiple high-uncertainty steps ({len(high_uncertainty_steps)}) - consider adding validation checkpoints")
    
    if not recs:
        recs.append("Confidence levels acceptable - proceed with standard validation")
    
    return recs


# Convenience function for quick analysis
def quick_analyze(
    agent: Any,
    task: Any,
    task_name: str = "task",
) -> UncertaintyReport:
    """
    Quick uncertainty analysis for a single task run.
    
    Args:
        agent: AgentBench agent
        task: Task to run
        task_name: Name of the task
    
    Returns:
        Uncertainty report
    
    Example:
        report = quick_analyze(agent, task, "my_task")
        report.print_summary()
    """
    wrapped = create_uncertainty_agent(agent)
    
    # Run task (assuming task has a standard interface)
    try:
        if hasattr(task, 'run'):
            result = task.run(wrapped)
            success = getattr(result, 'success', True)
        else:
            # Manual interaction
            success = True
    except Exception as e:
        success = False
    
    return wrapped.get_uncertainty_report(task_name=task_name, success=success)


__all__ = [
    "UncertaintyReport",
    "UncertaintyCallback",
    "create_uncertainty_agent",
    "analyze_saved_runs",
    "quick_analyze",
]

