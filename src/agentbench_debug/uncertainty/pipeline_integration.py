"""Pipeline integration helpers for uncertainty tracking in AgentBench."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .agent_wrapper import UncertaintyAwareAgent, UncertaintyTracker, StepInfo
from .confidence_extractor import ConfidenceExtractor
from .orchestration_harness import OrchestrationHarness, WorkflowResult

if TYPE_CHECKING:
    from src.client.agent import AgentClient


@dataclass
class UncertaintyReport:
    """Summary report of uncertainty across multiple runs."""
    total_runs: int
    successful_runs: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    high_uncertainty_runs: int
    confidence_by_task_type: Dict[str, float]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": self.successful_runs / self.total_runs if self.total_runs > 0 else 0,
            "mean_confidence": self.mean_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "high_uncertainty_runs": self.high_uncertainty_runs,
            "confidence_by_task_type": self.confidence_by_task_type,
            "n_runs_detailed": len(self.runs),
        }
    
    @classmethod
    def from_runs(cls, runs: List[Dict[str, Any]], threshold: float = 0.5) -> "UncertaintyReport":
        """Create report from list of run summaries."""
        if not runs:
            return cls(
                total_runs=0,
                successful_runs=0,
                mean_confidence=0.0,
                min_confidence=0.0,
                max_confidence=0.0,
                high_uncertainty_runs=0,
                confidence_by_task_type={},
            )
        
        confidences = [r.get("mean_confidence", 0.7) for r in runs]
        successes = sum(1 for r in runs if r.get("success", False))
        
        # Group by task type
        by_task = {}
        for r in runs:
            task_type = r.get("task_type", "unknown")
            if task_type not in by_task:
                by_task[task_type] = []
            by_task[task_type].append(r.get("mean_confidence", 0.7))
        
        confidence_by_task = {
            k: sum(v) / len(v) for k, v in by_task.items()
        }
        
        high_u_count = sum(1 for c in confidences if c < threshold)
        
        return cls(
            total_runs=len(runs),
            successful_runs=successes,
            mean_confidence=sum(confidences) / len(confidences),
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            high_uncertainty_runs=high_u_count,
            confidence_by_task_type=confidence_by_task,
            runs=runs,
        )


class UncertaintyCallback:
    """
    Callback handler for uncertainty tracking during AgentBench runs.
    
    Can be attached to task clients to track uncertainty without
    modifying the core pipeline.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        print_steps: bool = True,
    ):
        self.output_dir = output_dir
        self.print_steps = print_steps
        self.run_summaries: List[Dict[str, Any]] = []
        self._current_tracker: Optional[UncertaintyTracker] = None
    
    def on_run_start(self, task_id: str, task_type: str) -> UncertaintyTracker:
        """Called when a run starts."""
        self._current_tracker = UncertaintyTracker(
            task_id=task_id,
            task_type=task_type,
        )
        return self._current_tracker
    
    def on_step(self, step: StepInfo) -> None:
        """Called after each inference step."""
        if self.print_steps:
            print(f"  [Step {step.step_idx}] confidence={step.confidence:.3f} ({step.confidence_source})")
    
    def on_run_end(self, success: bool) -> Dict[str, Any]:
        """Called when a run ends."""
        if self._current_tracker is None:
            return {}
        
        summary = self._current_tracker.get_summary()
        summary["success"] = success
        self.run_summaries.append(summary)
        
        if self.print_steps:
            print(f"  Run complete: mean_conf={summary['mean_confidence']:.3f}, trend={summary['trend']}")
        
        return summary
    
    def get_report(self) -> UncertaintyReport:
        """Get aggregated report."""
        return UncertaintyReport.from_runs(self.run_summaries)
    
    def save_report(self, filepath: str) -> None:
        """Save report to JSON file."""
        report = self.get_report()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)


def create_uncertainty_agent(
    agent: "AgentClient",
    task_name: str,
    callback: Optional[UncertaintyCallback] = None,
) -> UncertaintyAwareAgent:
    """
    Create an uncertainty-aware agent for a specific task.
    
    Args:
        agent: The base agent to wrap
        task_name: Name of the task (e.g., "alfworld-std")
        callback: Optional callback for logging
    
    Returns:
        UncertaintyAwareAgent wrapper
    """
    # Detect task type from name
    task_type = _detect_task_type(task_name)
    
    # Create step callback if we have a callback handler
    on_step = callback.on_step if callback else None
    
    return UncertaintyAwareAgent(
        agent=agent,
        task_id=task_name,
        task_type=task_type,
        on_step_callback=on_step,
    )


def _detect_task_type(task_name: str) -> str:
    """Detect task type from task name."""
    name_lower = task_name.lower()
    
    if "alfworld" in name_lower:
        return "alfworld"
    elif "dbbench" in name_lower or "db" in name_lower:
        return "dbbench"
    elif "os" in name_lower and "webshop" not in name_lower:
        return "os_interaction"
    elif "kg" in name_lower or "knowledge" in name_lower:
        return "knowledgegraph"
    elif "webshop" in name_lower:
        return "webshop"
    elif "swebench" in name_lower or "swe" in name_lower:
        return "swebench"
    elif "toolemu" in name_lower:
        return "toolemu"
    
    return "unknown"


def analyze_saved_runs(
    runs_jsonl_path: str,
    output_path: Optional[str] = None,
) -> UncertaintyReport:
    """
    Analyze uncertainty from saved runs.jsonl files.
    
    This performs post-hoc analysis on existing AgentBench results.
    
    Args:
        runs_jsonl_path: Path to runs.jsonl file
        output_path: Optional path to save analysis
    
    Returns:
        UncertaintyReport with analysis results
    """
    extractor = ConfidenceExtractor()
    runs = []
    
    with open(runs_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                run = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Extract info from run
            output = run.get("output", {})
            history = output.get("history", [])
            result = run.get("result", False)
            
            # Compute confidence from history (semantic analysis)
            confidences = []
            for msg in history:
                if msg.get("role") == "agent":
                    content = msg.get("content", "")
                    conf, _ = extractor._extract_from_semantics(content)
                    confidences.append(conf)
            
            mean_conf = sum(confidences) / len(confidences) if confidences else 0.7
            
            runs.append({
                "task_id": str(run.get("index", "")),
                "task_type": _detect_task_type(runs_jsonl_path),
                "success": result,
                "mean_confidence": mean_conf,
                "n_steps": len(confidences),
                "trend": "stable",
            })
    
    report = UncertaintyReport.from_runs(runs)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
    
    return report


__all__ = [
    "UncertaintyReport",
    "UncertaintyCallback",
    "create_uncertainty_agent",
    "analyze_saved_runs",
]

