"""Agent wrapper for real-time uncertainty tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .confidence_extractor import ConfidenceExtractor
from .hierarchical import HierarchicalUncertainty

if TYPE_CHECKING:
    from src.client.agent import AgentClient


@dataclass
class StepInfo:
    """Information about a single inference step."""
    step_idx: int
    input_messages: List[Dict[str, Any]]
    output: str
    confidence: float
    confidence_source: str
    action_type: str = "unknown"
    raw_response: Optional[Dict[str, Any]] = None


class UncertaintyTracker:
    """
    Tracks uncertainty across agent inference steps.
    
    This is a standalone tracker that can be used independently
    of the agent wrapper if needed.
    """
    
    def __init__(
        self,
        task_id: str = "",
        task_type: str = "unknown",
        default_confidence: float = 0.70,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.confidence_extractor = ConfidenceExtractor(default_confidence)
        self.hierarchical = HierarchicalUncertainty()
        self.steps: List[StepInfo] = []
        self._step_counter = 0
    
    def record_inference(
        self,
        input_messages: List[Dict[str, Any]],
        output: str,
        raw_response: Optional[Dict[str, Any]] = None,
        action_type: str = "unknown",
        confidence_override: Optional[float] = None,
    ) -> StepInfo:
        """Record an inference call and its uncertainty."""
        step_idx = self._step_counter
        self._step_counter += 1
        
        # Extract confidence
        if confidence_override is not None:
            confidence = confidence_override
            source = "override"
        else:
            confidence, source = self.confidence_extractor.extract(
                raw_response, output
            )
        
        # Track in hierarchical system
        self.hierarchical.add_step(
            step_idx=step_idx,
            action=output[:100] if output else "",
            action_type=action_type,
            observation="",  # Not available at inference time
            confidence=confidence,
        )
        
        step = StepInfo(
            step_idx=step_idx,
            input_messages=input_messages,
            output=output,
            confidence=confidence,
            confidence_source=source,
            action_type=action_type,
            raw_response=raw_response,
        )
        
        self.steps.append(step)
        return step
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked uncertainty."""
        if not self.steps:
            return {
                "n_steps": 0,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "trend": "stable",
            }
        
        confidences = [s.confidence for s in self.steps]
        trajectory = self.hierarchical.compute_trajectory_uncertainty()
        
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "n_steps": len(self.steps),
            "mean_confidence": trajectory.get("mean_confidence", sum(confidences) / len(confidences)),
            "min_confidence": trajectory.get("min_confidence", min(confidences)),
            "trend": trajectory.get("trend", "stable"),
            "high_uncertainty_steps": trajectory.get("high_uncertainty_steps", 0),
        }
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.steps.clear()
        self.hierarchical.reset()
        self._step_counter = 0


class UncertaintyAwareAgent:
    """
    Wrapper around an AgentClient that tracks uncertainty.
    
    Intercepts inference calls to extract and track confidence.
    """
    
    def __init__(
        self,
        agent: "AgentClient",
        task_id: str = "",
        task_type: str = "unknown",
        default_confidence: float = 0.70,
        on_step_callback: Optional[Callable[[StepInfo], None]] = None,
    ):
        """
        Wrap an agent with uncertainty tracking.
        
        Args:
            agent: The AgentClient to wrap
            task_id: Task identifier
            task_type: Type of task (alfworld, dbbench, etc.)
            default_confidence: Default confidence when extraction fails
            on_step_callback: Optional callback called after each step
        """
        self._agent = agent
        self._tracker = UncertaintyTracker(
            task_id=task_id,
            task_type=task_type,
            default_confidence=default_confidence,
        )
        self._on_step = on_step_callback
        self._last_raw_response: Optional[Dict[str, Any]] = None
    
    @property
    def tracker(self) -> UncertaintyTracker:
        """Access the underlying tracker."""
        return self._tracker
    
    @property
    def name(self) -> str:
        """Get the agent's name."""
        return getattr(self._agent, "name", "unknown")
    
    def inference(self, history: Any) -> str:
        """
        Perform inference with uncertainty tracking.
        
        Args:
            history: Message history (format depends on agent)
        
        Returns:
            Agent's output string
        """
        # Call the underlying agent
        output = self._agent.inference(history)
        
        # Try to get raw response if available
        raw_response = None
        if hasattr(self._agent, "get_last_raw_response"):
            raw_response = self._agent.get_last_raw_response()
        elif hasattr(self._agent, "_last_raw_response"):
            raw_response = self._agent._last_raw_response
        
        self._last_raw_response = raw_response
        
        # Infer action type from task type
        action_type = self._infer_action_type(output)
        
        # Convert history to list for recording
        history_list = history if isinstance(history, list) else [{"content": str(history)}]
        
        # Record the step
        step = self._tracker.record_inference(
            input_messages=history_list,
            output=output,
            raw_response=raw_response,
            action_type=action_type,
        )
        
        # Callback
        if self._on_step:
            self._on_step(step)
        
        return output
    
    def _infer_action_type(self, output: str) -> str:
        """Infer action type from output and task type."""
        task_type = self._tracker.task_type.lower()
        
        # Map task types to action types
        if "alfworld" in task_type:
            return "environment_action"
        elif "dbbench" in task_type or "db" in task_type:
            return "query"
        elif "os" in task_type:
            return "shell_command"
        elif "kg" in task_type or "knowledge" in task_type:
            return "query"
        elif "webshop" in task_type or "web" in task_type:
            return "search"
        elif "swebench" in task_type or "swe" in task_type:
            return "shell_command"
        
        return "unknown"
    
    def get_last_raw_response(self) -> Optional[Dict[str, Any]]:
        """Get the last raw API response."""
        return self._last_raw_response
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of tracked uncertainty."""
        return self._tracker.get_summary()
    
    def reset(self) -> None:
        """Reset uncertainty tracking."""
        self._tracker.reset()
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped agent."""
        return getattr(self._agent, name)


def wrap_agent(
    agent: "AgentClient",
    task_id: str = "",
    task_type: str = "unknown",
    **kwargs,
) -> UncertaintyAwareAgent:
    """
    Convenience function to wrap an agent with uncertainty tracking.
    
    Args:
        agent: The agent to wrap
        task_id: Task identifier
        task_type: Type of task
        **kwargs: Additional arguments passed to UncertaintyAwareAgent
    
    Returns:
        UncertaintyAwareAgent wrapper
    """
    return UncertaintyAwareAgent(
        agent=agent,
        task_id=task_id,
        task_type=task_type,
        **kwargs,
    )


__all__ = [
    "UncertaintyAwareAgent",
    "UncertaintyTracker",
    "StepInfo",
    "wrap_agent",
]

