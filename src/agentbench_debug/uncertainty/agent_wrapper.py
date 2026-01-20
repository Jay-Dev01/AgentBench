"""
Uncertainty-Aware Agent Wrapper.

Wraps AgentBench agents to automatically capture confidence/uncertainty
signals during inference, enabling real-time uncertainty estimation.

This module provides:
- UncertaintyAwareAgent: Wrapper that intercepts agent calls
- Request modification to request logprobs from LLM APIs
- Automatic confidence extraction and recording
- Integration with the orchestration harness
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .confidence_extractor import ConfidenceExtractor, ConfidenceSignals
from .hierarchical import HierarchicalUncertaintyPropagator


@dataclass
class InferenceRecord:
    """Record of a single inference call with uncertainty."""
    timestamp: datetime
    step_idx: int
    
    # Input
    messages: List[Dict[str, str]]
    tools_available: List[str]
    
    # Output
    response_content: str
    raw_response: Optional[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    
    # Confidence
    confidence_signals: ConfidenceSignals
    confidence: float
    
    # Timing
    latency_ms: float
    tokens_used: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class UncertaintyAwareAgent:
    """
    Wrapper for AgentBench agents that captures uncertainty signals.
    
    This wrapper:
    1. Intercepts inference calls
    2. Modifies requests to include logprobs (where supported)
    3. Extracts confidence from responses
    4. Records all inference data for analysis
    
    Usage:
        from agentbench_debug.uncertainty import UncertaintyAwareAgent
        
        # Wrap an existing agent
        wrapped_agent = UncertaintyAwareAgent(original_agent)
        
        # Use like normal agent
        response = wrapped_agent.inference(messages)
        
        # Get uncertainty data
        records = wrapped_agent.get_inference_records()
        confidence = wrapped_agent.get_last_confidence()
    """
    
    def __init__(
        self,
        agent: Any,
        api_type: str = "auto",
        request_logprobs: bool = True,
        logprobs_count: int = 5,
        uncertainty_threshold: float = 0.35,
        enable_propagation: bool = True,
    ):
        """
        Initialize uncertainty-aware wrapper.
        
        Args:
            agent: The AgentBench agent to wrap
            api_type: API type for confidence extraction ("openai", "gemini", "auto")
            request_logprobs: Whether to modify requests to request logprobs
            logprobs_count: Number of top logprobs to request
            uncertainty_threshold: Threshold for flagging high uncertainty
            enable_propagation: Whether to track hierarchical uncertainty
        """
        self._agent = agent
        self._api_type = api_type
        self._request_logprobs = request_logprobs
        self._logprobs_count = logprobs_count
        self._uncertainty_threshold = uncertainty_threshold
        
        # Components
        self._extractor = ConfidenceExtractor()
        self._propagator = HierarchicalUncertaintyPropagator(
            uncertainty_threshold=uncertainty_threshold
        ) if enable_propagation else None
        
        # State
        self._records: List[InferenceRecord] = []
        self._step_idx = 0
        self._last_confidence: Optional[ConfidenceSignals] = None
        self._raw_response: Optional[Dict[str, Any]] = None
    
    # =========================================================================
    # Main Interface
    # =========================================================================
    
    def inference(self, history: Any) -> str:
        """
        Perform inference with uncertainty tracking.
        
        Args:
            history: Input history/messages for the agent
        
        Returns:
            Agent response string
        """
        start_time = time.time()
        
        # Parse input
        messages, tools = self._parse_input(history)
        
        # Modify request for logprobs if supported
        modified_history = self._modify_request(history)
        
        # Get raw response (try to capture full response)
        response_content, raw_response = self._call_agent(modified_history)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract confidence
        confidence_signals = self._extractor.extract(
            raw_response or response_content,
            api_type=self._api_type,
            content=response_content,
        )
        self._last_confidence = confidence_signals
        self._raw_response = raw_response
        
        # Extract tool calls
        tool_calls = self._extract_tool_calls(response_content, raw_response)
        
        # Estimate tokens (rough)
        tokens_used = self._estimate_tokens(messages, response_content)
        
        # Record uncertainty in propagator
        if self._propagator:
            action_name = tool_calls[0]["name"] if tool_calls else "respond"
            action_type = self._infer_action_type(action_name, tool_calls)
            
            self._propagator.compute_action_uncertainty(
                action_name=action_name,
                action_type=action_type,
                confidence=confidence_signals.confidence,
            )
            
            self._propagator.compute_observation_uncertainty(
                response=raw_response or response_content,
            )
            
            self._propagator.next_step()
        
        # Create record
        record = InferenceRecord(
            timestamp=datetime.now(),
            step_idx=self._step_idx,
            messages=messages,
            tools_available=tools,
            response_content=response_content,
            raw_response=raw_response,
            tool_calls=tool_calls,
            confidence_signals=confidence_signals,
            confidence=confidence_signals.confidence,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        self._records.append(record)
        self._step_idx += 1
        
        return response_content
    
    def _call_agent(self, history: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Call underlying agent and capture response."""
        # Check if agent has a method to get raw response
        if hasattr(self._agent, 'inference_with_raw'):
            response, raw = self._agent.inference_with_raw(history)
            return response, raw
        
        # Check if agent stores raw response
        response = self._agent.inference(history)
        
        # Try to get raw response from agent
        raw_response = None
        if hasattr(self._agent, '_last_raw_response'):
            raw_response = self._agent._last_raw_response
        elif hasattr(self._agent, 'last_response'):
            raw_response = self._agent.last_response
        
        return response, raw_response
    
    # =========================================================================
    # Request Modification
    # =========================================================================
    
    def _modify_request(self, history: Any) -> Any:
        """Modify request to include logprobs parameter."""
        if not self._request_logprobs:
            return history
        
        # If history is a dict, we can add logprobs parameter
        if isinstance(history, dict):
            modified = history.copy()
            
            # OpenAI format
            if "messages" in modified:
                modified["logprobs"] = True
                modified["top_logprobs"] = self._logprobs_count
            
            # Gemini format - use candidateCount for multiple samples
            if "contents" in modified or "prompt" in modified:
                modified["generationConfig"] = modified.get("generationConfig", {})
                # Gemini doesn't have direct logprobs, but we can request candidates
            
            return modified
        
        return history
    
    def _parse_input(
        self,
        history: Any,
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Parse input to extract messages and tools."""
        messages = []
        tools = []
        
        if isinstance(history, dict):
            messages = history.get("messages", [])
            tools_data = history.get("tools", [])
            tools = [t.get("function", {}).get("name", "") for t in tools_data if isinstance(t, dict)]
        elif isinstance(history, list):
            messages = history
        
        return messages, tools
    
    def _extract_tool_calls(
        self,
        content: str,
        raw_response: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract tool calls from response."""
        tool_calls = []
        
        # Check raw response for tool calls
        if raw_response:
            # OpenAI format
            choices = raw_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                if "tool_calls" in message:
                    for tc in message["tool_calls"]:
                        tool_calls.append({
                            "id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "arguments": tc.get("function", {}).get("arguments"),
                        })
        
        # Try to parse from content if no tool calls found
        if not tool_calls and content:
            # Look for action patterns
            import re
            
            # Pattern: Action: tool_name[args]
            match = re.search(r'Action:\s*(\w+)\[(.*?)\]', content)
            if match:
                tool_calls.append({
                    "name": match.group(1),
                    "arguments": match.group(2),
                })
            
            # Pattern: <action>tool_name</action>
            match = re.search(r'<action>(.+?)</action>', content)
            if match:
                tool_calls.append({
                    "name": match.group(1).strip(),
                    "arguments": "",
                })
        
        return tool_calls
    
    def _infer_action_type(
        self,
        action_name: str,
        tool_calls: List[Dict[str, Any]],
    ) -> str:
        """Infer action type from action name."""
        name_lower = action_name.lower()
        
        if any(kw in name_lower for kw in ["auth", "login", "token"]):
            return "auth"
        if any(kw in name_lower for kw in ["delete", "remove", "drop"]):
            return "delete"
        if any(kw in name_lower for kw in ["create", "write", "insert", "update", "post"]):
            return "write"
        if any(kw in name_lower for kw in ["get", "read", "query", "search", "list", "fetch"]):
            return "query"
        if any(kw in name_lower for kw in ["validate", "check", "verify"]):
            return "validate"
        if any(kw in name_lower for kw in ["sync", "commit", "submit"]):
            return "sync"
        
        return "default"
    
    def _estimate_tokens(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> int:
        """Rough token estimation."""
        # Very rough: ~4 characters per token
        input_chars = sum(len(m.get("content", "")) for m in messages)
        output_chars = len(response)
        
        return (input_chars + output_chars) // 4
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def get_inference_records(self) -> List[InferenceRecord]:
        """Get all recorded inferences."""
        return list(self._records)
    
    def get_last_confidence(self) -> Optional[float]:
        """Get confidence from last inference."""
        return self._last_confidence.confidence if self._last_confidence else None
    
    def get_last_confidence_signals(self) -> Optional[ConfidenceSignals]:
        """Get full confidence signals from last inference."""
        return self._last_confidence
    
    def get_uncertainty_analysis(self) -> Optional[Dict[str, Any]]:
        """Get hierarchical uncertainty analysis if enabled."""
        if not self._propagator:
            return None
        
        result = self._propagator.analyze_complete()
        
        return {
            "aggregated_score": result.aggregated_score,
            "trajectory_uncertainty": result.trajectory_level.cumulative_uncertainty,
            "final_confidence": result.trajectory_level.final_confidence,
            "trend": result.trajectory_level.uncertainty_trend,
            "critical_steps": result.trajectory_level.critical_steps,
            "n_actions": len(result.action_level),
        }
    
    def get_confidence_history(self) -> List[float]:
        """Get list of confidence scores across all inferences."""
        return [r.confidence for r in self._records]
    
    def get_high_uncertainty_steps(self) -> List[int]:
        """Get step indices with high uncertainty."""
        return [
            r.step_idx for r in self._records
            if r.confidence < (1.0 - self._uncertainty_threshold)
        ]
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self._records = []
        self._step_idx = 0
        self._last_confidence = None
        self._raw_response = None
        if self._propagator:
            self._propagator.reset()
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_records(self) -> List[Dict[str, Any]]:
        """Export records as serializable dictionaries."""
        return [
            {
                "step_idx": r.step_idx,
                "timestamp": r.timestamp.isoformat(),
                "confidence": r.confidence,
                "confidence_source": r.confidence_signals.source,
                "tool_calls": r.tool_calls,
                "latency_ms": r.latency_ms,
                "tokens_used": r.tokens_used,
                "uncertainty_phrases": r.confidence_signals.uncertainty_phrases,
                "certainty_phrases": r.confidence_signals.certainty_phrases,
                "mean_logprob": r.confidence_signals.mean_logprob,
                "perplexity": r.confidence_signals.perplexity,
            }
            for r in self._records
        ]
    
    # =========================================================================
    # Proxy Methods
    # =========================================================================
    
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to wrapped agent."""
        return getattr(self._agent, name)


def wrap_agent(
    agent: Any,
    api_type: str = "auto",
    request_logprobs: bool = True,
    uncertainty_threshold: float = 0.35,
) -> UncertaintyAwareAgent:
    """
    Convenience function to wrap an agent with uncertainty tracking.
    
    Args:
        agent: AgentBench agent to wrap
        api_type: API type ("openai", "gemini", "anthropic", "auto")
        request_logprobs: Whether to request logprobs from API
        uncertainty_threshold: Threshold for high uncertainty
    
    Returns:
        Wrapped agent with uncertainty tracking
    """
    return UncertaintyAwareAgent(
        agent=agent,
        api_type=api_type,
        request_logprobs=request_logprobs,
        uncertainty_threshold=uncertainty_threshold,
    )


class UncertaintyTracker:
    """
    Standalone uncertainty tracker for manual integration.
    
    Use this when you can't wrap the agent but want to track
    uncertainty from captured responses.
    
    Usage:
        tracker = UncertaintyTracker()
        
        # After each agent call
        response = agent.inference(messages)
        confidence = tracker.record_response(response, api_response)
        
        # Get analysis
        analysis = tracker.get_analysis()
    """
    
    def __init__(
        self,
        api_type: str = "auto",
        uncertainty_threshold: float = 0.35,
    ):
        """Initialize tracker."""
        self._extractor = ConfidenceExtractor()
        self._propagator = HierarchicalUncertaintyPropagator(
            uncertainty_threshold=uncertainty_threshold
        )
        self._api_type = api_type
        self._threshold = uncertainty_threshold
        
        self._confidences: List[float] = []
        self._signals: List[ConfidenceSignals] = []
        self._step_idx = 0
    
    def record_response(
        self,
        content: str,
        raw_response: Optional[Dict[str, Any]] = None,
        action_name: str = "action",
        action_type: str = "default",
    ) -> float:
        """
        Record a response and extract confidence.
        
        Args:
            content: Response content string
            raw_response: Optional raw API response dict
            action_name: Name of the action taken
            action_type: Type of action
        
        Returns:
            Extracted confidence score
        """
        # Extract confidence
        signals = self._extractor.extract(
            raw_response or content,
            api_type=self._api_type,
            content=content,
        )
        
        self._confidences.append(signals.confidence)
        self._signals.append(signals)
        
        # Update propagator
        self._propagator.compute_action_uncertainty(
            action_name=action_name,
            action_type=action_type,
            confidence=signals.confidence,
        )
        self._propagator.compute_observation_uncertainty(response=raw_response or content)
        self._propagator.next_step()
        
        self._step_idx += 1
        
        return signals.confidence
    
    def get_confidence_history(self) -> List[float]:
        """Get all confidence scores."""
        return list(self._confidences)
    
    def get_mean_confidence(self) -> float:
        """Get mean confidence across all steps."""
        return sum(self._confidences) / len(self._confidences) if self._confidences else 0.5
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get full uncertainty analysis."""
        result = self._propagator.analyze_complete()
        
        return {
            "n_steps": self._step_idx,
            "mean_confidence": self.get_mean_confidence(),
            "min_confidence": min(self._confidences) if self._confidences else 0.5,
            "aggregated_score": result.aggregated_score,
            "trajectory_uncertainty": result.trajectory_level.cumulative_uncertainty,
            "final_confidence": result.trajectory_level.final_confidence,
            "trend": result.trajectory_level.uncertainty_trend,
            "critical_steps": result.trajectory_level.critical_steps,
            "high_uncertainty_count": len(result.trajectory_level.critical_steps),
        }
    
    def is_high_uncertainty(self) -> bool:
        """Check if current trajectory has high uncertainty."""
        if not self._confidences:
            return False
        
        mean_conf = self.get_mean_confidence()
        return mean_conf < (1.0 - self._threshold)
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._confidences = []
        self._signals = []
        self._step_idx = 0
        self._propagator.reset()


__all__ = [
    "UncertaintyAwareAgent",
    "UncertaintyTracker",
    "InferenceRecord",
    "wrap_agent",
]

