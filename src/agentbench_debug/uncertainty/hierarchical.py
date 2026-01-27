"""Hierarchical uncertainty propagation (SAUP-style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class UncertaintyLevel(Enum):
    """Levels of uncertainty in the hierarchy."""
    TOKEN = "token"
    ACTION = "action"
    OBSERVATION = "observation"
    TRAJECTORY = "trajectory"


@dataclass
class TokenUncertainty:
    """Token-level uncertainty from logprobs."""
    token: str
    logprob: float
    probability: float
    
    @property
    def uncertainty(self) -> float:
        """Higher probability = lower uncertainty."""
        return 1.0 - self.probability


@dataclass
class ActionUncertainty:
    """Action-level uncertainty aggregated from tokens."""
    action_type: str
    confidence: float
    token_uncertainties: List[TokenUncertainty] = field(default_factory=list)
    criticality: float = 1.0  # How critical this action type is
    
    @property
    def weighted_uncertainty(self) -> float:
        """Uncertainty weighted by criticality."""
        return (1.0 - self.confidence) * self.criticality


@dataclass
class ObservationUncertainty:
    """Observation-level uncertainty from environment feedback."""
    observation: str
    is_error: bool = False
    is_ambiguous: bool = False
    confidence: float = 1.0
    
    @property
    def uncertainty(self) -> float:
        base = 1.0 - self.confidence
        if self.is_error:
            base = max(base, 0.8)
        if self.is_ambiguous:
            base = max(base, 0.5)
        return base


@dataclass
class StepRecord:
    """Record of a single step in the trajectory."""
    step_idx: int
    action: str
    action_type: str
    observation: str
    confidence: float
    action_uncertainty: Optional[ActionUncertainty] = None
    observation_uncertainty: Optional[ObservationUncertainty] = None
    
    @property
    def combined_uncertainty(self) -> float:
        """Combine action and observation uncertainties."""
        action_u = self.action_uncertainty.weighted_uncertainty if self.action_uncertainty else (1.0 - self.confidence)
        obs_u = self.observation_uncertainty.uncertainty if self.observation_uncertainty else 0.0
        # Weighted average: action is primary
        return 0.7 * action_u + 0.3 * obs_u


class HierarchicalUncertainty:
    """
    SAUP-style hierarchical uncertainty propagation.
    
    Propagates uncertainty from token -> action -> observation -> trajectory levels.
    """
    
    # Default criticality weights for different action types
    DEFAULT_CRITICALITY = {
        "environment_action": 1.0,  # ALFWorld actions
        "query": 0.9,              # Database queries
        "shell_command": 1.0,      # OS commands
        "api_call": 0.9,           # API invocations
        "search": 0.7,             # Search operations
        "navigation": 0.8,         # Navigation actions
        "submit": 1.0,             # Final submissions
        "unknown": 0.8,            # Fallback
    }
    
    def __init__(
        self,
        criticality_weights: Optional[Dict[str, float]] = None,
        uncertainty_threshold: float = 0.5,
    ):
        self.criticality_weights = {
            **self.DEFAULT_CRITICALITY,
            **(criticality_weights or {}),
        }
        self.uncertainty_threshold = uncertainty_threshold
        self.steps: List[StepRecord] = []
    
    def get_criticality(self, action_type: str) -> float:
        """Get criticality weight for an action type."""
        return self.criticality_weights.get(
            action_type,
            self.criticality_weights["unknown"]
        )
    
    def add_step(
        self,
        step_idx: int,
        action: str,
        action_type: str,
        observation: str,
        confidence: float,
        token_logprobs: Optional[List[float]] = None,
        is_error: bool = False,
    ) -> StepRecord:
        """Add a step and compute its hierarchical uncertainty."""
        import math
        
        # Build token-level uncertainty if logprobs available
        token_uncertainties = []
        if token_logprobs:
            for i, lp in enumerate(token_logprobs):
                prob = math.exp(lp)
                token_uncertainties.append(
                    TokenUncertainty(
                        token=f"token_{i}",
                        logprob=lp,
                        probability=prob,
                    )
                )
        
        # Action-level uncertainty
        action_uncertainty = ActionUncertainty(
            action_type=action_type,
            confidence=confidence,
            token_uncertainties=token_uncertainties,
            criticality=self.get_criticality(action_type),
        )
        
        # Observation-level uncertainty
        observation_uncertainty = ObservationUncertainty(
            observation=observation[:100] if observation else "",
            is_error=is_error,
            is_ambiguous=self._detect_ambiguity(observation),
            confidence=0.3 if is_error else 0.9,
        )
        
        # Create step record
        step = StepRecord(
            step_idx=step_idx,
            action=action,
            action_type=action_type,
            observation=observation[:100] if observation else "",
            confidence=confidence,
            action_uncertainty=action_uncertainty,
            observation_uncertainty=observation_uncertainty,
        )
        
        self.steps.append(step)
        return step
    
    def _detect_ambiguity(self, observation: str) -> bool:
        """Detect if an observation is ambiguous."""
        if not observation:
            return False
        
        obs_lower = observation.lower()
        ambiguous_indicators = [
            "multiple",
            "several",
            "could be",
            "might be",
            "unclear",
            "ambiguous",
            "which one",
            "not found",
        ]
        return any(ind in obs_lower for ind in ambiguous_indicators)
    
    def compute_trajectory_uncertainty(self) -> Dict[str, Any]:
        """
        Compute trajectory-level uncertainty from all steps.
        
        Returns comprehensive uncertainty metrics.
        """
        if not self.steps:
            return {
                "mean_uncertainty": 0.0,
                "max_uncertainty": 0.0,
                "min_uncertainty": 0.0,
                "high_uncertainty_steps": 0,
                "trend": "stable",
                "n_steps": 0,
            }
        
        uncertainties = [s.combined_uncertainty for s in self.steps]
        confidences = [s.confidence for s in self.steps]
        
        mean_u = sum(uncertainties) / len(uncertainties)
        mean_conf = sum(confidences) / len(confidences)
        
        # Trend analysis
        if len(self.steps) >= 2:
            mid = len(self.steps) // 2
            first_half = sum(u for u in uncertainties[:mid]) / mid if mid > 0 else 0
            second_half = sum(u for u in uncertainties[mid:]) / (len(uncertainties) - mid)
            
            if second_half > first_half + 0.1:
                trend = "increasing"
            elif second_half < first_half - 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "mean_uncertainty": mean_u,
            "mean_confidence": mean_conf,
            "max_uncertainty": max(uncertainties),
            "min_uncertainty": min(uncertainties),
            "min_confidence": min(confidences),
            "high_uncertainty_steps": sum(
                1 for u in uncertainties if u > self.uncertainty_threshold
            ),
            "trend": trend,
            "n_steps": len(self.steps),
            "trajectory_confidence": 1.0 - mean_u,
        }
    
    def get_critical_steps(self) -> List[StepRecord]:
        """Get steps with high uncertainty and high criticality."""
        return [
            s for s in self.steps
            if s.combined_uncertainty > self.uncertainty_threshold
        ]
    
    def reset(self) -> None:
        """Clear all recorded steps."""
        self.steps.clear()


__all__ = [
    "HierarchicalUncertainty",
    "UncertaintyLevel",
    "TokenUncertainty",
    "ActionUncertainty",
    "ObservationUncertainty",
    "StepRecord",
]

