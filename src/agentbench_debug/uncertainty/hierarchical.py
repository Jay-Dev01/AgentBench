"""
Hierarchical Uncertainty Propagation (SAUP-style) for API orchestration.

Implements multi-level uncertainty tracking:
- Token-level: Entropy over LLM output distribution
- Action-level: Confidence in choosing correct API action
- Observation-level: Reliability of API response interpretation
- Trajectory-level: Cumulative confidence across entire workflow

Based on Situational Awareness Uncertainty Propagation (SAUP) principles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ActionCriticality(Enum):
    """Action criticality levels for weighted uncertainty."""
    AUTH = 2.0       # Authentication actions (high weight)
    DELETE = 2.0     # Destructive actions (high weight)
    WRITE = 1.5      # Write/modify actions
    QUERY = 0.8      # Read-only queries
    VALIDATE = 1.0   # Validation steps
    SYNC = 1.2       # Synchronization actions
    DEFAULT = 1.0    # Default weight


# HMM states for trajectory-level uncertainty
WORKFLOW_STATES = ["auth", "query", "validate", "sync", "recover", "complete"]


@dataclass
class TokenUncertainty:
    """Token-level uncertainty from LLM output distribution."""
    entropy: float                    # Shannon entropy: H = -Σ p_i log(p_i)
    top_p_mass: float                 # Probability mass of top-p tokens
    perplexity: float                 # exp(H)
    low_prob_tokens: List[str]        # Tokens with p < threshold
    max_logprob: float                # Highest log probability
    mean_logprob: float               # Average log probability


@dataclass
class ActionUncertainty:
    """Action-level uncertainty for a single decision step."""
    step_idx: int
    action_type: str                  # e.g., "auth", "query", "delete"
    action_name: str                  # API/tool name
    confidence: float                 # 0-1 confidence score
    criticality_weight: float         # Weight based on action type
    weighted_uncertainty: float       # confidence * weight
    alternatives_considered: int      # Number of alternative actions
    selection_entropy: float          # Entropy over action alternatives
    parameters_uncertainty: float     # Uncertainty in parameter values
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationUncertainty:
    """Observation-level uncertainty for API response interpretation."""
    step_idx: int
    response_type: str                # success, error, partial, timeout
    schema_validation_prob: float     # Probability response matches expected schema
    semantic_consistency: float       # Consistency with previous observations
    error_signal_strength: float      # Clarity of error signals (0=ambiguous, 1=clear)
    interpretation_confidence: float  # Confidence in correct interpretation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryUncertainty:
    """Trajectory-level cumulative uncertainty across workflow."""
    trajectory_id: str
    total_steps: int
    cumulative_uncertainty: float     # Weighted sum of step uncertainties
    uncertainty_trend: str            # increasing, decreasing, stable
    critical_steps: List[int]         # Steps with uncertainty > threshold
    recovery_points: List[int]        # Steps where uncertainty reset (recovery)
    hmm_state_sequence: List[str]     # Inferred HMM state sequence
    final_confidence: float           # Final trajectory confidence


@dataclass
class HierarchicalUncertaintyResult:
    """Complete hierarchical uncertainty analysis result."""
    token_level: Optional[TokenUncertainty]
    action_level: List[ActionUncertainty]
    observation_level: List[ObservationUncertainty]
    trajectory_level: TrajectoryUncertainty
    aggregated_score: float           # Single summary score 0-1 (1=certain)
    calibrated: bool                  # Whether calibration was applied


class HierarchicalUncertaintyPropagator:
    """
    SAUP-style hierarchical uncertainty propagation for API orchestration.
    
    Tracks uncertainty at multiple levels and propagates through workflow,
    using situational weights and optional HMM for dynamic weight learning.
    """
    
    def __init__(
        self,
        action_weights: Optional[Dict[str, float]] = None,
        uncertainty_threshold: float = 0.35,
        enable_hmm: bool = True,
        decay_factor: float = 0.9,
    ):
        """
        Initialize the propagator.
        
        Args:
            action_weights: Custom weights for action types
            uncertainty_threshold: Threshold for flagging high uncertainty
            enable_hmm: Whether to use HMM for state sequence inference
            decay_factor: Decay factor for temporal uncertainty smoothing
        """
        self.action_weights = action_weights or {
            "auth": ActionCriticality.AUTH.value,
            "authenticate": ActionCriticality.AUTH.value,
            "login": ActionCriticality.AUTH.value,
            "delete": ActionCriticality.DELETE.value,
            "remove": ActionCriticality.DELETE.value,
            "write": ActionCriticality.WRITE.value,
            "create": ActionCriticality.WRITE.value,
            "update": ActionCriticality.WRITE.value,
            "query": ActionCriticality.QUERY.value,
            "get": ActionCriticality.QUERY.value,
            "read": ActionCriticality.QUERY.value,
            "list": ActionCriticality.QUERY.value,
            "validate": ActionCriticality.VALIDATE.value,
            "check": ActionCriticality.VALIDATE.value,
            "sync": ActionCriticality.SYNC.value,
            "commit": ActionCriticality.SYNC.value,
        }
        self.threshold = uncertainty_threshold
        self.enable_hmm = enable_hmm
        self.decay_factor = decay_factor
        
        # HMM transition matrix (simplified)
        self._init_hmm()
        
        # Running state
        self._action_uncertainties: List[ActionUncertainty] = []
        self._observation_uncertainties: List[ObservationUncertainty] = []
        self._step_idx = 0
    
    def _init_hmm(self) -> None:
        """Initialize HMM transition probabilities."""
        n_states = len(WORKFLOW_STATES)
        # Simple transition matrix: auth -> query -> validate -> sync -> complete
        self.transition_matrix = np.zeros((n_states, n_states))
        
        state_idx = {s: i for i, s in enumerate(WORKFLOW_STATES)}
        
        # Define likely transitions
        transitions = [
            ("auth", "query", 0.7), ("auth", "auth", 0.2), ("auth", "recover", 0.1),
            ("query", "validate", 0.5), ("query", "query", 0.3), ("query", "sync", 0.15),
            ("query", "recover", 0.05),
            ("validate", "sync", 0.6), ("validate", "query", 0.2), ("validate", "validate", 0.15),
            ("validate", "recover", 0.05),
            ("sync", "complete", 0.5), ("sync", "query", 0.2), ("sync", "validate", 0.15),
            ("sync", "recover", 0.1), ("sync", "sync", 0.05),
            ("recover", "auth", 0.3), ("recover", "query", 0.4), ("recover", "recover", 0.2),
            ("recover", "complete", 0.1),
            ("complete", "complete", 1.0),
        ]
        
        for from_state, to_state, prob in transitions:
            self.transition_matrix[state_idx[from_state], state_idx[to_state]] = prob
        
        # Normalize rows
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums
    
    # =========================================================================
    # Token-Level Uncertainty
    # =========================================================================
    
    def compute_token_uncertainty(
        self,
        logprobs: Optional[List[float]] = None,
        token_probs: Optional[List[Dict[str, float]]] = None,
        low_prob_threshold: float = 0.1,
    ) -> Optional[TokenUncertainty]:
        """
        Compute token-level uncertainty from LLM output probabilities.
        
        Args:
            logprobs: Log probabilities for generated tokens
            token_probs: Per-token probability distributions
            low_prob_threshold: Threshold for flagging low-probability tokens
        
        Returns:
            TokenUncertainty or None if no probability info available
        """
        if logprobs is None and token_probs is None:
            return None
        
        if logprobs is not None:
            # Convert log probs to regular probs
            probs = [math.exp(lp) for lp in logprobs]
            
            # Entropy approximation from top token probs
            entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0) / max(len(probs), 1)
            perplexity = math.exp(entropy)
            
            max_logprob = max(logprobs) if logprobs else 0.0
            mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
            
            # Top-p mass (assume we have top tokens)
            top_p_mass = sum(sorted(probs, reverse=True)[:5])
            
            low_prob_tokens = [f"token_{i}" for i, p in enumerate(probs) if p < low_prob_threshold]
            
            return TokenUncertainty(
                entropy=entropy,
                top_p_mass=top_p_mass,
                perplexity=perplexity,
                low_prob_tokens=low_prob_tokens,
                max_logprob=max_logprob,
                mean_logprob=mean_logprob,
            )
        
        # Handle token_probs format
        if token_probs:
            all_entropies = []
            all_max_probs = []
            low_prob_tokens = []
            
            for i, dist in enumerate(token_probs):
                probs = list(dist.values())
                if probs:
                    # Token entropy
                    h = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                    all_entropies.append(h)
                    all_max_probs.append(max(probs))
                    
                    # Check for low-confidence tokens
                    if max(probs) < low_prob_threshold:
                        token = max(dist, key=dist.get)
                        low_prob_tokens.append(token)
            
            avg_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
            
            return TokenUncertainty(
                entropy=avg_entropy,
                top_p_mass=sum(all_max_probs) / len(all_max_probs) if all_max_probs else 0.0,
                perplexity=math.exp(avg_entropy),
                low_prob_tokens=low_prob_tokens,
                max_logprob=math.log(max(all_max_probs)) if all_max_probs else 0.0,
                mean_logprob=sum(math.log(p + 1e-10) for p in all_max_probs) / len(all_max_probs)
                if all_max_probs else 0.0,
            )
        
        return None
    
    # =========================================================================
    # Action-Level Uncertainty
    # =========================================================================
    
    def compute_action_uncertainty(
        self,
        action_name: str,
        action_type: Optional[str] = None,
        confidence: float = 1.0,
        alternatives: Optional[List[Tuple[str, float]]] = None,
        parameter_confidences: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionUncertainty:
        """
        Compute action-level uncertainty for a decision step.
        
        Args:
            action_name: Name of the action/tool being called
            action_type: Type category (auth, query, delete, etc.)
            confidence: Base confidence in this action (0-1)
            alternatives: List of (action_name, score) alternatives considered
            parameter_confidences: Confidence for each parameter
            metadata: Additional context
        
        Returns:
            ActionUncertainty for this step
        """
        # Infer action type from name if not provided
        if action_type is None:
            action_type = self._infer_action_type(action_name)
        
        # Get criticality weight
        weight = self.action_weights.get(
            action_type.lower(), 
            self.action_weights.get(action_name.lower(), ActionCriticality.DEFAULT.value)
        )
        
        # Compute selection entropy from alternatives
        selection_entropy = 0.0
        n_alternatives = 0
        if alternatives:
            n_alternatives = len(alternatives)
            scores = [s for _, s in alternatives]
            total = sum(scores)
            if total > 0:
                probs = [s / total for s in scores]
                selection_entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        
        # Compute parameter uncertainty
        params_uncertainty = 0.0
        if parameter_confidences:
            params_uncertainty = 1.0 - (sum(parameter_confidences.values()) / len(parameter_confidences))
        
        # Weighted uncertainty: (1 - confidence) * weight
        weighted_unc = (1.0 - confidence) * weight
        
        result = ActionUncertainty(
            step_idx=self._step_idx,
            action_type=action_type,
            action_name=action_name,
            confidence=confidence,
            criticality_weight=weight,
            weighted_uncertainty=weighted_unc,
            alternatives_considered=n_alternatives,
            selection_entropy=selection_entropy,
            parameters_uncertainty=params_uncertainty,
            metadata=metadata or {},
        )
        
        self._action_uncertainties.append(result)
        return result
    
    def _infer_action_type(self, action_name: str) -> str:
        """Infer action type from action name."""
        name_lower = action_name.lower()
        
        for keyword in ["auth", "login", "token", "credential", "oauth"]:
            if keyword in name_lower:
                return "auth"
        
        for keyword in ["delete", "remove", "drop", "destroy"]:
            if keyword in name_lower:
                return "delete"
        
        for keyword in ["create", "write", "insert", "add", "post", "put", "update"]:
            if keyword in name_lower:
                return "write"
        
        for keyword in ["get", "read", "query", "fetch", "list", "search"]:
            if keyword in name_lower:
                return "query"
        
        for keyword in ["validate", "check", "verify", "test"]:
            if keyword in name_lower:
                return "validate"
        
        for keyword in ["sync", "commit", "push", "submit"]:
            if keyword in name_lower:
                return "sync"
        
        return "default"
    
    # =========================================================================
    # Observation-Level Uncertainty
    # =========================================================================
    
    def compute_observation_uncertainty(
        self,
        response: Any,
        expected_schema: Optional[Dict[str, Any]] = None,
        previous_observations: Optional[List[Any]] = None,
        error_indicators: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ObservationUncertainty:
        """
        Compute observation-level uncertainty for API response interpretation.
        
        Args:
            response: The API response to analyze
            expected_schema: Expected response schema for validation
            previous_observations: Previous responses for consistency check
            error_indicators: List of error patterns to check
            metadata: Additional context
        
        Returns:
            ObservationUncertainty for this observation
        """
        # Determine response type
        response_type = self._classify_response(response, error_indicators)
        
        # Schema validation probability
        schema_prob = self._validate_schema(response, expected_schema)
        
        # Semantic consistency with previous observations
        consistency = self._check_consistency(response, previous_observations)
        
        # Error signal strength (how clear is it if there's an error?)
        error_strength = self._compute_error_signal_strength(response, response_type)
        
        # Overall interpretation confidence
        interpretation_conf = (schema_prob + consistency + error_strength) / 3.0
        
        result = ObservationUncertainty(
            step_idx=self._step_idx,
            response_type=response_type,
            schema_validation_prob=schema_prob,
            semantic_consistency=consistency,
            error_signal_strength=error_strength,
            interpretation_confidence=interpretation_conf,
            metadata=metadata or {},
        )
        
        self._observation_uncertainties.append(result)
        return result
    
    def _classify_response(
        self,
        response: Any,
        error_indicators: Optional[List[str]] = None,
    ) -> str:
        """Classify response type."""
        if response is None:
            return "timeout"
        
        if isinstance(response, dict):
            # Check for error fields
            if response.get("error") or response.get("Error"):
                return "error"
            if response.get("status") in ["error", "failed", "failure"]:
                return "error"
            if response.get("partial") or response.get("incomplete"):
                return "partial"
            
        if isinstance(response, str):
            response_lower = response.lower()
            if error_indicators:
                for indicator in error_indicators:
                    if indicator.lower() in response_lower:
                        return "error"
            if "error" in response_lower or "failed" in response_lower:
                return "error"
            if "timeout" in response_lower:
                return "timeout"
        
        return "success"
    
    def _validate_schema(
        self,
        response: Any,
        expected_schema: Optional[Dict[str, Any]],
    ) -> float:
        """Compute schema validation probability."""
        if expected_schema is None:
            return 1.0  # No schema to validate against
        
        if not isinstance(response, dict):
            return 0.5  # Can't validate non-dict response
        
        required_fields = expected_schema.get("required", [])
        properties = expected_schema.get("properties", {})
        
        if not required_fields and not properties:
            return 1.0
        
        # Check required fields
        present_required = sum(1 for f in required_fields if f in response)
        required_score = present_required / len(required_fields) if required_fields else 1.0
        
        # Check property types (simplified)
        type_matches = 0
        type_checks = 0
        for prop, spec in properties.items():
            if prop in response:
                type_checks += 1
                expected_type = spec.get("type", "any")
                actual = response[prop]
                
                if expected_type == "string" and isinstance(actual, str):
                    type_matches += 1
                elif expected_type == "number" and isinstance(actual, (int, float)):
                    type_matches += 1
                elif expected_type == "boolean" and isinstance(actual, bool):
                    type_matches += 1
                elif expected_type == "array" and isinstance(actual, list):
                    type_matches += 1
                elif expected_type == "object" and isinstance(actual, dict):
                    type_matches += 1
                elif expected_type == "any":
                    type_matches += 1
        
        type_score = type_matches / type_checks if type_checks > 0 else 1.0
        
        return (required_score + type_score) / 2.0
    
    def _check_consistency(
        self,
        response: Any,
        previous: Optional[List[Any]],
    ) -> float:
        """Check semantic consistency with previous observations."""
        if previous is None or len(previous) == 0:
            return 1.0  # No history to compare
        
        # Simple consistency check based on response structure
        if not isinstance(response, dict):
            return 0.8  # Non-dict responses harder to compare
        
        # Compare keys with recent observations
        current_keys = set(response.keys()) if isinstance(response, dict) else set()
        
        consistency_scores = []
        for prev in previous[-3:]:  # Check last 3
            if isinstance(prev, dict):
                prev_keys = set(prev.keys())
                if prev_keys:
                    overlap = len(current_keys & prev_keys) / len(current_keys | prev_keys)
                    consistency_scores.append(overlap)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0
    
    def _compute_error_signal_strength(self, response: Any, response_type: str) -> float:
        """Compute how clear the error signal is."""
        if response_type == "success":
            return 1.0  # Clear success
        
        if response_type == "error":
            # Check if error is well-defined
            if isinstance(response, dict):
                if response.get("error_code") and response.get("error_message"):
                    return 1.0  # Very clear error
                if response.get("error"):
                    return 0.8  # Moderately clear
            return 0.6  # Error detected but unclear
        
        if response_type == "partial":
            return 0.5  # Partial results are ambiguous
        
        if response_type == "timeout":
            return 0.7  # Timeout is clear but cause might not be
        
        return 0.5  # Default moderate clarity
    
    # =========================================================================
    # Trajectory-Level Uncertainty
    # =========================================================================
    
    def compute_trajectory_uncertainty(
        self,
        trajectory_id: str = "default",
    ) -> TrajectoryUncertainty:
        """
        Compute trajectory-level cumulative uncertainty.
        
        Uses HMM state sequence inference and situational weights.
        
        Args:
            trajectory_id: Identifier for this trajectory
        
        Returns:
            TrajectoryUncertainty for the complete workflow
        """
        n_steps = len(self._action_uncertainties)
        
        if n_steps == 0:
            return TrajectoryUncertainty(
                trajectory_id=trajectory_id,
                total_steps=0,
                cumulative_uncertainty=0.0,
                uncertainty_trend="stable",
                critical_steps=[],
                recovery_points=[],
                hmm_state_sequence=[],
                final_confidence=1.0,
            )
        
        # Compute weighted cumulative uncertainty
        weighted_sum = 0.0
        weight_sum = 0.0
        uncertainties = []
        critical_steps = []
        
        for action_unc in self._action_uncertainties:
            u = action_unc.weighted_uncertainty
            w = action_unc.criticality_weight
            
            weighted_sum += u * w
            weight_sum += w
            uncertainties.append(u)
            
            if u > self.threshold:
                critical_steps.append(action_unc.step_idx)
        
        cumulative = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # Compute trend
        trend = self._compute_trend(uncertainties)
        
        # Identify recovery points (uncertainty drops after high uncertainty)
        recovery_points = self._find_recovery_points(uncertainties)
        
        # HMM state sequence inference
        state_sequence = self._infer_hmm_states() if self.enable_hmm else []
        
        # Final confidence (decay-weighted recent steps)
        final_conf = self._compute_final_confidence(uncertainties)
        
        return TrajectoryUncertainty(
            trajectory_id=trajectory_id,
            total_steps=n_steps,
            cumulative_uncertainty=cumulative,
            uncertainty_trend=trend,
            critical_steps=critical_steps,
            recovery_points=recovery_points,
            hmm_state_sequence=state_sequence,
            final_confidence=final_conf,
        )
    
    def _compute_trend(self, uncertainties: List[float]) -> str:
        """Compute uncertainty trend over trajectory."""
        if len(uncertainties) < 3:
            return "stable"
        
        # Simple linear regression
        n = len(uncertainties)
        x_mean = (n - 1) / 2
        y_mean = sum(uncertainties) / n
        
        numerator = sum((i - x_mean) * (u - y_mean) for i, u in enumerate(uncertainties))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _find_recovery_points(self, uncertainties: List[float]) -> List[int]:
        """Find steps where uncertainty significantly drops (recovery)."""
        recovery_points = []
        
        for i in range(1, len(uncertainties)):
            prev = uncertainties[i - 1]
            curr = uncertainties[i]
            
            # Recovery: previous was high, current is low
            if prev > self.threshold and curr < self.threshold * 0.7:
                recovery_points.append(i)
        
        return recovery_points
    
    def _infer_hmm_states(self) -> List[str]:
        """Infer HMM state sequence from action types."""
        state_sequence = []
        
        for action_unc in self._action_uncertainties:
            action_type = action_unc.action_type.lower()
            
            # Map action type to HMM state
            if action_type in ["auth", "authenticate", "login"]:
                state = "auth"
            elif action_type in ["query", "get", "read", "list"]:
                state = "query"
            elif action_type in ["validate", "check", "verify"]:
                state = "validate"
            elif action_type in ["sync", "commit", "write", "create", "update"]:
                state = "sync"
            elif action_unc.weighted_uncertainty > self.threshold:
                state = "recover"
            else:
                state = "query"  # default
            
            state_sequence.append(state)
        
        # Check for completion at end
        if state_sequence and self._action_uncertainties[-1].weighted_uncertainty < 0.1:
            state_sequence[-1] = "complete"
        
        return state_sequence
    
    def _compute_final_confidence(self, uncertainties: List[float]) -> float:
        """Compute final trajectory confidence with decay weighting."""
        if not uncertainties:
            return 1.0
        
        # Weight recent steps more heavily
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, u in enumerate(uncertainties):
            # Exponential weight favoring recent steps
            weight = self.decay_factor ** (len(uncertainties) - 1 - i)
            weighted_sum += (1.0 - u) * weight  # Convert uncertainty to confidence
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 1.0
    
    # =========================================================================
    # Full Hierarchical Analysis
    # =========================================================================
    
    def analyze_complete(
        self,
        trajectory_id: str = "default",
        logprobs: Optional[List[float]] = None,
    ) -> HierarchicalUncertaintyResult:
        """
        Perform complete hierarchical uncertainty analysis.
        
        Returns:
            HierarchicalUncertaintyResult with all levels
        """
        token_unc = self.compute_token_uncertainty(logprobs=logprobs)
        trajectory_unc = self.compute_trajectory_uncertainty(trajectory_id=trajectory_id)
        
        # Aggregate score: weighted combination of all levels
        scores = []
        weights = []
        
        if token_unc:
            # Token-level: use perplexity-based score
            token_score = min(1.0, token_unc.perplexity / 10.0)
            scores.append(1.0 - token_score)
            weights.append(0.2)
        
        if self._action_uncertainties:
            # Action-level: average confidence
            action_score = sum(a.confidence for a in self._action_uncertainties) / len(self._action_uncertainties)
            scores.append(action_score)
            weights.append(0.3)
        
        if self._observation_uncertainties:
            # Observation-level: average interpretation confidence
            obs_score = sum(o.interpretation_confidence for o in self._observation_uncertainties) / len(self._observation_uncertainties)
            scores.append(obs_score)
            weights.append(0.2)
        
        # Trajectory-level: final confidence
        scores.append(trajectory_unc.final_confidence)
        weights.append(0.3)
        
        aggregated = sum(s * w for s, w in zip(scores, weights)) / sum(weights) if weights else 1.0
        
        return HierarchicalUncertaintyResult(
            token_level=token_unc,
            action_level=list(self._action_uncertainties),
            observation_level=list(self._observation_uncertainties),
            trajectory_level=trajectory_unc,
            aggregated_score=aggregated,
            calibrated=False,
        )
    
    # =========================================================================
    # Step Management
    # =========================================================================
    
    def next_step(self) -> None:
        """Advance to next step in trajectory."""
        self._step_idx += 1
    
    def reset(self) -> None:
        """Reset propagator state for new trajectory."""
        self._action_uncertainties = []
        self._observation_uncertainties = []
        self._step_idx = 0


__all__ = [
    "HierarchicalUncertaintyPropagator",
    "HierarchicalUncertaintyResult",
    "TokenUncertainty",
    "ActionUncertainty",
    "ObservationUncertainty",
    "TrajectoryUncertainty",
    "ActionCriticality",
]

