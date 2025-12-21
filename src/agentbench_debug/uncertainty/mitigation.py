"""
Mitigation Strategies for API Orchestration.

Implements uncertainty-aware mitigation strategies:
1. Uncertainty-aware retry with exponential backoff
2. Cross-API validation
3. Circuit breaker pattern
4. Authentication resilience
5. Rate limiting intelligence

These strategies form the Decision Layer that dynamically selects
recovery approaches based on uncertainty signals and error patterns.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class MitigationStrategy(Enum):
    """Available mitigation strategies."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    CROSS_API_VALIDATION = "cross_api_validation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_SERVICE = "fallback_service"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    TOKEN_REFRESH = "token_refresh"
    RATE_LIMITER = "rate_limiter"
    CACHE_RESPONSE = "cache_response"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, rejecting requests
    HALF_OPEN = "half_open"    # Testing if service recovered


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    attempts: int
    total_delay_ms: float
    final_result: Any
    error_sequence: List[str]
    strategy_adjustments: List[str]


@dataclass
class ValidationResult:
    """Result of cross-API validation."""
    valid: bool
    consistency_score: float       # 0-1
    discrepancies: List[Dict[str, Any]]
    validated_fields: List[str]
    failed_fields: List[str]


@dataclass
class MitigationRecord:
    """Record of a mitigation action."""
    timestamp: datetime
    strategy: MitigationStrategy
    trigger_reason: str
    uncertainty_at_trigger: float
    success: bool
    duration_ms: float
    details: Dict[str, Any]


class BaseMitigationStrategy(ABC):
    """Base class for mitigation strategies."""
    
    @abstractmethod
    def should_apply(self, uncertainty: float, error_type: Optional[str] = None) -> bool:
        """Determine if this strategy should be applied."""
        pass
    
    @abstractmethod
    def apply(self, context: Dict[str, Any]) -> Any:
        """Apply the mitigation strategy."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name."""
        pass


class UncertaintyAwareRetry(BaseMitigationStrategy):
    """
    Uncertainty-aware retry strategy with adaptive backoff.
    
    Features:
    - Confidence-based retry limits
    - Adaptive delay based on uncertainty
    - Error-type aware retry decisions
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: float = 1000,
        max_delay_ms: float = 30000,
        uncertainty_threshold: float = 0.5,
        jitter_factor: float = 0.2,
    ):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay_ms: Base delay for exponential backoff
            max_delay_ms: Maximum delay cap
            uncertainty_threshold: Threshold for adjusting retry behavior
            jitter_factor: Random jitter to add to delays
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.uncertainty_threshold = uncertainty_threshold
        self.jitter_factor = jitter_factor
        
        # Non-retryable error types
        self._non_retryable = {
            "invalid_credentials",
            "permission_denied",
            "schema_validation_failed",
            "not_found",
            "invalid_argument",
        }
    
    def get_name(self) -> str:
        return "uncertainty_aware_retry"
    
    def should_apply(self, uncertainty: float, error_type: Optional[str] = None) -> bool:
        """
        Determine if retry should be attempted.
        
        High uncertainty + retryable error = should retry
        """
        if error_type and error_type in self._non_retryable:
            return False
        return True
    
    def compute_delay(
        self,
        attempt: int,
        uncertainty: float,
        retry_after: Optional[float] = None,
    ) -> float:
        """
        Compute delay for next retry.
        
        Uses exponential backoff with uncertainty adjustment.
        Higher uncertainty = slightly faster initial retries (may succeed with different approach)
        """
        if retry_after:
            return retry_after * 1000  # Convert to ms
        
        # Base exponential backoff
        delay = self.base_delay_ms * (2 ** attempt)
        
        # Uncertainty adjustment: high uncertainty -> shorter delays initially
        if uncertainty > self.uncertainty_threshold:
            uncertainty_factor = 1.0 - (uncertainty - self.uncertainty_threshold)
            delay *= max(0.5, uncertainty_factor)
        
        # Add jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor)
        delay *= (1 + jitter)
        
        # Cap at max
        delay = min(delay, self.max_delay_ms)
        
        return delay
    
    def apply(
        self,
        context: Dict[str, Any],
    ) -> RetryResult:
        """
        Apply retry strategy.
        
        Context should contain:
        - operation: Callable to retry
        - uncertainty: Current uncertainty score
        - error_type: Optional error type string
        
        Returns RetryResult
        """
        operation = context.get("operation")
        uncertainty = context.get("uncertainty", 0.0)
        
        if not callable(operation):
            return RetryResult(
                success=False,
                attempts=0,
                total_delay_ms=0,
                final_result=None,
                error_sequence=["No operation provided"],
                strategy_adjustments=[],
            )
        
        errors = []
        adjustments = []
        total_delay = 0.0
        
        # Adjust max retries based on uncertainty
        effective_max = self.max_retries
        if uncertainty > self.uncertainty_threshold:
            # Higher uncertainty = more retries allowed
            effective_max = min(self.max_retries + 2, 5)
            adjustments.append(f"Increased max_retries to {effective_max} due to high uncertainty")
        
        for attempt in range(effective_max):
            try:
                result = operation()
                return RetryResult(
                    success=True,
                    attempts=attempt + 1,
                    total_delay_ms=total_delay,
                    final_result=result,
                    error_sequence=errors,
                    strategy_adjustments=adjustments,
                )
            except Exception as e:
                error_msg = str(e)
                errors.append(f"Attempt {attempt + 1}: {error_msg}")
                
                # Check if retryable
                error_type = context.get("error_type")
                if error_type and error_type in self._non_retryable:
                    adjustments.append(f"Stopping: non-retryable error type '{error_type}'")
                    break
                
                # Compute and apply delay
                if attempt < effective_max - 1:
                    delay = self.compute_delay(attempt, uncertainty)
                    total_delay += delay
                    # In real implementation: time.sleep(delay / 1000)
        
        return RetryResult(
            success=False,
            attempts=len(errors),
            total_delay_ms=total_delay,
            final_result=None,
            error_sequence=errors,
            strategy_adjustments=adjustments,
        )


class CrossAPIValidator(BaseMitigationStrategy):
    """
    Cross-API validation for data consistency.
    
    Validates data across multiple API responses to detect inconsistencies.
    """
    
    def __init__(
        self,
        consistency_threshold: float = 0.8,
        required_fields: Optional[List[str]] = None,
    ):
        """
        Initialize validator.
        
        Args:
            consistency_threshold: Minimum score for valid consistency
            required_fields: Fields that must be consistent across APIs
        """
        self.consistency_threshold = consistency_threshold
        self.required_fields = required_fields or []
    
    def get_name(self) -> str:
        return "cross_api_validator"
    
    def should_apply(self, uncertainty: float, error_type: Optional[str] = None) -> bool:
        """Apply when uncertainty is moderate or higher."""
        return uncertainty > 0.3
    
    def apply(self, context: Dict[str, Any]) -> ValidationResult:
        """
        Validate consistency across API responses.
        
        Context should contain:
        - responses: List of (api_name, response_data) tuples
        - fields_to_check: Optional list of fields to validate
        """
        responses = context.get("responses", [])
        fields = context.get("fields_to_check", self.required_fields)
        
        if len(responses) < 2:
            return ValidationResult(
                valid=True,
                consistency_score=1.0,
                discrepancies=[],
                validated_fields=[],
                failed_fields=[],
            )
        
        discrepancies = []
        validated = []
        failed = []
        
        # Compare each field across all responses
        for field_name in fields:
            values = []
            for api_name, response in responses:
                if isinstance(response, dict):
                    value = response.get(field_name)
                    if value is not None:
                        values.append((api_name, value))
            
            if len(values) < 2:
                continue
            
            # Check consistency
            unique_values = set(str(v) for _, v in values)
            
            if len(unique_values) == 1:
                validated.append(field_name)
            else:
                failed.append(field_name)
                discrepancies.append({
                    "field": field_name,
                    "values": {api: val for api, val in values},
                    "unique_count": len(unique_values),
                })
        
        # Compute consistency score
        total_fields = len(validated) + len(failed)
        score = len(validated) / total_fields if total_fields > 0 else 1.0
        
        return ValidationResult(
            valid=score >= self.consistency_threshold,
            consistency_score=score,
            discrepancies=discrepancies,
            validated_fields=validated,
            failed_fields=failed,
        )
    
    def reconcile(
        self,
        discrepancies: List[Dict[str, Any]],
        strategy: str = "majority",
    ) -> Dict[str, Any]:
        """
        Reconcile discrepancies between API responses.
        
        Args:
            discrepancies: List of discrepancy records
            strategy: Reconciliation strategy (majority, latest, primary)
        
        Returns:
            Reconciled values
        """
        reconciled = {}
        
        for disc in discrepancies:
            field = disc["field"]
            values = disc["values"]
            
            if strategy == "majority":
                # Use most common value
                from collections import Counter
                value_counts = Counter(str(v) for v in values.values())
                most_common = value_counts.most_common(1)[0][0]
                reconciled[field] = most_common
            
            elif strategy == "latest":
                # Assume last in list is latest
                reconciled[field] = list(values.values())[-1]
            
            elif strategy == "primary":
                # Use first API's value
                reconciled[field] = list(values.values())[0]
        
        return reconciled


class CircuitBreaker(BaseMitigationStrategy):
    """
    Circuit breaker pattern for failure isolation.
    
    Prevents cascade failures by stopping requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        half_open_requests: int = 1,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout_seconds: Time before attempting recovery
            half_open_requests: Requests allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.half_open_requests = half_open_requests
        
        # State per service
        self._states: Dict[str, CircuitState] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_time: Dict[str, datetime] = {}
        self._half_open_attempts: Dict[str, int] = {}
    
    def get_name(self) -> str:
        return "circuit_breaker"
    
    def should_apply(self, uncertainty: float, error_type: Optional[str] = None) -> bool:
        """Always apply circuit breaker logic."""
        return True
    
    def get_state(self, service_id: str) -> CircuitState:
        """Get current state for a service."""
        if service_id not in self._states:
            self._states[service_id] = CircuitState.CLOSED
            self._failure_counts[service_id] = 0
            self._half_open_attempts[service_id] = 0
        
        state = self._states[service_id]
        
        # Check if should transition from OPEN to HALF_OPEN
        if state == CircuitState.OPEN:
            last_failure = self._last_failure_time.get(service_id)
            if last_failure and datetime.now() - last_failure >= self.recovery_timeout:
                self._states[service_id] = CircuitState.HALF_OPEN
                self._half_open_attempts[service_id] = 0
                return CircuitState.HALF_OPEN
        
        return self._states[service_id]
    
    def is_allowed(self, service_id: str) -> bool:
        """Check if request to service is allowed."""
        state = self.get_state(service_id)
        
        if state == CircuitState.CLOSED:
            return True
        
        if state == CircuitState.HALF_OPEN:
            # Allow limited requests in half-open
            if self._half_open_attempts[service_id] < self.half_open_requests:
                return True
            return False
        
        # OPEN state
        return False
    
    def record_success(self, service_id: str) -> None:
        """Record successful request."""
        state = self.get_state(service_id)
        
        if state == CircuitState.HALF_OPEN:
            # Success in half-open -> close circuit
            self._states[service_id] = CircuitState.CLOSED
            self._failure_counts[service_id] = 0
        
        elif state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_counts[service_id] = 0
    
    def record_failure(self, service_id: str) -> None:
        """Record failed request."""
        state = self.get_state(service_id)
        
        if state == CircuitState.HALF_OPEN:
            # Failure in half-open -> back to open
            self._states[service_id] = CircuitState.OPEN
            self._last_failure_time[service_id] = datetime.now()
        
        elif state == CircuitState.CLOSED:
            self._failure_counts[service_id] = self._failure_counts.get(service_id, 0) + 1
            
            if self._failure_counts[service_id] >= self.failure_threshold:
                # Open circuit
                self._states[service_id] = CircuitState.OPEN
                self._last_failure_time[service_id] = datetime.now()
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply circuit breaker logic.
        
        Context should contain:
        - service_id: Service identifier
        - operation: Callable to execute
        
        Returns dict with success status and result/error
        """
        service_id = context.get("service_id", "default")
        operation = context.get("operation")
        
        if not self.is_allowed(service_id):
            return {
                "success": False,
                "blocked": True,
                "circuit_state": self._states[service_id].value,
                "error": "Circuit breaker is open",
            }
        
        if not callable(operation):
            return {
                "success": False,
                "blocked": False,
                "error": "No operation provided",
            }
        
        try:
            result = operation()
            self.record_success(service_id)
            return {
                "success": True,
                "blocked": False,
                "result": result,
                "circuit_state": self._states[service_id].value,
            }
        except Exception as e:
            self.record_failure(service_id)
            return {
                "success": False,
                "blocked": False,
                "error": str(e),
                "circuit_state": self._states[service_id].value,
            }
    
    def reset(self, service_id: Optional[str] = None) -> None:
        """Reset circuit breaker state."""
        if service_id:
            self._states[service_id] = CircuitState.CLOSED
            self._failure_counts[service_id] = 0
        else:
            self._states.clear()
            self._failure_counts.clear()
            self._last_failure_time.clear()
            self._half_open_attempts.clear()


class RateLimiter(BaseMitigationStrategy):
    """
    Uncertainty-aware rate limiter.
    
    Manages request rates with adaptive pacing based on uncertainty.
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20,
        uncertainty_slowdown_factor: float = 0.5,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Base rate limit
            burst_size: Maximum burst size
            uncertainty_slowdown_factor: Factor to reduce rate when uncertainty is high
        """
        self.base_rps = requests_per_second
        self.burst_size = burst_size
        self.uncertainty_slowdown = uncertainty_slowdown_factor
        
        # Token bucket state
        self._tokens = burst_size
        self._last_update = datetime.now()
    
    def get_name(self) -> str:
        return "rate_limiter"
    
    def should_apply(self, uncertainty: float, error_type: Optional[str] = None) -> bool:
        """Always apply rate limiting."""
        return True
    
    def _refill_tokens(self, uncertainty: float) -> None:
        """Refill tokens based on elapsed time and uncertainty."""
        now = datetime.now()
        elapsed = (now - self._last_update).total_seconds()
        self._last_update = now
        
        # Adjust rate based on uncertainty
        effective_rps = self.base_rps
        if uncertainty > 0.5:
            effective_rps *= self.uncertainty_slowdown
        
        # Add tokens
        new_tokens = elapsed * effective_rps
        self._tokens = min(self.burst_size, self._tokens + new_tokens)
    
    def acquire(self, uncertainty: float = 0.0) -> Tuple[bool, float]:
        """
        Try to acquire a token.
        
        Returns:
            (success, wait_time_seconds)
        """
        self._refill_tokens(uncertainty)
        
        if self._tokens >= 1:
            self._tokens -= 1
            return (True, 0.0)
        
        # Calculate wait time
        effective_rps = self.base_rps
        if uncertainty > 0.5:
            effective_rps *= self.uncertainty_slowdown
        
        wait_time = (1 - self._tokens) / effective_rps
        return (False, wait_time)
    
    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply rate limiting.
        
        Context should contain:
        - uncertainty: Current uncertainty score
        
        Returns dict with allowed status and wait time
        """
        uncertainty = context.get("uncertainty", 0.0)
        allowed, wait_time = self.acquire(uncertainty)
        
        return {
            "allowed": allowed,
            "wait_time_seconds": wait_time,
            "current_tokens": self._tokens,
            "effective_rps": self.base_rps * (
                self.uncertainty_slowdown if uncertainty > 0.5 else 1.0
            ),
        }


class DecisionLayer:
    """
    Decision layer for selecting mitigation strategies.
    
    Dynamically chooses the most appropriate recovery strategy
    based on uncertainty signals, error patterns, and past outcomes.
    """
    
    def __init__(self):
        """Initialize decision layer with default strategies."""
        self.retry_strategy = UncertaintyAwareRetry()
        self.validator = CrossAPIValidator()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        
        # History for learning
        self._decision_history: List[MitigationRecord] = []
        self._strategy_success_rates: Dict[str, Tuple[int, int]] = {}  # (successes, total)
    
    def select_strategy(
        self,
        uncertainty: float,
        error_type: Optional[str] = None,
        error_category: Optional[str] = None,
        previous_strategies: Optional[List[str]] = None,
    ) -> List[MitigationStrategy]:
        """
        Select appropriate mitigation strategies.
        
        Args:
            uncertainty: Current uncertainty score
            error_type: Specific error type
            error_category: Error category
            previous_strategies: Strategies already tried
        
        Returns:
            Ordered list of recommended strategies
        """
        strategies = []
        previous = set(previous_strategies or [])
        
        # Authentication errors -> token refresh
        if error_category == "authentication":
            if MitigationStrategy.TOKEN_REFRESH.value not in previous:
                strategies.append(MitigationStrategy.TOKEN_REFRESH)
        
        # Rate limiting -> rate limiter + backoff
        if error_category == "rate_limiting":
            strategies.append(MitigationStrategy.RATE_LIMITER)
            if MitigationStrategy.RETRY_WITH_BACKOFF.value not in previous:
                strategies.append(MitigationStrategy.RETRY_WITH_BACKOFF)
        
        # High uncertainty -> validation + careful retry
        if uncertainty > 0.5:
            strategies.append(MitigationStrategy.CROSS_API_VALIDATION)
            if len(previous) < 2:
                strategies.append(MitigationStrategy.RETRY_WITH_BACKOFF)
        
        # Repeated failures -> circuit breaker + fallback
        if len(previous) >= 2:
            strategies.append(MitigationStrategy.CIRCUIT_BREAKER)
            strategies.append(MitigationStrategy.FALLBACK_SERVICE)
        
        # If many retries failed -> graceful degradation
        if len(previous) >= 3:
            strategies.append(MitigationStrategy.GRACEFUL_DEGRADATION)
        
        # Default: simple retry for transient errors
        if not strategies and error_type not in {
            "invalid_credentials",
            "permission_denied",
            "schema_validation_failed",
        }:
            strategies.append(MitigationStrategy.RETRY_WITH_BACKOFF)
        
        return strategies
    
    def execute_strategy(
        self,
        strategy: MitigationStrategy,
        context: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """
        Execute a mitigation strategy.
        
        Args:
            strategy: Strategy to execute
            context: Execution context
        
        Returns:
            (success, result)
        """
        start_time = datetime.now()
        uncertainty = context.get("uncertainty", 0.0)
        trigger_reason = context.get("trigger_reason", "unknown")
        
        success = False
        result = None
        
        try:
            if strategy == MitigationStrategy.RETRY_WITH_BACKOFF:
                retry_result = self.retry_strategy.apply(context)
                success = retry_result.success
                result = retry_result
            
            elif strategy == MitigationStrategy.CROSS_API_VALIDATION:
                validation = self.validator.apply(context)
                success = validation.valid
                result = validation
            
            elif strategy == MitigationStrategy.CIRCUIT_BREAKER:
                cb_result = self.circuit_breaker.apply(context)
                success = cb_result.get("success", False)
                result = cb_result
            
            elif strategy == MitigationStrategy.RATE_LIMITER:
                rl_result = self.rate_limiter.apply(context)
                success = rl_result.get("allowed", False)
                result = rl_result
            
            elif strategy == MitigationStrategy.FALLBACK_SERVICE:
                # Would call fallback in real implementation
                fallback = context.get("fallback_operation")
                if callable(fallback):
                    result = fallback()
                    success = True
                else:
                    success = False
                    result = {"error": "No fallback configured"}
            
            elif strategy == MitigationStrategy.GRACEFUL_DEGRADATION:
                # Return partial/cached result
                cached = context.get("cached_result")
                if cached is not None:
                    success = True
                    result = {"partial": True, "cached_result": cached}
                else:
                    success = False
                    result = {"error": "No cached result available"}
            
            else:
                result = {"error": f"Unknown strategy: {strategy}"}
        
        except Exception as e:
            result = {"error": str(e)}
        
        # Record decision
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        record = MitigationRecord(
            timestamp=start_time,
            strategy=strategy,
            trigger_reason=trigger_reason,
            uncertainty_at_trigger=uncertainty,
            success=success,
            duration_ms=duration_ms,
            details={"result_summary": str(result)[:200]},
        )
        self._decision_history.append(record)
        
        # Update success rates
        key = strategy.value
        current = self._strategy_success_rates.get(key, (0, 0))
        self._strategy_success_rates[key] = (
            current[0] + (1 if success else 0),
            current[1] + 1,
        )
        
        return success, result
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get success rate statistics for each strategy."""
        stats = {}
        
        for strategy, (successes, total) in self._strategy_success_rates.items():
            stats[strategy] = {
                "attempts": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0.0,
            }
        
        return stats
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Export decision history."""
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "strategy": r.strategy.value,
                "trigger_reason": r.trigger_reason,
                "uncertainty": r.uncertainty_at_trigger,
                "success": r.success,
                "duration_ms": r.duration_ms,
            }
            for r in self._decision_history
        ]
    
    def reset(self) -> None:
        """Reset all state."""
        self._decision_history = []
        self._strategy_success_rates = {}
        self.circuit_breaker.reset()


__all__ = [
    "MitigationStrategy",
    "UncertaintyAwareRetry",
    "CrossAPIValidator",
    "CircuitBreaker",
    "RateLimiter",
    "DecisionLayer",
    "RetryResult",
    "ValidationResult",
    "CircuitState",
    "MitigationRecord",
]

