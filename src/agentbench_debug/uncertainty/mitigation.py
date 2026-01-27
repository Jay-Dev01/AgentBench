"""Mitigation strategies for high-uncertainty situations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MitigationStrategy(Enum):
    """Available mitigation strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    VALIDATION = "validation"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMIT = "rate_limit"
    SKIP = "skip"


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


class RetryStrategy:
    """Retry with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def execute(
        self,
        func: Callable[[], Any],
        should_retry: Optional[Callable[[Exception], bool]] = None,
    ) -> Any:
        """
        Execute function with retries.
        
        Args:
            func: Function to execute
            should_retry: Optional predicate to determine if retry is appropriate
        
        Returns:
            Result of successful execution
        
        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if should_retry and not should_retry(e):
                    raise
                
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.base_delay * (self.config.exponential_base ** attempt),
                        self.config.max_delay,
                    )
                    time.sleep(delay)
        
        raise last_exception


class CircuitBreaker:
    """
    Circuit breaker pattern for failing fast.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing fast, no calls allowed
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    return True
            return False
        
        # HALF_OPEN state
        return self.half_open_calls < self.config.half_open_max_calls
    
    def record_success(self) -> None:
        """Record a successful execution."""
        if self.state == "HALF_OPEN":
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
        elif self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"
    
    def execute(self, func: Callable[[], Any]) -> Any:
        """Execute with circuit breaker protection."""
        if not self.can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker is {self.state}, not allowing execution"
            )
        
        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise


class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open."""
    pass


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(
        self,
        requests_per_second: float = 1.0,
        burst_size: int = 1,
    ):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.time()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a token, blocking if necessary.
        
        Returns True if token acquired, False if timeout.
        """
        start_time = time.time()
        
        while True:
            # Replenish tokens
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            
            # Check timeout
            if timeout is not None:
                if now - start_time >= timeout:
                    return False
            
            # Wait a bit
            wait_time = (1.0 - self.tokens) / self.rate
            time.sleep(min(wait_time, 0.1))


@dataclass
class MitigationDecision:
    """Decision from the mitigation layer."""
    strategy: MitigationStrategy
    should_proceed: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionLayer:
    """
    Decision layer that selects appropriate mitigation strategies.
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.retry_strategy = RetryStrategy(retry_config)
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
        self.rate_limiter = RateLimiter()
    
    def decide(
        self,
        confidence: float,
        error_count: int = 0,
        action_type: str = "unknown",
    ) -> MitigationDecision:
        """
        Decide on mitigation strategy based on current state.
        
        Args:
            confidence: Current confidence level
            error_count: Number of recent errors
            action_type: Type of action being performed
        
        Returns:
            MitigationDecision with recommended strategy
        """
        uncertainty = 1.0 - confidence
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return MitigationDecision(
                strategy=MitigationStrategy.CIRCUIT_BREAKER,
                should_proceed=False,
                reason="Circuit breaker open",
                metadata={"state": self.circuit_breaker.state},
            )
        
        # High uncertainty on critical actions
        if uncertainty > self.uncertainty_threshold and action_type in ["submit", "shell_command"]:
            return MitigationDecision(
                strategy=MitigationStrategy.VALIDATION,
                should_proceed=True,
                reason="High uncertainty on critical action - extra validation recommended",
                metadata={"uncertainty": uncertainty, "action_type": action_type},
            )
        
        # Multiple errors suggest retry
        if error_count > 0:
            return MitigationDecision(
                strategy=MitigationStrategy.RETRY,
                should_proceed=True,
                reason="Recent errors detected",
                metadata={"error_count": error_count},
            )
        
        # Normal operation
        return MitigationDecision(
            strategy=MitigationStrategy.SKIP,
            should_proceed=True,
            reason="Normal operation",
            metadata={"uncertainty": uncertainty},
        )


__all__ = [
    "MitigationStrategy",
    "RetryConfig",
    "CircuitBreakerConfig",
    "RetryStrategy",
    "CircuitBreaker",
    "CircuitBreakerOpenException",
    "RateLimiter",
    "DecisionLayer",
    "MitigationDecision",
]

