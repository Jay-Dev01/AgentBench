"""
API Orchestration Error Taxonomy.

Implements the error typology from API-ORCHA-Bench:
1. Authentication failures
2. Rate limiting violations
3. Partial workflow failures
4. Data consistency errors
5. Coordination failures

Also includes error logging, attribution, and recovery guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ErrorCategory(Enum):
    """Main error categories for API orchestration."""
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    PARTIAL_FAILURE = "partial_failure"
    DATA_CONSISTENCY = "data_consistency"
    COORDINATION = "coordination"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class AuthenticationErrorType(Enum):
    """Authentication error subtypes."""
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_REFRESH_FAILED = "token_refresh_failed"
    OAUTH_FLOW_VIOLATION = "oauth_flow_violation"
    MFA_REQUIRED = "mfa_required"
    PERMISSION_DENIED = "permission_denied"
    SERVICE_AUTH_FAILURE = "service_to_service_auth_failure"


class RateLimitErrorType(Enum):
    """Rate limiting error subtypes."""
    QUOTA_EXCEEDED = "quota_exceeded"
    BURST_LIMIT = "burst_limit"
    IMPROPER_BACKOFF = "improper_backoff"
    CONCURRENT_LIMIT = "concurrent_limit"
    DAILY_LIMIT = "daily_limit"
    PER_USER_LIMIT = "per_user_limit"


class PartialFailureErrorType(Enum):
    """Partial workflow failure subtypes."""
    INCOMPLETE_RECOVERY = "incomplete_recovery"
    INCONSISTENT_RETRY = "inconsistent_retry"
    CASCADE_FAILURE = "cascade_failure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    TIMEOUT = "timeout"
    PARTIAL_RESPONSE = "partial_response"


class DataConsistencyErrorType(Enum):
    """Data consistency error subtypes."""
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    IDEMPOTENCY_VIOLATION = "idempotency_violation"
    STALE_DATA = "stale_data"
    CONFLICT = "conflict"
    INTEGRITY_VIOLATION = "integrity_violation"
    CROSS_API_MISMATCH = "cross_api_mismatch"


class CoordinationErrorType(Enum):
    """Coordination failure subtypes."""
    SEQUENCE_VIOLATION = "sequence_violation"
    RACE_CONDITION = "race_condition"
    DEPENDENCY_ERROR = "dependency_error"
    AGGREGATION_ERROR = "aggregation_error"
    DEADLOCK = "deadlock"
    SYNC_FAILURE = "sync_failure"


@dataclass
class ErrorInstance:
    """A single error instance with full context."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    error_type: str                    # Specific error type within category
    step_idx: int
    action_name: str
    api_endpoint: Optional[str]
    
    # Error details
    message: str
    raw_response: Optional[str]
    http_status: Optional[int]
    
    # Context
    preceding_actions: List[str]       # Actions before this error
    uncertainty_at_error: float        # Uncertainty score when error occurred
    
    # Attribution
    root_cause: Optional[str]
    cascading_from: Optional[str]      # Error ID if this cascaded from another
    
    # Recovery
    recoverable: bool
    recovery_attempted: bool
    recovery_successful: bool
    recovery_strategy_used: Optional[str]
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorSummary:
    """Aggregated error summary for a workflow run."""
    total_errors: int
    errors_by_category: Dict[str, int]
    errors_by_type: Dict[str, int]
    critical_errors: List[str]         # Error IDs of critical errors
    recovered_errors: List[str]        # Error IDs of successfully recovered errors
    mean_uncertainty_at_error: float
    error_rate: float                  # errors / total_steps
    recovery_rate: float               # recovered / total_errors


class ErrorTaxonomy:
    """
    API orchestration error taxonomy with logging and analysis.
    
    Provides:
    - Structured error classification
    - Error logging with full context
    - Attribution analysis
    - Recovery guidance
    - Aggregated statistics
    """
    
    def __init__(self):
        """Initialize error taxonomy."""
        self._errors: List[ErrorInstance] = []
        self._error_counter = 0
        self._total_steps = 0
        
        # Error classification rules
        self._classification_rules = self._build_classification_rules()
        
        # Recovery strategies per error type
        self._recovery_strategies = self._build_recovery_strategies()
    
    def _build_classification_rules(self) -> Dict[str, Tuple[ErrorCategory, str]]:
        """Build error classification rules from keywords/patterns."""
        return {
            # Authentication
            "401": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.PERMISSION_DENIED.value),
            "403": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.PERMISSION_DENIED.value),
            "unauthorized": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.INVALID_CREDENTIALS.value),
            "token expired": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.TOKEN_EXPIRED.value),
            "token_expired": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.TOKEN_EXPIRED.value),
            "invalid token": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.INVALID_CREDENTIALS.value),
            "oauth": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.OAUTH_FLOW_VIOLATION.value),
            "mfa required": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.MFA_REQUIRED.value),
            "2fa": (ErrorCategory.AUTHENTICATION, AuthenticationErrorType.MFA_REQUIRED.value),
            
            # Rate limiting
            "429": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.QUOTA_EXCEEDED.value),
            "rate limit": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.QUOTA_EXCEEDED.value),
            "too many requests": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.QUOTA_EXCEEDED.value),
            "quota exceeded": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.QUOTA_EXCEEDED.value),
            "throttl": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.QUOTA_EXCEEDED.value),
            "burst": (ErrorCategory.RATE_LIMITING, RateLimitErrorType.BURST_LIMIT.value),
            
            # Partial failures
            "timeout": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.TIMEOUT.value),
            "timed out": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.TIMEOUT.value),
            "504": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.TIMEOUT.value),
            "partial": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.PARTIAL_RESPONSE.value),
            "incomplete": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.PARTIAL_RESPONSE.value),
            "circuit": (ErrorCategory.PARTIAL_FAILURE, PartialFailureErrorType.CIRCUIT_BREAKER_OPEN.value),
            
            # Data consistency
            "schema": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.SCHEMA_VALIDATION_FAILED.value),
            "validation": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.SCHEMA_VALIDATION_FAILED.value),
            "invalid format": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.SCHEMA_VALIDATION_FAILED.value),
            "conflict": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.CONFLICT.value),
            "409": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.CONFLICT.value),
            "stale": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.STALE_DATA.value),
            "idempotency": (ErrorCategory.DATA_CONSISTENCY, DataConsistencyErrorType.IDEMPOTENCY_VIOLATION.value),
            
            # Coordination
            "sequence": (ErrorCategory.COORDINATION, CoordinationErrorType.SEQUENCE_VIOLATION.value),
            "order": (ErrorCategory.COORDINATION, CoordinationErrorType.SEQUENCE_VIOLATION.value),
            "dependency": (ErrorCategory.COORDINATION, CoordinationErrorType.DEPENDENCY_ERROR.value),
            "not found": (ErrorCategory.COORDINATION, CoordinationErrorType.DEPENDENCY_ERROR.value),
            "404": (ErrorCategory.COORDINATION, CoordinationErrorType.DEPENDENCY_ERROR.value),
            "race": (ErrorCategory.COORDINATION, CoordinationErrorType.RACE_CONDITION.value),
            "deadlock": (ErrorCategory.COORDINATION, CoordinationErrorType.DEADLOCK.value),
            
            # System
            "500": (ErrorCategory.SYSTEM, "internal_server_error"),
            "502": (ErrorCategory.SYSTEM, "bad_gateway"),
            "503": (ErrorCategory.SYSTEM, "service_unavailable"),
            "connection": (ErrorCategory.SYSTEM, "connection_error"),
            "network": (ErrorCategory.SYSTEM, "network_error"),
        }
    
    def _build_recovery_strategies(self) -> Dict[str, List[str]]:
        """Build recovery strategy recommendations per error type."""
        return {
            # Authentication
            AuthenticationErrorType.TOKEN_EXPIRED.value: [
                "refresh_token",
                "re_authenticate",
                "use_cached_credentials",
            ],
            AuthenticationErrorType.INVALID_CREDENTIALS.value: [
                "verify_credentials",
                "prompt_user",
                "use_fallback_auth",
            ],
            AuthenticationErrorType.PERMISSION_DENIED.value: [
                "check_permissions",
                "request_elevated_access",
                "use_alternative_endpoint",
            ],
            
            # Rate limiting
            RateLimitErrorType.QUOTA_EXCEEDED.value: [
                "exponential_backoff",
                "respect_retry_after",
                "distribute_requests",
            ],
            RateLimitErrorType.BURST_LIMIT.value: [
                "throttle_requests",
                "batch_operations",
                "use_rate_limiter",
            ],
            
            # Partial failures
            PartialFailureErrorType.TIMEOUT.value: [
                "retry_with_backoff",
                "increase_timeout",
                "use_async_pattern",
            ],
            PartialFailureErrorType.PARTIAL_RESPONSE.value: [
                "request_remaining",
                "merge_results",
                "validate_completeness",
            ],
            PartialFailureErrorType.CASCADE_FAILURE.value: [
                "circuit_breaker",
                "fallback_service",
                "graceful_degradation",
            ],
            
            # Data consistency
            DataConsistencyErrorType.SCHEMA_VALIDATION_FAILED.value: [
                "validate_before_send",
                "transform_data",
                "use_schema_version",
            ],
            DataConsistencyErrorType.CONFLICT.value: [
                "fetch_latest",
                "merge_changes",
                "use_conditional_update",
            ],
            DataConsistencyErrorType.STALE_DATA.value: [
                "refresh_cache",
                "use_etag",
                "poll_for_update",
            ],
            
            # Coordination
            CoordinationErrorType.SEQUENCE_VIOLATION.value: [
                "reorder_operations",
                "add_dependency_check",
                "use_transaction",
            ],
            CoordinationErrorType.DEPENDENCY_ERROR.value: [
                "verify_prerequisites",
                "create_missing_resource",
                "use_fallback",
            ],
            CoordinationErrorType.RACE_CONDITION.value: [
                "use_locking",
                "retry_with_check",
                "use_optimistic_concurrency",
            ],
        }
    
    def classify_error(
        self,
        message: str,
        http_status: Optional[int] = None,
        raw_response: Optional[str] = None,
    ) -> Tuple[ErrorCategory, str]:
        """
        Classify an error based on message and context.
        
        Args:
            message: Error message
            http_status: HTTP status code if available
            raw_response: Raw API response if available
        
        Returns:
            (ErrorCategory, error_type) tuple
        """
        # Combine all text for matching
        text = f"{message} {raw_response or ''}".lower()
        
        # Check HTTP status first
        if http_status:
            status_str = str(http_status)
            if status_str in self._classification_rules:
                return self._classification_rules[status_str]
        
        # Check message patterns
        for pattern, (category, error_type) in self._classification_rules.items():
            if pattern.lower() in text:
                return (category, error_type)
        
        # Default
        return (ErrorCategory.UNKNOWN, "unknown_error")
    
    def log_error(
        self,
        step_idx: int,
        action_name: str,
        message: str,
        http_status: Optional[int] = None,
        raw_response: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        preceding_actions: Optional[List[str]] = None,
        uncertainty: float = 0.0,
        cascading_from: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ErrorInstance:
        """
        Log an error with full context and automatic classification.
        
        Returns:
            The created ErrorInstance
        """
        self._error_counter += 1
        error_id = f"err_{self._error_counter:04d}"
        
        # Classify error
        category, error_type = self.classify_error(message, http_status, raw_response)
        
        # Get recovery strategies
        recovery_strategies = self._recovery_strategies.get(error_type, [])
        recoverable = len(recovery_strategies) > 0
        
        error = ErrorInstance(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            error_type=error_type,
            step_idx=step_idx,
            action_name=action_name,
            api_endpoint=api_endpoint,
            message=message,
            raw_response=raw_response,
            http_status=http_status,
            preceding_actions=preceding_actions or [],
            uncertainty_at_error=uncertainty,
            root_cause=None,
            cascading_from=cascading_from,
            recoverable=recoverable,
            recovery_attempted=False,
            recovery_successful=False,
            recovery_strategy_used=None,
            metadata=metadata or {},
        )
        
        self._errors.append(error)
        return error
    
    def mark_step(self) -> None:
        """Mark that a step was executed (for error rate calculation)."""
        self._total_steps += 1
    
    def mark_recovery(
        self,
        error_id: str,
        strategy_used: str,
        successful: bool,
    ) -> None:
        """
        Mark that a recovery was attempted for an error.
        
        Args:
            error_id: The error ID
            strategy_used: Name of recovery strategy used
            successful: Whether recovery succeeded
        """
        for error in self._errors:
            if error.error_id == error_id:
                error.recovery_attempted = True
                error.recovery_strategy_used = strategy_used
                error.recovery_successful = successful
                break
    
    def get_recovery_strategies(self, error_type: str) -> List[str]:
        """Get recommended recovery strategies for an error type."""
        return self._recovery_strategies.get(error_type, [])
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInstance]:
        """Get all errors of a specific category."""
        return [e for e in self._errors if e.category == category]
    
    def get_cascade_chain(self, error_id: str) -> List[ErrorInstance]:
        """Get the chain of cascading errors starting from an error."""
        chain = []
        current_id = error_id
        
        while current_id:
            for error in self._errors:
                if error.error_id == current_id:
                    chain.append(error)
                    current_id = error.cascading_from
                    break
            else:
                break
        
        return list(reversed(chain))
    
    def compute_summary(self) -> ErrorSummary:
        """Compute aggregated error summary."""
        if not self._errors:
            return ErrorSummary(
                total_errors=0,
                errors_by_category={},
                errors_by_type={},
                critical_errors=[],
                recovered_errors=[],
                mean_uncertainty_at_error=0.0,
                error_rate=0.0,
                recovery_rate=0.0,
            )
        
        # Count by category
        by_category: Dict[str, int] = {}
        for error in self._errors:
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # Count by type
        by_type: Dict[str, int] = {}
        for error in self._errors:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
        
        # Critical errors (non-recoverable or authentication)
        critical = [
            e.error_id for e in self._errors
            if not e.recoverable or e.category == ErrorCategory.AUTHENTICATION
        ]
        
        # Recovered errors
        recovered = [
            e.error_id for e in self._errors
            if e.recovery_successful
        ]
        
        # Mean uncertainty at error
        uncertainties = [e.uncertainty_at_error for e in self._errors]
        mean_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        
        # Error rate
        error_rate = len(self._errors) / self._total_steps if self._total_steps > 0 else 0.0
        
        # Recovery rate
        attempted = [e for e in self._errors if e.recovery_attempted]
        recovery_rate = len(recovered) / len(attempted) if attempted else 0.0
        
        return ErrorSummary(
            total_errors=len(self._errors),
            errors_by_category=by_category,
            errors_by_type=by_type,
            critical_errors=critical,
            recovered_errors=recovered,
            mean_uncertainty_at_error=mean_unc,
            error_rate=error_rate,
            recovery_rate=recovery_rate,
        )
    
    def get_error_attribution(self, error_id: str) -> Dict[str, Any]:
        """
        Get detailed attribution analysis for an error.
        
        Returns dict with root cause analysis, contributing factors, and suggestions.
        """
        error = None
        for e in self._errors:
            if e.error_id == error_id:
                error = e
                break
        
        if not error:
            return {"error": "Error not found"}
        
        # Analyze cascade chain
        chain = self.get_cascade_chain(error_id)
        root_error = chain[0] if chain else error
        
        # Identify contributing factors
        factors = []
        
        if error.uncertainty_at_error > 0.5:
            factors.append("High uncertainty before error (>0.5)")
        
        if len(error.preceding_actions) > 5:
            factors.append("Long action sequence before error")
        
        if error.cascading_from:
            factors.append(f"Cascaded from previous error: {error.cascading_from}")
        
        if error.category == ErrorCategory.COORDINATION:
            factors.append("API coordination/ordering issue")
        
        # Get recovery recommendations
        strategies = self.get_recovery_strategies(error.error_type)
        
        return {
            "error_id": error_id,
            "category": error.category.value,
            "error_type": error.error_type,
            "root_cause_error": root_error.error_id if chain else error_id,
            "cascade_length": len(chain),
            "contributing_factors": factors,
            "recovery_strategies": strategies,
            "recoverable": error.recoverable,
            "uncertainty_at_error": error.uncertainty_at_error,
            "message": error.message,
        }
    
    def reset(self) -> None:
        """Reset all error tracking."""
        self._errors = []
        self._error_counter = 0
        self._total_steps = 0
    
    def export_errors(self) -> List[Dict[str, Any]]:
        """Export all errors as dictionaries."""
        return [
            {
                "error_id": e.error_id,
                "timestamp": e.timestamp.isoformat(),
                "category": e.category.value,
                "error_type": e.error_type,
                "step_idx": e.step_idx,
                "action_name": e.action_name,
                "api_endpoint": e.api_endpoint,
                "message": e.message,
                "http_status": e.http_status,
                "uncertainty_at_error": e.uncertainty_at_error,
                "recoverable": e.recoverable,
                "recovery_attempted": e.recovery_attempted,
                "recovery_successful": e.recovery_successful,
                "recovery_strategy_used": e.recovery_strategy_used,
                "cascading_from": e.cascading_from,
                "metadata": e.metadata,
            }
            for e in self._errors
        ]


__all__ = [
    "ErrorTaxonomy",
    "ErrorCategory",
    "ErrorInstance",
    "ErrorSummary",
    "AuthenticationErrorType",
    "RateLimitErrorType",
    "PartialFailureErrorType",
    "DataConsistencyErrorType",
    "CoordinationErrorType",
]

