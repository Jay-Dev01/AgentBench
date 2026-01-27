"""Error taxonomy for API orchestration failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(Enum):
    """Categories of API orchestration errors."""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    SERVICE_UNAVAILABLE = "service_unavailable"
    CONTEXT_LIMIT = "context_limit"
    CONTENT_FILTER = "content_filter"
    NETWORK = "network"
    PARSE_ERROR = "parse_error"
    TASK_ERROR = "task_error"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"  # Recoverable, minor impact
    MEDIUM = "medium"  # May affect task, needs attention
    HIGH = "high"  # Likely task failure
    CRITICAL = "critical"  # Immediate task termination


@dataclass
class ErrorRecord:
    """Record of a single error occurrence."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    step_idx: int
    timestamp: datetime = field(default_factory=datetime.now)
    raw_error: Optional[str] = None
    recoverable: bool = True
    recovery_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "step_idx": self.step_idx,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "recovery_action": self.recovery_action,
        }


class ErrorTaxonomy:
    """
    Classify and track API orchestration errors.
    
    Provides:
    - Error classification from raw error messages
    - Error logging and statistics
    - Recovery recommendations
    """
    
    # Patterns for error classification
    ERROR_PATTERNS = {
        ErrorCategory.RATE_LIMIT: [
            "rate limit",
            "too many requests",
            "429",
            "throttl",
            "quota exceeded",
        ],
        ErrorCategory.AUTHENTICATION: [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "api key",
            "authentication",
            "invalid key",
        ],
        ErrorCategory.TIMEOUT: [
            "timeout",
            "timed out",
            "deadline exceeded",
            "504",
        ],
        ErrorCategory.INVALID_REQUEST: [
            "400",
            "bad request",
            "invalid",
            "malformed",
        ],
        ErrorCategory.SERVICE_UNAVAILABLE: [
            "503",
            "502",
            "service unavailable",
            "server error",
            "500",
        ],
        ErrorCategory.CONTEXT_LIMIT: [
            "context",
            "token limit",
            "too long",
            "max tokens",
            "context length",
        ],
        ErrorCategory.CONTENT_FILTER: [
            "content filter",
            "content policy",
            "flagged",
            "inappropriate",
        ],
        ErrorCategory.NETWORK: [
            "connection",
            "network",
            "dns",
            "ssl",
            "certificate",
        ],
        ErrorCategory.PARSE_ERROR: [
            "json",
            "parse",
            "decode",
            "syntax",
        ],
    }
    
    # Recovery recommendations
    RECOVERY_ACTIONS = {
        ErrorCategory.RATE_LIMIT: "Wait and retry with exponential backoff",
        ErrorCategory.AUTHENTICATION: "Check API key and permissions",
        ErrorCategory.TIMEOUT: "Retry with increased timeout or simpler request",
        ErrorCategory.INVALID_REQUEST: "Validate request format and parameters",
        ErrorCategory.SERVICE_UNAVAILABLE: "Wait and retry, check service status",
        ErrorCategory.CONTEXT_LIMIT: "Truncate context or split into smaller requests",
        ErrorCategory.CONTENT_FILTER: "Rephrase request to avoid flagged content",
        ErrorCategory.NETWORK: "Check network connectivity, retry",
        ErrorCategory.PARSE_ERROR: "Check response format, handle malformed data",
        ErrorCategory.TASK_ERROR: "Review task requirements and agent output",
        ErrorCategory.UNKNOWN: "Log details for investigation",
    }
    
    def __init__(self):
        self.errors: List[ErrorRecord] = []
        self.error_counts: Dict[ErrorCategory, int] = {
            cat: 0 for cat in ErrorCategory
        }
    
    def classify(self, error_message: str) -> ErrorCategory:
        """Classify an error message into a category."""
        if not error_message:
            return ErrorCategory.UNKNOWN
        
        error_lower = error_message.lower()
        
        for category, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return category
        
        return ErrorCategory.UNKNOWN
    
    def get_severity(self, category: ErrorCategory) -> ErrorSeverity:
        """Determine severity based on error category."""
        severity_map = {
            ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.CRITICAL,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.INVALID_REQUEST: ErrorSeverity.MEDIUM,
            ErrorCategory.SERVICE_UNAVAILABLE: ErrorSeverity.HIGH,
            ErrorCategory.CONTEXT_LIMIT: ErrorSeverity.HIGH,
            ErrorCategory.CONTENT_FILTER: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.PARSE_ERROR: ErrorSeverity.LOW,
            ErrorCategory.TASK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
        }
        return severity_map.get(category, ErrorSeverity.MEDIUM)
    
    def is_recoverable(self, category: ErrorCategory) -> bool:
        """Determine if an error category is recoverable."""
        non_recoverable = {
            ErrorCategory.AUTHENTICATION,
            ErrorCategory.CONTENT_FILTER,
        }
        return category not in non_recoverable
    
    def record_error(
        self,
        error_message: str,
        step_idx: int,
        raw_error: Optional[str] = None,
    ) -> ErrorRecord:
        """
        Record and classify an error.
        
        Returns the created ErrorRecord.
        """
        category = self.classify(error_message)
        severity = self.get_severity(category)
        recoverable = self.is_recoverable(category)
        recovery_action = self.RECOVERY_ACTIONS.get(category)
        
        record = ErrorRecord(
            category=category,
            severity=severity,
            message=error_message[:500],  # Truncate long messages
            step_idx=step_idx,
            raw_error=raw_error[:1000] if raw_error else None,
            recoverable=recoverable,
            recovery_action=recovery_action,
        )
        
        self.errors.append(record)
        self.error_counts[category] += 1
        
        return record
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of recorded errors."""
        return {
            "total_errors": len(self.errors),
            "by_category": {
                cat.value: count
                for cat, count in self.error_counts.items()
                if count > 0
            },
            "by_severity": {
                sev.value: sum(
                    1 for e in self.errors if e.severity == sev
                )
                for sev in ErrorSeverity
            },
            "recoverable_count": sum(1 for e in self.errors if e.recoverable),
            "non_recoverable_count": sum(1 for e in self.errors if not e.recoverable),
        }
    
    def reset(self) -> None:
        """Clear all recorded errors."""
        self.errors.clear()
        self.error_counts = {cat: 0 for cat in ErrorCategory}


__all__ = [
    "ErrorTaxonomy",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorRecord",
]

