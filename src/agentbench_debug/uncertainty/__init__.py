"""
Uncertainty quantification modules for AgentBench.

This module provides comprehensive uncertainty estimation for LLM agent workflows:

Components:
- quantifier: Basic uncertainty from sample disagreement
- confidence_extractor: Extract confidence from API responses (logprobs, finish_reason)
- hierarchical: SAUP-style multi-level uncertainty propagation
- calibration: ECE, Brier score, temperature scaling
- scoring: Progress, completion, and composite scores
- error_taxonomy: Error classification and recovery guidance
- mitigation: Retry, circuit breaker, rate limiting strategies
- orchestration_harness: Unified interface for all components
- agent_wrapper: Wrap agents for real-time uncertainty tracking
- pipeline_integration: Helpers for AgentBench integration

Usage:
    from src.agentbench_debug.uncertainty import (
        UncertaintyAwareAgent,
        create_uncertainty_agent,
        OrchestrationHarness,
        ConfidenceExtractor,
    )
    
    # Wrap an agent for uncertainty tracking
    wrapped_agent = create_uncertainty_agent(agent, "alfworld-std")
    
    # Or use the harness directly
    harness = OrchestrationHarness(task_id="task1", task_type="alfworld")
    harness.record_step(action="look", action_type="environment_action", ...)
    result = harness.finish_workflow(task_completed=True)
"""

# Core quantification
from .quantifier import UncertaintyQuantifier, UOut

# Confidence extraction from API responses
from .confidence_extractor import ConfidenceExtractor

# Hierarchical uncertainty propagation
from .hierarchical import (
    HierarchicalUncertainty,
    UncertaintyLevel,
    TokenUncertainty,
    ActionUncertainty,
    ObservationUncertainty,
    StepRecord,
)

# Calibration metrics
from .calibration import (
    CalibrationMetrics,
    CalibrationResult,
    OutcomeAnalysis,
    TemperatureScaling,
)

# Scoring system
from .scoring import (
    Scorer,
    StepResult,
    ProgressScore,
    CompletionScore,
    CompositeScore,
    EvaluationResult,
)

# Error taxonomy
from .error_taxonomy import (
    ErrorTaxonomy,
    ErrorCategory,
    ErrorSeverity,
    ErrorRecord,
)

# Mitigation strategies
from .mitigation import (
    MitigationStrategy,
    RetryConfig,
    CircuitBreakerConfig,
    RetryStrategy,
    CircuitBreaker,
    CircuitBreakerOpenException,
    RateLimiter,
    DecisionLayer,
    MitigationDecision,
)

# Orchestration harness
from .orchestration_harness import (
    OrchestrationHarness,
    WorkflowStep,
    WorkflowResult,
)

# Agent wrapper
from .agent_wrapper import (
    UncertaintyAwareAgent,
    UncertaintyTracker,
    StepInfo,
    wrap_agent,
)

# Pipeline integration
from .pipeline_integration import (
    UncertaintyReport,
    UncertaintyCallback,
    create_uncertainty_agent,
    analyze_saved_runs,
)


__all__ = [
    # Quantifier
    "UncertaintyQuantifier",
    "UOut",
    # Confidence Extractor
    "ConfidenceExtractor",
    # Hierarchical
    "HierarchicalUncertainty",
    "UncertaintyLevel",
    "TokenUncertainty",
    "ActionUncertainty",
    "ObservationUncertainty",
    "StepRecord",
    # Calibration
    "CalibrationMetrics",
    "CalibrationResult",
    "OutcomeAnalysis",
    "TemperatureScaling",
    # Scoring
    "Scorer",
    "StepResult",
    "ProgressScore",
    "CompletionScore",
    "CompositeScore",
    "EvaluationResult",
    # Error Taxonomy
    "ErrorTaxonomy",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorRecord",
    # Mitigation
    "MitigationStrategy",
    "RetryConfig",
    "CircuitBreakerConfig",
    "RetryStrategy",
    "CircuitBreaker",
    "CircuitBreakerOpenException",
    "RateLimiter",
    "DecisionLayer",
    "MitigationDecision",
    # Orchestration
    "OrchestrationHarness",
    "WorkflowStep",
    "WorkflowResult",
    # Agent Wrapper
    "UncertaintyAwareAgent",
    "UncertaintyTracker",
    "StepInfo",
    "wrap_agent",
    # Pipeline Integration
    "UncertaintyReport",
    "UncertaintyCallback",
    "create_uncertainty_agent",
    "analyze_saved_runs",
]
