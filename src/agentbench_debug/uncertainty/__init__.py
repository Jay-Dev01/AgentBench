"""
Uncertainty Quantification Modules for AgentBench.

This package implements the API-ORCHA-Bench uncertainty estimation framework:

1. Hierarchical Uncertainty Propagation (SAUP-style):
   - Token-level: Entropy from LLM output distribution
   - Action-level: Confidence in API action selection
   - Observation-level: Reliability of response interpretation
   - Trajectory-level: Cumulative workflow confidence

2. Calibration Metrics:
   - Expected Calibration Error (ECE)
   - Brier Score
   - Maximum Calibration Error (MCE)
   - Reliability Diagrams

3. Error Taxonomy (API Orchestration):
   - Authentication failures
   - Rate limiting violations
   - Partial workflow failures
   - Data consistency errors
   - Coordination failures

4. Scoring System:
   - Progress Score (incremental completion)
   - Full Completion Score (binary success)
   - Composite Score (weighted combination)

5. Mitigation Strategies:
   - Uncertainty-aware retry with exponential backoff
   - Cross-API validation
   - Circuit breaker pattern
   - Rate limiting intelligence

6. Orchestration Harness:
   - Unified evaluation interface
   - Trajectory recording and analysis
   - Aggregate metrics computation
"""

# Core quantifier (existing)
from .quantifier import UncertaintyQuantifier, UOut

# Hierarchical uncertainty propagation
from .hierarchical import (
    HierarchicalUncertaintyPropagator,
    HierarchicalUncertaintyResult,
    TokenUncertainty,
    ActionUncertainty,
    ObservationUncertainty,
    TrajectoryUncertainty,
    ActionCriticality,
)

# Calibration metrics
from .calibration import (
    CalibrationMetrics,
    CalibrationResult,
    CalibrationBin,
    ReliabilityDiagram,
    TemperatureScaling,
)

# Error taxonomy
from .error_taxonomy import (
    ErrorTaxonomy,
    ErrorCategory,
    ErrorInstance,
    ErrorSummary,
    AuthenticationErrorType,
    RateLimitErrorType,
    PartialFailureErrorType,
    DataConsistencyErrorType,
    CoordinationErrorType,
)

# Scoring system
from .scoring import (
    ScoringSystem,
    Checkpoint,
    CheckpointType,
    ProgressScore,
    FullCompletionScore,
    CompositeScore,
    UncertaintyWeightedScore,
    FullEvaluationResult,
)

# Mitigation strategies
from .mitigation import (
    MitigationStrategy,
    UncertaintyAwareRetry,
    CrossAPIValidator,
    CircuitBreaker,
    RateLimiter,
    DecisionLayer,
    RetryResult,
    ValidationResult,
    CircuitState,
    MitigationRecord,
)

# Orchestration harness
from .orchestration_harness import (
    OrchestrationHarness,
    EvaluationConfig,
    WorkflowRun,
    StepRecord,
    evaluate_trajectory,
)


__all__ = [
    # Core
    "UncertaintyQuantifier",
    "UOut",
    
    # Hierarchical
    "HierarchicalUncertaintyPropagator",
    "HierarchicalUncertaintyResult",
    "TokenUncertainty",
    "ActionUncertainty",
    "ObservationUncertainty",
    "TrajectoryUncertainty",
    "ActionCriticality",
    
    # Calibration
    "CalibrationMetrics",
    "CalibrationResult",
    "CalibrationBin",
    "ReliabilityDiagram",
    "TemperatureScaling",
    
    # Errors
    "ErrorTaxonomy",
    "ErrorCategory",
    "ErrorInstance",
    "ErrorSummary",
    "AuthenticationErrorType",
    "RateLimitErrorType",
    "PartialFailureErrorType",
    "DataConsistencyErrorType",
    "CoordinationErrorType",
    
    # Scoring
    "ScoringSystem",
    "Checkpoint",
    "CheckpointType",
    "ProgressScore",
    "FullCompletionScore",
    "CompositeScore",
    "UncertaintyWeightedScore",
    "FullEvaluationResult",
    
    # Mitigation
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
    
    # Harness
    "OrchestrationHarness",
    "EvaluationConfig",
    "WorkflowRun",
    "StepRecord",
    "evaluate_trajectory",
]

