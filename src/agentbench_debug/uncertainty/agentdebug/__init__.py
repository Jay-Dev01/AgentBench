"""
AgentDebug Module for AgentBench
--------------------------------
This package provides error detection, fine-grained analysis,
and critical error identification utilities for debugging
agent trajectories within AgentBench environments.
"""

from .error_definitions_loader import ErrorDefinitionsLoader
from .fine_grained_analysis import FineGrainedAnalyzer, ModuleError, StepAnalysis
from .critical_error_detection_unified import detect_critical_error_unified, CriticalError

__all__ = [
    "ErrorDefinitionsLoader",
    "FineGrainedAnalyzer",
    "ModuleError",
    "StepAnalysis",
    "detect_critical_error_unified",
    "CriticalError",
]
