"""
Calibration Metrics for Uncertainty Quantification.

Implements:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability Diagrams
- Static Calibration Error (SCE)
- Maximum Calibration Error (MCE)

These metrics measure how well-calibrated an agent's confidence predictions are
relative to actual success outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationBin:
    """A single bin in a reliability diagram."""
    bin_idx: int
    lower_bound: float
    upper_bound: float
    mean_confidence: float       # Average predicted confidence in bin
    mean_accuracy: float         # Average actual accuracy in bin
    count: int                   # Number of samples in bin
    gap: float                   # |accuracy - confidence|
    samples: List[int] = field(default_factory=list)  # Sample indices


@dataclass
class ReliabilityDiagram:
    """Complete reliability diagram data."""
    bins: List[CalibrationBin]
    n_bins: int
    total_samples: int
    perfectly_calibrated_line: List[Tuple[float, float]]  # For plotting
    

@dataclass
class CalibrationResult:
    """Complete calibration analysis result."""
    ece: float                   # Expected Calibration Error
    mce: float                   # Maximum Calibration Error
    sce: float                   # Static Calibration Error
    brier_score: float           # Brier Score
    reliability_diagram: ReliabilityDiagram
    overconfident: bool          # True if agent is generally overconfident
    underconfident: bool         # True if agent is generally underconfident
    calibration_slope: float     # Slope of calibration curve (ideal = 1.0)
    calibration_intercept: float # Intercept of calibration curve (ideal = 0.0)
    confidence_histogram: List[int]  # Distribution of confidence values


class CalibrationMetrics:
    """
    Compute calibration metrics for uncertainty-aware agents.
    
    Measures alignment between predicted confidence and actual success,
    essential for API orchestration where over/underconfidence can lead
    to failed workflows or unnecessary retries.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for ECE and reliability diagrams
        """
        self.n_bins = n_bins
        self._predictions: List[Tuple[float, bool, int]] = []  # (confidence, correct, sample_idx)
    
    def add_prediction(
        self,
        confidence: float,
        correct: bool,
        sample_idx: Optional[int] = None,
    ) -> None:
        """
        Add a prediction-outcome pair.
        
        Args:
            confidence: Predicted confidence (0-1)
            correct: Whether the prediction was actually correct
            sample_idx: Optional sample identifier
        """
        if sample_idx is None:
            sample_idx = len(self._predictions)
        
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        self._predictions.append((confidence, correct, sample_idx))
    
    def add_batch(
        self,
        confidences: List[float],
        outcomes: List[bool],
    ) -> None:
        """
        Add a batch of prediction-outcome pairs.
        
        Args:
            confidences: List of confidence values
            outcomes: List of correctness indicators
        """
        for i, (conf, correct) in enumerate(zip(confidences, outcomes)):
            self.add_prediction(conf, correct, len(self._predictions) + i)
    
    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE = Σ (n_b / N) * |accuracy_b - confidence_b|
        
        Lower is better. 0 = perfectly calibrated.
        
        Returns:
            ECE value (0-1)
        """
        if not self._predictions:
            return 0.0
        
        bins = self._create_bins()
        
        n_total = len(self._predictions)
        ece = 0.0
        
        for bin_data in bins:
            if bin_data.count > 0:
                weight = bin_data.count / n_total
                ece += weight * bin_data.gap
        
        return ece
    
    def compute_mce(self) -> float:
        """
        Compute Maximum Calibration Error.
        
        MCE = max_b |accuracy_b - confidence_b|
        
        Returns:
            MCE value (0-1)
        """
        if not self._predictions:
            return 0.0
        
        bins = self._create_bins()
        
        gaps = [b.gap for b in bins if b.count > 0]
        
        return max(gaps) if gaps else 0.0
    
    def compute_sce(self) -> float:
        """
        Compute Static Calibration Error (class-conditional ECE).
        
        Measures calibration error separately for positive and negative classes.
        
        Returns:
            SCE value (0-1)
        """
        if not self._predictions:
            return 0.0
        
        # Split into positive and negative
        positives = [(c, o, i) for c, o, i in self._predictions if o]
        negatives = [(c, o, i) for c, o, i in self._predictions if not o]
        
        sce = 0.0
        n_total = len(self._predictions)
        
        for subset in [positives, negatives]:
            if not subset:
                continue
            
            # Create bins for this subset
            bins = self._create_bins_from_predictions(subset)
            
            for bin_data in bins:
                if bin_data.count > 0:
                    weight = bin_data.count / n_total
                    sce += weight * bin_data.gap
        
        return sce
    
    def compute_brier_score(self) -> float:
        """
        Compute Brier Score.
        
        Brier = (1/N) * Σ (confidence - outcome)^2
        
        Lower is better. 0 = perfect predictions.
        
        Returns:
            Brier score (0-1)
        """
        if not self._predictions:
            return 0.0
        
        total_squared_error = 0.0
        
        for confidence, correct, _ in self._predictions:
            outcome = 1.0 if correct else 0.0
            total_squared_error += (confidence - outcome) ** 2
        
        return total_squared_error / len(self._predictions)
    
    def compute_reliability_diagram(self) -> ReliabilityDiagram:
        """
        Compute data for a reliability diagram.
        
        Returns:
            ReliabilityDiagram with bin data for visualization
        """
        bins = self._create_bins()
        
        return ReliabilityDiagram(
            bins=bins,
            n_bins=self.n_bins,
            total_samples=len(self._predictions),
            perfectly_calibrated_line=[
                (0.0, 0.0),
                (0.5, 0.5),
                (1.0, 1.0),
            ],
        )
    
    def compute_calibration_slope_intercept(self) -> Tuple[float, float]:
        """
        Compute linear calibration curve parameters.
        
        Ideal: slope=1.0, intercept=0.0
        
        Returns:
            (slope, intercept) tuple
        """
        if not self._predictions:
            return (1.0, 0.0)
        
        bins = self._create_bins()
        
        # Get non-empty bins
        points = [(b.mean_confidence, b.mean_accuracy) for b in bins if b.count > 0]
        
        if len(points) < 2:
            return (1.0, 0.0)
        
        # Linear regression
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        x_mean = sum(x_vals) / len(x_vals)
        y_mean = sum(y_vals) / len(y_vals)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        if denominator == 0:
            return (1.0, y_mean)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        return (slope, intercept)
    
    def compute_all(self) -> CalibrationResult:
        """
        Compute all calibration metrics.
        
        Returns:
            CalibrationResult with all metrics
        """
        ece = self.compute_ece()
        mce = self.compute_mce()
        sce = self.compute_sce()
        brier = self.compute_brier_score()
        reliability = self.compute_reliability_diagram()
        slope, intercept = self.compute_calibration_slope_intercept()
        
        # Determine over/underconfidence
        overconfident = False
        underconfident = False
        
        if self._predictions:
            total_gap = 0.0
            for bin_data in reliability.bins:
                if bin_data.count > 0:
                    # Positive gap = overconfident, negative = underconfident
                    signed_gap = bin_data.mean_confidence - bin_data.mean_accuracy
                    total_gap += signed_gap * bin_data.count
            
            avg_gap = total_gap / len(self._predictions)
            
            if avg_gap > 0.05:
                overconfident = True
            elif avg_gap < -0.05:
                underconfident = True
        
        # Confidence histogram
        histogram = [0] * self.n_bins
        for conf, _, _ in self._predictions:
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            histogram[bin_idx] += 1
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            sce=sce,
            brier_score=brier,
            reliability_diagram=reliability,
            overconfident=overconfident,
            underconfident=underconfident,
            calibration_slope=slope,
            calibration_intercept=intercept,
            confidence_histogram=histogram,
        )
    
    def _create_bins(self) -> List[CalibrationBin]:
        """Create bins from stored predictions."""
        return self._create_bins_from_predictions(self._predictions)
    
    def _create_bins_from_predictions(
        self,
        predictions: List[Tuple[float, bool, int]],
    ) -> List[CalibrationBin]:
        """Create calibration bins from prediction list."""
        bins: List[CalibrationBin] = []
        
        bin_width = 1.0 / self.n_bins
        
        for i in range(self.n_bins):
            lower = i * bin_width
            upper = (i + 1) * bin_width
            
            bins.append(CalibrationBin(
                bin_idx=i,
                lower_bound=lower,
                upper_bound=upper,
                mean_confidence=0.0,
                mean_accuracy=0.0,
                count=0,
                gap=0.0,
                samples=[],
            ))
        
        # Assign predictions to bins
        for confidence, correct, sample_idx in predictions:
            bin_idx = min(int(confidence * self.n_bins), self.n_bins - 1)
            bins[bin_idx].samples.append(sample_idx)
        
        # Compute bin statistics
        for bin_data in bins:
            bin_data.count = len(bin_data.samples)
            
            if bin_data.count > 0:
                # Get predictions in this bin
                bin_predictions = [
                    (c, o) for c, o, i in predictions
                    if i in bin_data.samples
                ]
                
                bin_data.mean_confidence = sum(c for c, _ in bin_predictions) / bin_data.count
                bin_data.mean_accuracy = sum(1 for _, o in bin_predictions if o) / bin_data.count
                bin_data.gap = abs(bin_data.mean_accuracy - bin_data.mean_confidence)
        
        return bins
    
    def reset(self) -> None:
        """Clear all stored predictions."""
        self._predictions = []
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get a summary dictionary of calibration metrics."""
        result = self.compute_all()
        
        return {
            "ece": result.ece,
            "mce": result.mce,
            "sce": result.sce,
            "brier_score": result.brier_score,
            "overconfident": result.overconfident,
            "underconfident": result.underconfident,
            "calibration_slope": result.calibration_slope,
            "calibration_intercept": result.calibration_intercept,
            "n_samples": len(self._predictions),
            "n_bins": self.n_bins,
        }
    
    def export_reliability_data(self) -> Dict[str, Any]:
        """Export data for plotting reliability diagram externally."""
        diagram = self.compute_reliability_diagram()
        
        return {
            "bin_centers": [(b.lower_bound + b.upper_bound) / 2 for b in diagram.bins],
            "accuracies": [b.mean_accuracy for b in diagram.bins],
            "confidences": [b.mean_confidence for b in diagram.bins],
            "counts": [b.count for b in diagram.bins],
            "gaps": [b.gap for b in diagram.bins],
            "n_samples": diagram.total_samples,
        }


class TemperatureScaling:
    """
    Temperature scaling for post-hoc calibration.
    
    Learns a temperature parameter T to calibrate confidence:
    calibrated_conf = softmax(logits / T)
    """
    
    def __init__(self, initial_temp: float = 1.0):
        """
        Initialize temperature scaling.
        
        Args:
            initial_temp: Initial temperature value
        """
        self.temperature = initial_temp
        self._logits: List[float] = []
        self._labels: List[bool] = []
    
    def add_sample(self, logit: float, correct: bool) -> None:
        """Add a training sample for temperature learning."""
        self._logits.append(logit)
        self._labels.append(correct)
    
    def fit(self, n_iterations: int = 100, learning_rate: float = 0.01) -> float:
        """
        Fit temperature parameter using gradient descent on NLL.
        
        Args:
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient descent
        
        Returns:
            Optimal temperature value
        """
        if not self._logits:
            return self.temperature
        
        temp = self.temperature
        
        for _ in range(n_iterations):
            # Compute gradient of NLL w.r.t. temperature
            grad = 0.0
            
            for logit, correct in zip(self._logits, self._labels):
                scaled = logit / temp
                prob = 1.0 / (1.0 + math.exp(-scaled))  # sigmoid
                
                target = 1.0 if correct else 0.0
                
                # Gradient
                grad += (prob - target) * (-logit / (temp ** 2))
            
            grad /= len(self._logits)
            
            # Update temperature
            temp = temp - learning_rate * grad
            temp = max(0.1, min(10.0, temp))  # Clamp to reasonable range
        
        self.temperature = temp
        return temp
    
    def calibrate(self, confidence: float) -> float:
        """
        Apply temperature scaling to a confidence value.
        
        Args:
            confidence: Original confidence (0-1)
        
        Returns:
            Calibrated confidence
        """
        # Convert confidence to logit
        confidence = max(1e-6, min(1 - 1e-6, confidence))
        logit = math.log(confidence / (1 - confidence))
        
        # Scale and convert back
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        
        return calibrated


__all__ = [
    "CalibrationMetrics",
    "CalibrationResult",
    "CalibrationBin",
    "ReliabilityDiagram",
    "TemperatureScaling",
]

