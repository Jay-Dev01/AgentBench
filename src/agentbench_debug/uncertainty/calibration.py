"""Calibration metrics for uncertainty estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    n_bins: int
    n_samples: int
    # New metrics
    spearman_rho: Optional[float] = None  # Rank correlation between confidence and success
    pearson_rho: Optional[float] = None  # Linear correlation between confidence and success
    auroc: Optional[float] = None  # Area under ROC curve
    mean_confidence: Optional[float] = None
    mean_accuracy: Optional[float] = None


@dataclass 
class OutcomeAnalysis:
    """Analysis of task outcomes vs confidence."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    
    # Confidence by outcome
    mean_confidence_success: float
    mean_confidence_failure: float
    confidence_gap: float  # Difference between success/failure confidence
    
    # Correlation
    spearman_rho: float
    spearman_p_value: float
    
    # Discrimination
    auroc: float
    
    # Calibration
    overconfidence_rate: float  # % of failures with high confidence
    underconfidence_rate: float  # % of successes with low confidence


class CalibrationMetrics:
    """
    Compute calibration metrics for uncertainty estimates.
    
    Metrics:
    - ECE (Expected Calibration Error): Weighted average of |accuracy - confidence|
    - MCE (Maximum Calibration Error): Maximum |accuracy - confidence| across bins
    - Brier Score: Mean squared error of probability estimates
    - Spearman ρ: Rank correlation between confidence and success
    - AUROC: Discrimination ability
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def compute(
        self,
        confidences: List[float],
        outcomes: List[bool],
    ) -> CalibrationResult:
        """
        Compute calibration metrics.
        
        Args:
            confidences: List of confidence scores [0, 1]
            outcomes: List of binary outcomes (True = success, False = failure)
        
        Returns:
            CalibrationResult with all metrics
        """
        if len(confidences) != len(outcomes):
            raise ValueError("confidences and outcomes must have same length")
        
        n_samples = len(confidences)
        if n_samples == 0:
            return CalibrationResult(
                ece=0.0,
                mce=0.0,
                brier_score=0.0,
                bin_accuracies=[],
                bin_confidences=[],
                bin_counts=[],
                n_bins=self.n_bins,
                n_samples=0,
            )
        
        # Initialize bins
        bin_counts = [0] * self.n_bins
        bin_correct = [0] * self.n_bins
        bin_confidence_sum = [0.0] * self.n_bins
        
        # Assign samples to bins
        for conf, outcome in zip(confidences, outcomes):
            # Clamp confidence to [0, 1]
            conf = max(0.0, min(1.0, conf))
            
            # Determine bin (handle edge case where conf == 1.0)
            bin_idx = min(int(conf * self.n_bins), self.n_bins - 1)
            
            bin_counts[bin_idx] += 1
            bin_confidence_sum[bin_idx] += conf
            if outcome:
                bin_correct[bin_idx] += 1
        
        # Compute per-bin accuracies and confidences
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                bin_accuracies.append(bin_correct[i] / bin_counts[i])
                bin_confidences.append(bin_confidence_sum[i] / bin_counts[i])
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append((i + 0.5) / self.n_bins)
        
        # ECE: weighted average of |accuracy - confidence|
        ece = 0.0
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / n_samples) * abs(
                    bin_accuracies[i] - bin_confidences[i]
                )
        
        # MCE: maximum |accuracy - confidence|
        mce = 0.0
        for i in range(self.n_bins):
            if bin_counts[i] > 0:
                gap = abs(bin_accuracies[i] - bin_confidences[i])
                mce = max(mce, gap)
        
        # Brier Score: mean squared error
        brier = 0.0
        for conf, outcome in zip(confidences, outcomes):
            target = 1.0 if outcome else 0.0
            brier += (conf - target) ** 2
        brier /= n_samples
        
        # Compute correlations
        outcome_values = [1.0 if o else 0.0 for o in outcomes]
        spearman = self._compute_spearman(confidences, outcome_values)
        pearson = self._compute_pearson(confidences, outcome_values)
        
        # Compute AUROC
        auroc = self._compute_auroc(confidences, outcomes)
        
        # Mean stats
        mean_conf = sum(confidences) / n_samples
        mean_acc = sum(1 for o in outcomes if o) / n_samples
        
        return CalibrationResult(
            ece=ece,
            mce=mce,
            brier_score=brier,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
            n_samples=n_samples,
            spearman_rho=spearman,
            pearson_rho=pearson,
            auroc=auroc,
            mean_confidence=mean_conf,
            mean_accuracy=mean_acc,
        )
    
    def _compute_spearman(
        self,
        x: List[float],
        y: List[float],
    ) -> float:
        """
        Compute Spearman rank correlation coefficient.
        
        Returns correlation between -1 and 1.
        """
        n = len(x)
        if n < 3:
            return 0.0
        
        # Compute ranks
        def rank(values: List[float]) -> List[float]:
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
            ranks = [0.0] * len(values)
            for rank_val, idx in enumerate(sorted_indices, 1):
                ranks[idx] = float(rank_val)
            return ranks
        
        rank_x = rank(x)
        rank_y = rank(y)
        
        # Compute Spearman correlation (Pearson on ranks)
        mean_rx = sum(rank_x) / n
        mean_ry = sum(rank_y) / n
        
        numerator = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
        
        var_rx = sum((rx - mean_rx) ** 2 for rx in rank_x)
        var_ry = sum((ry - mean_ry) ** 2 for ry in rank_y)
        
        denominator = math.sqrt(var_rx * var_ry)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _compute_pearson(
        self,
        x: List[float],
        y: List[float],
    ) -> float:
        """
        Compute Pearson correlation coefficient.
        
        For binary outcomes (0/1), this is equivalent to point-biserial correlation.
        Returns correlation between -1 and 1.
        """
        n = len(x)
        if n < 3:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = math.sqrt(var_x * var_y)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _compute_auroc(
        self,
        confidences: List[float],
        outcomes: List[bool],
    ) -> float:
        """
        Compute Area Under ROC Curve.
        
        Measures discrimination ability: probability that a random positive
        has higher confidence than a random negative.
        """
        positives = [c for c, o in zip(confidences, outcomes) if o]
        negatives = [c for c, o in zip(confidences, outcomes) if not o]
        
        if not positives or not negatives:
            return 0.5  # Random baseline
        
        # Mann-Whitney U statistic
        n_concordant = 0
        n_ties = 0
        
        for p in positives:
            for n in negatives:
                if p > n:
                    n_concordant += 1
                elif p == n:
                    n_ties += 0.5
        
        auroc = (n_concordant + n_ties) / (len(positives) * len(negatives))
        return auroc
    
    def analyze_outcomes(
        self,
        confidences: List[float],
        outcomes: List[bool],
        high_conf_threshold: float = 0.7,
        low_conf_threshold: float = 0.5,
    ) -> OutcomeAnalysis:
        """
        Detailed analysis of task outcomes vs confidence.
        
        Args:
            confidences: List of confidence scores
            outcomes: List of task success (True/False)
            high_conf_threshold: Threshold for "high confidence"
            low_conf_threshold: Threshold for "low confidence"
        
        Returns:
            OutcomeAnalysis with detailed metrics
        """
        n = len(confidences)
        if n == 0:
            return OutcomeAnalysis(
                total_tasks=0,
                successful_tasks=0,
                failed_tasks=0,
                success_rate=0.0,
                mean_confidence_success=0.0,
                mean_confidence_failure=0.0,
                confidence_gap=0.0,
                spearman_rho=0.0,
                spearman_p_value=1.0,
                auroc=0.5,
                overconfidence_rate=0.0,
                underconfidence_rate=0.0,
            )
        
        # Basic counts
        successes = sum(1 for o in outcomes if o)
        failures = n - successes
        
        # Confidence by outcome
        conf_success = [c for c, o in zip(confidences, outcomes) if o]
        conf_failure = [c for c, o in zip(confidences, outcomes) if not o]
        
        mean_conf_success = sum(conf_success) / len(conf_success) if conf_success else 0.0
        mean_conf_failure = sum(conf_failure) / len(conf_failure) if conf_failure else 0.0
        
        # Spearman correlation
        outcome_values = [1.0 if o else 0.0 for o in outcomes]
        spearman = self._compute_spearman(confidences, outcome_values)
        
        # Approximate p-value (using t-distribution approximation)
        if n > 3:
            t_stat = spearman * math.sqrt((n - 2) / (1 - spearman ** 2 + 1e-10))
            # Simplified p-value (would need scipy for exact)
            p_value = 2 * (1 - self._norm_cdf(abs(t_stat) / math.sqrt(n)))
        else:
            p_value = 1.0
        
        # AUROC
        auroc = self._compute_auroc(confidences, outcomes)
        
        # Over/under confidence rates
        overconfident = sum(
            1 for c, o in zip(confidences, outcomes)
            if not o and c >= high_conf_threshold
        )
        underconfident = sum(
            1 for c, o in zip(confidences, outcomes)
            if o and c < low_conf_threshold
        )
        
        overconf_rate = overconfident / failures if failures > 0 else 0.0
        underconf_rate = underconfident / successes if successes > 0 else 0.0
        
        return OutcomeAnalysis(
            total_tasks=n,
            successful_tasks=successes,
            failed_tasks=failures,
            success_rate=successes / n,
            mean_confidence_success=mean_conf_success,
            mean_confidence_failure=mean_conf_failure,
            confidence_gap=mean_conf_success - mean_conf_failure,
            spearman_rho=spearman,
            spearman_p_value=p_value,
            auroc=auroc,
            overconfidence_rate=overconf_rate,
            underconfidence_rate=underconf_rate,
        )
    
    def _norm_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def compute_reliability_diagram_data(
        self,
        result: CalibrationResult,
    ) -> Dict[str, Any]:
        """
        Get data for plotting a reliability diagram.
        
        Returns dict with:
        - bin_centers: x-axis values (confidence)
        - bin_accuracies: y-axis values (accuracy)
        - bin_counts: sample counts per bin
        - perfect_calibration: diagonal line for reference
        """
        bin_centers = [
            (i + 0.5) / result.n_bins for i in range(result.n_bins)
        ]
        
        return {
            "bin_centers": bin_centers,
            "bin_accuracies": result.bin_accuracies,
            "bin_counts": result.bin_counts,
            "perfect_calibration": bin_centers,  # y = x line
            "ece": result.ece,
            "mce": result.mce,
            "spearman_rho": result.spearman_rho,
            "auroc": result.auroc,
        }


class TemperatureScaling:
    """
    Post-hoc calibration using temperature scaling.
    
    Finds optimal temperature T to scale logits: softmax(z/T)
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature
        self.is_fitted = False
    
    def fit(
        self,
        confidences: List[float],
        outcomes: List[bool],
        lr: float = 0.01,
        max_iters: int = 100,
    ) -> float:
        """
        Fit temperature using gradient descent on NLL.
        
        Returns the optimized temperature.
        """
        if not confidences or not outcomes:
            return self.temperature
        
        # Convert to log-odds space for optimization
        temp = self.temperature
        
        for _ in range(max_iters):
            # Compute gradient of NLL w.r.t. temperature
            grad = 0.0
            for conf, outcome in zip(confidences, outcomes):
                # Clamp to avoid log(0)
                conf = max(1e-7, min(1 - 1e-7, conf))
                
                # Scaled confidence (simplified model)
                scaled = self._scale_confidence(conf, temp)
                
                target = 1.0 if outcome else 0.0
                grad += (scaled - target) * scaled * (1 - scaled) / temp
            
            grad /= len(confidences)
            
            # Update temperature
            temp = max(0.1, temp - lr * grad)
        
        self.temperature = temp
        self.is_fitted = True
        return temp
    
    def _scale_confidence(self, conf: float, temp: float) -> float:
        """Apply temperature scaling to a confidence score."""
        # Convert to logit, scale, convert back
        conf = max(1e-7, min(1 - 1e-7, conf))
        logit = math.log(conf / (1 - conf))
        scaled_logit = logit / temp
        return 1 / (1 + math.exp(-scaled_logit))
    
    def calibrate(self, confidence: float) -> float:
        """Apply temperature scaling to calibrate a confidence score."""
        return self._scale_confidence(confidence, self.temperature)
    
    def calibrate_batch(self, confidences: List[float]) -> List[float]:
        """Calibrate a batch of confidence scores."""
        return [self.calibrate(c) for c in confidences]


__all__ = [
    "CalibrationMetrics",
    "CalibrationResult",
    "OutcomeAnalysis",
    "TemperatureScaling",
]
