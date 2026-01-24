from collections import Counter
from typing import Any, Dict, List, Literal, TypedDict

Status = Literal["confident","low_confidence"]

class UOut(TypedDict):
    step_idx: int
    module: str
    method: str
    score: float
    status: Status
    meta: Dict[str, Any]

class UncertaintyQuantifier:
    """Compute disagreement-based uncertainty from multiple sampled outputs."""

    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold

    @staticmethod
    def _agreement_uncertainty(outputs: List[str]) -> float:
        if not outputs:
            return 1.0

        normalized = [o.strip() for o in outputs if o and o.strip()]
        if not normalized:
            return 1.0

        counter = Counter(normalized)
        most_common_count = counter.most_common(1)[0][1]
        agreement = most_common_count / len(normalized)
        return 1.0 - agreement  # higher = more uncertain

    def from_samples(self, step_idx: int, module: str, samples: List[str]) -> UOut:
        score = float(self._agreement_uncertainty(samples))
        status: Status = "low_confidence" if score > self.threshold else "confident"

        return {
            "step_idx": step_idx,
            "module": module,
            "method": "self_consistency",
            "score": score,
            "status": status,
            "meta": {"k": len(samples), "threshold": self.threshold},
        }

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
UOut = Dict[str, Any]


@dataclass
class UncertaintyQuantifier:
    """Compute a simple uncertainty score from candidate action samples.

    Parameters
    ----------
    threshold : float, optional
        A cut-off in the range ``[0, 1]`` used to convert the numeric
        uncertainty score into a qualitative status.  Scores above
        ``threshold`` will yield a status of ``"low_confidence"``, while
        scores at or below the threshold will yield ``"confident"``.

    Examples
    --------
    >>> uq = UncertaintyQuantifier(threshold=0.35)
    >>> samples = ["look around", "look around", "examine room"]
    >>> result = uq.from_samples(step_idx=0, module="action", samples=samples)
    >>> round(result["score"], 2)
    0.33
    >>> result["status"]
    'confident'
    """

    threshold: float = 0.5

    def from_samples(
        self,
        *,
        step_idx: int,
        module: str,
        samples: List[str],
    ) -> UOut:
        """Compute the uncertainty score from a list of candidate samples.

        Parameters
        ----------
        step_idx : int
            The index of the current step (not used in the calculation but
            provided for completeness).
        module : str
            The name of the module generating the samples (e.g., ``"action"``).
        samples : list of str
            A collection of candidate actions produced by the agent.  The
            collection may contain duplicates.

        Returns
        -------
        dict
            A dictionary containing the numeric uncertainty score, a
            qualitative status string, and metadata about the sample
            distribution.
        """
        # Guard against empty inputs
        if not samples:
            return {
                "score": 0.0,
                "status": "confident",
                "meta": {"detail": "no_samples", "distribution": {}},
            }

        # Count occurrences of each unique sample
        counts = Counter([s.strip() for s in samples if s is not None])
        total = sum(counts.values())
        if total == 0:
            return {
                "score": 0.0,
                "status": "confident",
                "meta": {"detail": "zero_total", "distribution": {}},
            }

        # Highest relative frequency determines confidence
        most_common_count = counts.most_common(1)[0][1]
        freq = most_common_count / total
        score = 1.0 - freq

        status = "low_confidence" if score > self.threshold else "confident"

        # Prepare metadata with distribution sorted descending
        distribution = {k: v / total for k, v in counts.most_common()}

        return {
            "score": score,
            "status": status,
            "meta": {
                "step_idx": step_idx,
                "module": module,
                "total_samples": total,
                "distribution": distribution,
            },
        }


__all__ = ["UncertaintyQuantifier", "UOut"]
