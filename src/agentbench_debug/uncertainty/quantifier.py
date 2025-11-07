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
