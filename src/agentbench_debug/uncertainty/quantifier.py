import math, numpy as np
from typing import List, Optional, Dict, Any, TypedDict, Literal

Status = Literal["confident","low_confidence"]

class UOut(TypedDict):
    step_idx: int
    module: str
    method: str
    score: float
    status: Status
    meta: Dict[str, Any]

class UncertaintyQuantifier:
    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    @staticmethod
    def _agreement_uncertainty(outputs: List[str]) -> float:
        if not outputs: return 1.0
        most = max(outputs.count(x) for x in set(outputs))
        agree = most / len(outputs)
        return 1.0 - agree  # higher = more uncertain

    def from_samples(self, step_idx: int, module: str, samples: List[str]) -> UOut:
        u = self._agreement_uncertainty(samples)
        return {
            "step_idx": step_idx,
            "module": module,
            "method": "self_consistency",
            "score": float(u),
            "status": "low_confidence" if u > self.threshold else "confident",
            "meta": {"k": len(samples)}
        }
