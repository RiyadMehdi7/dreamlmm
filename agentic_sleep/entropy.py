from __future__ import annotations

import math
from typing import Iterable, Sequence


def shannon_entropy(probs: Sequence[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps), 2) for p in probs if p > 0)


def normalize(scores: Iterable[float], temperature: float = 1.0) -> list[float]:
    vals = list(scores)
    if not vals:
        return [1.0]
    t = max(temperature, 1e-6)
    mx = max(vals)
    exps = [math.exp((v - mx) / t) for v in vals]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def entropy_spike(prev_entropy: float, current_entropy: float, threshold: float) -> bool:
    return (current_entropy - prev_entropy) > threshold

