from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from time import perf_counter


@dataclass
class Timer:
    _start: float = field(default_factory=perf_counter)

    def elapsed_ms(self) -> float:
        return (perf_counter() - self._start) * 1000.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    sorted_vals = sorted(values)
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def summarize_latencies_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "median_ms": 0.0}
    return {
        "count": float(len(values)),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "median_ms": median(values),
    }

