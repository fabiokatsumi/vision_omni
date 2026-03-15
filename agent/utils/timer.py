"""Lightweight step timer for profiling agent loop stages."""

import time
from collections import OrderedDict


class StepTimer:
    def __init__(self):
        self._starts: dict[str, float] = {}
        self._durations: OrderedDict[str, float] = OrderedDict()

    def start(self, name: str):
        self._starts[name] = time.time()

    def stop(self, name: str):
        if name in self._starts:
            elapsed = time.time() - self._starts.pop(name)
            self._durations[name] = self._durations.get(name, 0) + elapsed

    def get(self, name: str) -> float:
        return self._durations.get(name, 0.0)

    def total(self) -> float:
        return sum(self._durations.values())

    def summary(self) -> str:
        parts = [f"{name}: {dur:.2f}s" for name, dur in self._durations.items()]
        return " | ".join(parts)

    def merge(self, other: "StepTimer"):
        for name, dur in other._durations.items():
            self._durations[name] = self._durations.get(name, 0) + dur
