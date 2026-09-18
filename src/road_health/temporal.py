"""Turning noisy per-frame flags into stable operational events.

A single bad frame is not a camera fault.  A truck passing close to the lens
flattens the histogram for four frames; a bird lands on the housing for half a
second; one frame drops and the inter-frame difference goes to zero.  Raising
an alarm on any of these trains operators to ignore alarms, which is the actual
failure mode that matters in a control room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from road_health.health_checks import CheckResult

STATUS_ORDER = {"OK": 0, "DEGRADED": 1, "UNUSABLE": 2}


@dataclass
class Event:
    condition: str
    severity: str
    start_frame: int
    start_time_s: float
    end_frame: Optional[int] = None
    end_time_s: Optional[float] = None
    n_bad_frames: int = 0
    peak_value: float = 0.0
    threshold: float = 0.0
    first_detail: str = ""
    detection_latency_frames: int = 0
    co_active: set = field(default_factory=set)
    ongoing: bool = True

    def duration_s(self) -> Optional[float]:
        if self.end_time_s is None:
            return None
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> Dict:
        return {
            "condition": self.condition,
            "severity": self.severity,
            "start_frame": self.start_frame,
            "start_time_s": round(self.start_time_s, 3),
            "end_frame": self.end_frame,
            "end_time_s": None if self.end_time_s is None else round(self.end_time_s, 3),
            "duration_s": None if self.duration_s() is None else round(self.duration_s(), 3),
            "n_bad_frames": self.n_bad_frames,
            "peak_value": round(self.peak_value, 5),
            "threshold": round(self.threshold, 5),
            "detection_latency_frames": self.detection_latency_frames,
            "co_active": sorted(self.co_active),
            "first_detail": self.first_detail,
            "ongoing": self.ongoing,
        }


class Debouncer:
    def __init__(self, enter_frames: int, exit_frames: int):
        self.enter_frames = max(1, int(enter_frames))
        self.exit_frames = max(1, int(exit_frames))
        self.drain = self.enter_frames / float(self.exit_frames)
        self.level = 0.0
        self.active = False

    def update(self, bad: bool) -> bool:
        if bad:
            self.level = min(self.enter_frames, self.level + 1.0)
        else:
            self.level = max(0.0, self.level - self.drain)

        if not self.active and self.level >= self.enter_frames:
            self.active = True
        elif self.active and self.level <= 0.0:
            self.active = False
        return self.active


class EventTracker:
    def __init__(self, conditions: Dict[str, str], enter: Dict[str, int], exit_: Dict[str, int]):
        self.severity = conditions
        self.debouncers = {
            name: Debouncer(enter[name], exit_[name]) for name in conditions
        }
        self.events: List[Event] = []
        self._open: Dict[str, Event] = {}
        self._first_bad: Dict[str, Optional[tuple]] = {n: None for n in conditions}
        self._bad_counts: Dict[str, int] = {n: 0 for n in conditions}

    def update(
        self, frame_index: int, timestamp_s: float, results: Dict[str, CheckResult]
    ) -> Dict[str, bool]:
        active: Dict[str, bool] = {}
        for name, deb in self.debouncers.items():
            res = results.get(name)
            bad = bool(res.triggered) if res is not None else False
            if bad:
                self._bad_counts[name] += 1
                if self._first_bad[name] is None:
                    self._first_bad[name] = (frame_index, timestamp_s)
            was = deb.active
            now = deb.update(bad)
            active[name] = now

            if now and not was:
                start_frame, start_time = self._first_bad[name] or (frame_index, timestamp_s)
                ev = Event(
                    condition=name,
                    severity=self.severity[name],
                    start_frame=start_frame,
                    start_time_s=start_time,
                    n_bad_frames=self._bad_counts[name],
                    peak_value=res.value if res else 0.0,
                    threshold=res.threshold if res else 0.0,
                    first_detail=res.detail if res else "",
                    detection_latency_frames=frame_index - start_frame,
                    co_active={
                        k for k, r in results.items() if k != name and r.triggered
                    },
                )
                self._open[name] = ev
                self.events.append(ev)
            elif now and was:
                ev = self._open.get(name)
                if ev is not None and res is not None:
                    ev.n_bad_frames = self._bad_counts[name]
                    ev.co_active |= {
                        k for k, r in results.items()
                        if k != name and r.triggered
                    }
                    if res.triggered and abs(res.value - res.threshold) > abs(
                        ev.peak_value - ev.threshold
                    ):
                        ev.peak_value = res.value
            elif was and not now:
                ev = self._open.pop(name, None)
                if ev is not None:
                    ev.end_frame = frame_index
                    ev.end_time_s = timestamp_s
                    ev.ongoing = False
                self._first_bad[name] = None
                self._bad_counts[name] = 0

            if not bad and not now:
                self._first_bad[name] = None
                self._bad_counts[name] = 0
        return active

    def finalize(self, frame_index: int, timestamp_s: float) -> None:
        for name, ev in list(self._open.items()):
            ev.end_frame = frame_index
            ev.end_time_s = timestamp_s
            ev.ongoing = True
            self._open.pop(name, None)


def frame_status(active: Dict[str, bool], severity: Dict[str, str]) -> str:
    worst = "OK"
    for name, is_active in active.items():
        if not is_active:
            continue
        sev = severity.get(name, "DEGRADED")
        if STATUS_ORDER[sev] > STATUS_ORDER[worst]:
            worst = sev
    return worst
