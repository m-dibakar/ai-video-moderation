from dataclasses import dataclass
from typing import List


@dataclass
class FrameScore:
    timestamp: float
    label: int
    confidence: float


@dataclass
class Violation:
    start_time: float
    end_time: float
    violation_type: str
    severity: str
    avg_confidence: float
    suggested_action: str


class TemporalFilter:
    """
    Converts frame-level scores into violation segments.
    Requires violations to persist for min_duration seconds
    before flagging — eliminates single-frame false positives.
    """
    def __init__(self, threshold=0.5, min_duration=3.0, gap_tolerance=2.0):
        self.threshold = threshold
        self.min_duration = min_duration
        self.gap_tolerance = gap_tolerance

    def filter(self, frame_scores: List[FrameScore]) -> List[Violation]:
        violations = []
        in_violation = False
        violation_start = None
        violation_frames = []

        for frame in frame_scores:
            is_violation = (frame.label == 1 and frame.confidence >= self.threshold)

            if is_violation and not in_violation:
                in_violation = True
                violation_start = frame.timestamp
                violation_frames = [frame]

            elif is_violation and in_violation:
                violation_frames.append(frame)

            elif not is_violation and in_violation:
                duration = frame.timestamp - violation_start
                if duration >= self.min_duration:
                    violations.append(
                        self._build_violation(violation_start, frame.timestamp, violation_frames)
                    )
                in_violation = False
                violation_frames = []

        if in_violation and violation_frames:
            duration = violation_frames[-1].timestamp - violation_start
            if duration >= self.min_duration:
                violations.append(
                    self._build_violation(violation_start, violation_frames[-1].timestamp, violation_frames)
                )

        return self._merge_adjacent(violations)

    def _build_violation(self, start, end, frames) -> Violation:
        avg_conf = sum(f.confidence for f in frames) / len(frames)
        if avg_conf > 0.85:
            severity, action = "high", "auto-blur"
        elif avg_conf > 0.65:
            severity, action = "medium", "flag-for-review"
        else:
            severity, action = "low", "log-only"

        return Violation(
            start_time=start,
            end_time=end,
            violation_type="nsfw",
            severity=severity,
            avg_confidence=round(avg_conf, 3),
            suggested_action=action
        )

    def _merge_adjacent(self, violations: List[Violation]) -> List[Violation]:
        if not violations:
            return []
        merged = [violations[0]]
        for v in violations[1:]:
            if v.start_time - merged[-1].end_time <= self.gap_tolerance:
                merged[-1].end_time = v.end_time
                merged[-1].avg_confidence = max(merged[-1].avg_confidence, v.avg_confidence)
            else:
                merged.append(v)
        return merged
