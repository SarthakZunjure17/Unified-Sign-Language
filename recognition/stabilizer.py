"""
recognition/stabilizer.py
-------------------------
Temporal gesture stabilization and majority-voting mechanism.
Prevents frame-by-frame flicker and suppresses duplicate gesture commitments.
"""

from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any, Set

class TemporalStabilizer:
    """
    Stabilizes gesture predictions across a rolling window using majority voting.
    Emits a single commit event per stable gesture and suppresses continuous duplicates.
    Re-arms only when stable prediction changes or sufficient idle/unknown frames elapse.
    """
    def __init__(
        self,
        window_size: int = 12,
        min_confidence: float = 0.70,
        min_stable_count: int = 7,
        rearm_frames: int = 8,
        cooldown_frames: Optional[int] = None,
        demo_safe: bool = True,
        excluded_classes: Optional[Set[int]] = None
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_stable_count = min_stable_count
        # Support both rearm_frames and legacy cooldown_frames parameter
        self.rearm_frames = cooldown_frames if cooldown_frames is not None else rearm_frames
        self.cooldown_frames = self.rearm_frames
        self.demo_safe = demo_safe
        self.excluded_classes = excluded_classes if excluded_classes is not None else {42}

        self.history = deque(maxlen=window_size)
        self.last_committed: Optional[str] = None
        self.last_committed_class: Optional[int] = None
        self.stable_gesture: Optional[str] = None
        self.stable_class: Optional[int] = None
        self.stable_label: str = "None"
        self.consecutive_idle_frames: int = 0
        self.last_commit_event: Optional[Dict[str, Any]] = None

    def update(self, prediction: Dict[str, Any], hand_detected: bool = True) -> Dict[str, Any]:
        """
        Updates stabilizer with the latest frame prediction.
        Emits commit ONLY when a gesture becomes stable AND stable_gesture != last_committed.
        In demo-safe mode, excluded classes (Class 42) are NEVER committed or stabilized.
        """
        is_excluded = self.demo_safe and (
            prediction.get("is_excluded", False)
            or prediction.get("class_id", -1) in self.excluded_classes
            or prediction.get("raw_class_id", -1) in self.excluded_classes
        )

        # If excluded, no hand detected, or invalid class ID, increment idle count
        if is_excluded:
            self.history.append((-1, "Unknown", 0.0))
            self.consecutive_idle_frames += 1
            if self.consecutive_idle_frames >= self.rearm_frames:
                self.last_committed = None
                self.last_committed_class = None
                self.stable_gesture = None
                self.stable_class = None
            return {
                "is_committed": False,
                "committed_class": None,
                "committed_label": None,
                "stable_label": "Gesture unavailable in demo mode",
                "stable_class": None,
                "status": "UNAVAILABLE (DEMO SAFE)"
            }

        if not hand_detected or prediction.get("class_id", -1) < 0:
            self.history.append((-1, "None", 0.0))
            self.consecutive_idle_frames += 1
            if self.consecutive_idle_frames >= self.rearm_frames:
                self.last_committed = None
                self.last_committed_class = None
                self.stable_gesture = None
                self.stable_class = None
                self.stable_label = "None"
            return {
                "is_committed": False,
                "committed_class": None,
                "committed_label": None,
                "stable_label": "None" if self.consecutive_idle_frames >= self.rearm_frames else self.stable_label,
                "stable_class": None if self.consecutive_idle_frames >= self.rearm_frames else self.stable_class,
                "status": "SEARCHING"
            }

        cid = prediction["class_id"]
        label = prediction["label"]
        conf = prediction.get("confidence", 0.0)

        self.history.append((cid, label, conf))

        # Discard low-confidence predictions or excluded classes
        valid_predictions = [
            c for c, l, p in self.history
            if p >= self.min_confidence and c >= 0 and (not self.demo_safe or c not in self.excluded_classes)
        ]

        if not valid_predictions:
            self.consecutive_idle_frames += 1
            if self.consecutive_idle_frames >= self.rearm_frames:
                self.last_committed = None
                self.last_committed_class = None
                self.stable_gesture = None
                self.stable_class = None
            return {
                "is_committed": False,
                "committed_class": None,
                "committed_label": None,
                "stable_label": "Unknown",
                "stable_class": None,
                "status": "DETECTING"
            }

        # Valid predictions present: reset idle frame counter
        self.consecutive_idle_frames = 0

        # Majority vote over valid predictions
        counts = Counter(valid_predictions)
        top_class, top_count = counts.most_common(1)[0]
        top_label = next(l for c, l, _ in self.history if c == top_class)

        if top_count >= self.min_stable_count:
            if self.demo_safe and (top_class in self.excluded_classes or str(top_label).lower() == "fuck"):
                return {
                    "is_committed": False,
                    "committed_class": None,
                    "committed_label": None,
                    "stable_label": "Gesture unavailable in demo mode",
                    "stable_class": None,
                    "status": "UNAVAILABLE (DEMO SAFE)"
                }

            self.stable_gesture = top_label
            self.stable_class = top_class
            self.stable_label = top_label

            # Check if this is a NEW stable gesture commit (suppress duplicates)
            if top_label != self.last_committed:
                self.last_committed = top_label
                self.last_committed_class = top_class
                self.last_commit_event = {"label": top_label, "class_id": top_class}
                return {
                    "is_committed": True,
                    "committed_class": top_class,
                    "committed_label": top_label,
                    "stable_label": top_label,
                    "stable_class": top_class,
                    "status": "STABLE (COMMITTED)"
                }
            else:
                # Already committed: suppress duplicate
                return {
                    "is_committed": False,
                    "committed_class": None,
                    "committed_label": None,
                    "stable_label": top_label,
                    "stable_class": top_class,
                    "status": "STABLE"
                }
        else:
            # Gesture not yet stable
            self.stable_gesture = None
            self.stable_class = top_class
            self.stable_label = f"{top_label} ({top_count}/{self.min_stable_count})"
            return {
                "is_committed": False,
                "committed_class": None,
                "committed_label": None,
                "stable_label": self.stable_label,
                "stable_class": top_class,
                "status": "DETECTING"
            }

    def reset(self):
        """Clears history and state."""
        self.history.clear()
        self.last_committed = None
        self.last_committed_class = None
        self.stable_gesture = None
        self.stable_class = None
        self.stable_label = "None"
        self.consecutive_idle_frames = 0
        self.last_commit_event = None
