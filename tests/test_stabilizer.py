"""
tests/test_stabilizer.py
------------------------
Verifies majority voting, duplicate gesture suppression, reset behavior,
and confidence filtering in TemporalStabilizer.
"""

import pytest
from recognition.stabilizer import TemporalStabilizer

@pytest.fixture
def stabilizer():
    return TemporalStabilizer(window_size=12, min_confidence=0.70, min_stable_count=7)

def test_stabilizer_majority_voting(stabilizer):
    pred_A = {"class_id": 0, "label": "A", "confidence": 0.95, "is_confident": True}
    pred_B = {"class_id": 1, "label": "B", "confidence": 0.90, "is_confident": True}

    # Feed 6 'A's -> not yet stable (requires 7)
    for _ in range(6):
        res = stabilizer.update(pred_A, hand_detected=True)
        assert not res["is_committed"]
        assert "STABLE" not in res["status"]

    # 7th 'A' -> reaches min_stable_count=7 -> commits!
    res7 = stabilizer.update(pred_A, hand_detected=True)
    assert res7["is_committed"]
    assert res7["committed_label"] == "A"
    assert res7["committed_class"] == 0
    assert "STABLE" in res7["status"]

def test_stabilizer_duplicate_gesture_suppression(stabilizer):
    pred_A = {"class_id": 0, "label": "A", "confidence": 0.98, "is_confident": True}
    
    commit_count = 0
    # Stream 20 consecutive 'A's
    for _ in range(20):
        res = stabilizer.update(pred_A, hand_detected=True)
        if res["is_committed"]:
            commit_count += 1

    # Must commit exactly once!
    assert commit_count == 1, f"Expected 1 commit, got {commit_count}"

def test_stabilizer_reset(stabilizer):
    pred_A = {"class_id": 0, "label": "A", "confidence": 0.98, "is_confident": True}
    for _ in range(8):
        stabilizer.update(pred_A, hand_detected=True)
    assert stabilizer.last_committed_class == 0

    stabilizer.reset()
    assert stabilizer.last_committed_class is None
    assert len(stabilizer.history) == 0
    assert stabilizer.stable_class is None

def test_stabilizer_low_confidence_discard(stabilizer):
    pred_low = {"class_id": 2, "label": "C", "confidence": 0.50, "is_confident": False}
    for _ in range(15):
        res = stabilizer.update(pred_low, hand_detected=True)
        assert not res["is_committed"]
        assert res["status"] == "DETECTING" or res["status"] == "SEARCHING"

def test_stable_u_one_commit(stabilizer):
    """Verifies that holding U until stable emits exactly one commit."""
    pred_U = {"class_id": 20, "label": "U", "confidence": 0.95, "is_confident": True}
    commits = []
    for _ in range(7):
        res = stabilizer.update(pred_U, hand_detected=True)
        if res["is_committed"]:
            commits.append(res)

    assert len(commits) == 1
    assert commits[0]["committed_label"] == "U"
    assert commits[0]["committed_class"] == 20

def test_continuous_u_no_duplicate_commits(stabilizer):
    """Verifies that holding U continuously for 40 frames emits NO duplicate commits."""
    pred_U = {"class_id": 20, "label": "U", "confidence": 0.95, "is_confident": True}
    commit_count = 0
    for _ in range(40):
        res = stabilizer.update(pred_U, hand_detected=True)
        if res["is_committed"]:
            commit_count += 1

    assert commit_count == 1, f"Expected exactly 1 commit for continuous U, got {commit_count}"

def test_u_unknown_u_two_separate_commits(stabilizer):
    """
    Verifies state machine re-arm sequence:
    U (committed) -> continuous U (no duplicate) -> Unknown (re-arm) -> U (committed again).
    """
    pred_U = {"class_id": 20, "label": "U", "confidence": 0.95, "is_confident": True}
    pred_unknown = {"class_id": -1, "label": "Unknown", "confidence": 0.0, "is_confident": False}

    all_commits = []

    # 1. Feed U for 8 frames -> commit 1
    for _ in range(8):
        res = stabilizer.update(pred_U, hand_detected=True)
        if res["is_committed"]:
            all_commits.append(res["committed_label"])
    assert len(all_commits) == 1
    assert all_commits[0] == "U"

    # 2. Continue holding U for 10 frames -> NO new commits
    for _ in range(10):
        res = stabilizer.update(pred_U, hand_detected=True)
        if res["is_committed"]:
            all_commits.append(res["committed_label"])
    assert len(all_commits) == 1

    # 3. Stream Unknown / No Hand for 10 frames -> stabilizer re-arms
    for _ in range(10):
        res = stabilizer.update(pred_unknown, hand_detected=False)
        assert not res["is_committed"]
    assert stabilizer.last_committed is None

    # 4. Stream U again for 8 frames -> commit 2
    for _ in range(8):
        res = stabilizer.update(pred_U, hand_detected=True)
        if res["is_committed"]:
            all_commits.append(res["committed_label"])

    assert len(all_commits) == 2
    assert all_commits == ["U", "U"]
