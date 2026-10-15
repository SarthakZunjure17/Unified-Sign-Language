"""
tests/test_labels.py
--------------------
Verifies label vocabulary structure, 44-class count, and Class 42 == "Fuck".
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_labels_file_exists():
    labels_path = PROJECT_ROOT / "metadata" / "labels.json"
    assert labels_path.exists(), f"Missing labels.json at {labels_path}"

def test_exactly_44_classes():
    labels_path = PROJECT_ROOT / "metadata" / "labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 44, f"Expected exactly 44 classes, got {len(data)}"

def test_class_42_is_fuck():
    labels_path = PROJECT_ROOT / "metadata" / "labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check item with class_id 42
    c42_item = next((item for item in data if item["class_id"] == 42), None)
    assert c42_item is not None, "Class 42 not found in labels.json"
    assert c42_item["label"] == "Fuck", f"Class 42 label is '{c42_item['label']}', MUST be 'Fuck'"

def test_all_classes_unique_and_sequential():
    labels_path = PROJECT_ROOT / "metadata" / "labels.json"
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cids = [item["class_id"] for item in data]
    assert len(set(cids)) == 44, "Duplicate class IDs found in labels.json"
    assert sorted(cids) == list(range(44)), "Class IDs are not sequential 0..43"
