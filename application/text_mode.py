"""
application/text_mode.py
------------------------
Manages sentence formation, vocabulary mapping, and editing in Text Mode.
"""

from typing import Dict, Any

class TextMode:
    """
    Translates committed gestures into structured words and sentences.
    Preserves all 44 classes.
    """
    SPECIAL_GESTURE_MAP = {
        36: " ",             # Best of Luck (space)
        37: " You ",         # You
        38: " I ",           # I/Me
        39: " like ",        # Like
        40: " remember ",    # Remember
        41: " love ",        # Love
        42: " fuck ",        # Fuck (Class 42 preserved)
        43: " I love you "   # I love you
    }

    def __init__(self, demo_safe: bool = True):
        self.sentence = ""
        self.demo_safe = demo_safe

    def append_gesture(self, class_id: int, label: str) -> str:
        """
        Appends gesture based on class ID:
        0-25: Alphabets A-Z
        26-35: Digits 0-9
        36-43: Defined vocabulary mappings
        In demo-safe mode, Class 42 is strictly ignored.
        """
        if self.demo_safe and (class_id == 42 or str(label).lower() == "fuck" or class_id < 0):
            return self.sentence

        if class_id in self.SPECIAL_GESTURE_MAP:
            token = self.SPECIAL_GESTURE_MAP[class_id]
            # Clean up double spaces if any
            if token.startswith(" ") and self.sentence.endswith(" "):
                token = token.lstrip()
            self.sentence += token
        elif 0 <= class_id <= 35:
            # Single character (A-Z, 0-9)
            self.sentence += label
        else:
            self.sentence += label

        return self.sentence

    def backspace(self) -> str:
        """Removes the last character or word."""
        if self.sentence:
            # If ends with whitespace, strip it first, else remove 1 char
            self.sentence = self.sentence[:-1]
        return self.sentence

    def clear(self) -> str:
        """Clears the sentence."""
        self.sentence = ""
        return self.sentence

    def get_sentence(self) -> str:
        """Returns the current accumulated text."""
        return self.sentence
