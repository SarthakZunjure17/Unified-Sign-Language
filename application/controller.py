"""
application/controller.py
-------------------------
Coordinates application modes (Text vs Calculator), input dispatching,
speech requests, and keyboard controls.
"""

from typing import Dict, Any, Optional
from .text_mode import TextMode
from .calculator import SafeCalculator
from .speech import SpeechEngine

class AppController:
    """
    Central controller handling state transitions, gesture commits, and keyboard shortcuts.
    """
    def __init__(self, default_mode: str = "TEXT", demo_safe: bool = True):
        self.mode = default_mode.upper() if default_mode.upper() in ["TEXT", "CALCULATOR"] else "TEXT"
        self.demo_safe = demo_safe
        self.text_mode = TextMode(demo_safe=demo_safe)
        self.calculator = SafeCalculator(demo_safe=demo_safe)
        self.speech = SpeechEngine(demo_safe=demo_safe)

        self.last_spoken_text = ""
        self.status_message = "Ready"

    def toggle_mode(self) -> str:
        """Toggles between TEXT and CALCULATOR mode."""
        if self.mode == "TEXT":
            self.mode = "CALCULATOR"
        else:
            self.mode = "TEXT"
        self.status_message = f"Switched to {self.mode} Mode"
        return self.mode

    def handle_committed_gesture(self, class_id: int, label: str):
        """Dispatches committed gesture to the active mode handler."""
        # Demo Safe Policy: Never commit Class 42
        if self.demo_safe and (class_id == 42 or str(label).lower() == "fuck" or class_id < 0):
            self.status_message = "Gesture unavailable in demo mode"
            return

        if self.mode == "TEXT":
            self.text_mode.append_gesture(class_id, label)
            self.status_message = f"Committed: {label}"
        elif self.mode == "CALCULATOR":
            self.calculator.add_gesture(class_id, label)
            if class_id == 36:
                self.status_message = "Calculated Result"
            else:
                self.status_message = f"Added Digit: {label}"

    def speak_current(self):
        """Speaks the active sentence or calculation result."""
        if self.mode == "TEXT":
            text_to_speak = self.text_mode.get_sentence()
        else:
            calc_res = self.calculator.result
            text_to_speak = f"Result is {calc_res}" if calc_res and not self.calculator.has_error else self.calculator.expression

        if self.demo_safe and text_to_speak:
            import re
            text_to_speak = re.sub(r'(?i)\bfuck\b', '', text_to_speak).strip()

        if text_to_speak:
            self.speech.speak(text_to_speak)
            self.status_message = "Speaking..."
        else:
            self.status_message = "Nothing to speak"

    def handle_key(self, key_code: int) -> bool:
        """
        Processes OpenCV waitKey codes:
        Returns:
            bool: False if application should exit (Q/ESC), True to continue.
        """
        if key_code in [ord('q'), ord('Q'), 27]:
            return False  # Exit

        elif key_code in [ord('m'), ord('M')]:
            self.toggle_mode()

        elif key_code in [ord('c'), ord('C')]:
            if self.mode == "TEXT":
                self.text_mode.clear()
                self.status_message = "Sentence Cleared"
            else:
                self.calculator.clear()
                self.status_message = "Expression Cleared"

        elif key_code in [ord('s'), ord('S')]:
            self.speak_current()

        elif key_code in [8, 127]:  # Backspace / Delete
            if self.mode == "TEXT":
                self.text_mode.backspace()
            else:
                self.calculator.backspace()
            self.status_message = "Backspace"

        # Calculator-specific operator inputs via keyboard
        elif self.mode == "CALCULATOR":
            if key_code in [ord('+'), ord('-'), ord('*'), ord('/'), ord('.')]:
                char = chr(key_code)
                self.calculator.add_token(char)
                self.status_message = f"Operator: {char}"
            elif key_code in [ord('='), 13]:  # '=' or Enter key
                self.calculator.calculate()
                self.status_message = "Calculated"

        return True

    def get_display_state(self) -> Dict[str, Any]:
        """Returns state payload for HUD rendering."""
        return {
            "mode": self.mode,
            "sentence": self.text_mode.get_sentence(),
            "expression": self.calculator.expression,
            "calc_result": self.calculator.result,
            "calc_has_error": self.calculator.has_error,
            "status_message": self.status_message,
            "speech_available": self.speech.is_available,
            "demo_safe": self.demo_safe
        }
