"""
application/speech.py
---------------------
Text-to-speech engine wrapper using pyttsx3.
Handles initialization failures gracefully and executes speech in background
threads to prevent blocking the webcam loop.
"""

import threading
from typing import Optional

class SpeechEngine:
    """
    Non-blocking Text-To-Speech synthesizer.
    """
    def __init__(self, rate: int = 150, volume: float = 1.0, demo_safe: bool = True):
        self.is_available = False
        self._lock = threading.Lock()
        self.rate = rate
        self.volume = volume
        self.demo_safe = demo_safe
        self._init_engine()

    def _init_engine(self):
        """Attempts to initialize pyttsx3."""
        try:
            import pyttsx3
            # Test initialization
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self.is_available = True
            print("SpeechEngine: pyttsx3 initialized successfully.")
        except Exception as e:
            self.is_available = False
            print(f"SpeechEngine: Warning - Text-to-speech unavailable ({e}). Continuing without audio.")

    def speak(self, text: str, block: bool = False):
        """
        Synthesizes speech. Default is non-blocking (background thread).
        In demo-safe mode, drops or sanitizes excluded offensive words.
        """
        clean_text = text.strip()
        if not clean_text or not self.is_available:
            return

        if self.demo_safe:
            import re
            clean_text = re.sub(r'(?i)\bfuck\b', '', clean_text).strip()
            if not clean_text:
                return

        if block:
            self._do_speak(clean_text)
        else:
            t = threading.Thread(target=self._do_speak, args=(clean_text,), daemon=True)
            t.start()

    def _do_speak(self, text: str):
        """Internal speech execution with thread synchronization."""
        with self._lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"SpeechEngine: Error speaking '{text}': {e}")
