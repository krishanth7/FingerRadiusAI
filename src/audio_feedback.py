"""
audio_feedback.py - Turn radius measurements into sound.

Built for accessibility first: someone who cannot see the dashboard can still
hear their hand open and close. A finger-pair distance maps to pitch, so
opening the hand raises the note, and the mapping is logarithmic because
human pitch perception is -- a linear Hz mapping sounds like it does nothing
at the top of the range and lurches at the bottom.

Playback is optional and degrades in a defined order:

1. ``sounddevice`` if it is installed and a device exists -- lowest latency.
2. ``simpleaudio`` as a fallback.
3. Neither: :meth:`AudioFeedback.play` becomes a no-op and
   :meth:`AudioFeedback.render_wav` still writes a file you can listen to
   afterwards. Nothing raises, because losing audio should not stop tracking.

Tone synthesis itself is pure NumPy and always available, so the mapping can
be tested anywhere -- including in CI with no sound card.
"""

from __future__ import annotations

import math
import os
import struct
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ToneMapper", "AudioFeedback", "SCALES"]

#: Semitone offsets within an octave. Snapping to a scale keeps a continuously
#: moving hand musical instead of producing a siren.
SCALES: Dict[str, List[int]] = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}


@dataclass
class ToneMapper:
    """Maps a radius in pixels to a frequency in hertz.

    Args:
        radius_min: Radius mapped to the lowest note. Values below clamp.
        radius_max: Radius mapped to the highest note.
        freq_min: Lowest frequency, in Hz. 220 is A3.
        freq_max: Highest frequency, in Hz. 880 is A5, two octaves up.
        scale: A key of :data:`SCALES`, or ``None`` for a continuous glide.

    Example:
        >>> mapper = ToneMapper(scale=None)
        >>> round(mapper.frequency(0))
        220
        >>> round(mapper.frequency(300))
        880
        >>> round(mapper.frequency(150))
        440
    """

    radius_min: float = 0.0
    radius_max: float = 300.0
    freq_min: float = 220.0
    freq_max: float = 880.0
    scale: Optional[str] = "pentatonic"

    def __post_init__(self) -> None:
        if self.radius_max <= self.radius_min:
            raise ValueError("radius_max must be greater than radius_min.")
        if self.freq_min <= 0 or self.freq_max <= self.freq_min:
            raise ValueError("Frequencies must be positive and increasing.")
        if self.scale is not None and self.scale not in SCALES:
            raise ValueError(
                f"Unknown scale {self.scale!r}. Available: {', '.join(SCALES)}, or None."
            )

    def normalise(self, radius: float) -> float:
        """Map a radius onto 0.0-1.0, clamped at both ends."""
        span = self.radius_max - self.radius_min
        return max(0.0, min(1.0, (float(radius) - self.radius_min) / span))

    def frequency(self, radius: float) -> float:
        """Return the frequency for a radius, in hertz.

        Interpolation is geometric, not linear: equal steps in radius give
        equal musical intervals, which is what the ear expects.
        """
        t = self.normalise(radius)
        freq = self.freq_min * (self.freq_max / self.freq_min) ** t
        if self.scale is None:
            return freq
        return self._snap(freq)

    def _snap(self, freq: float) -> float:
        """Move a frequency to the nearest note of the scale."""
        semitones = 12.0 * math.log2(freq / self.freq_min)
        octave, within = divmod(semitones, 12.0)
        degrees = SCALES[self.scale]
        nearest = min(degrees + [12], key=lambda d: abs(d - within))
        return self.freq_min * (2.0 ** ((octave * 12 + nearest) / 12.0))

    def note_name(self, radius: float) -> str:
        """Nearest note name, for display. A4 = 440 Hz."""
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        freq = self.frequency(radius)
        midi = int(round(69 + 12 * math.log2(freq / 440.0)))
        return f"{names[midi % 12]}{midi // 12 - 1}"


class AudioFeedback:
    """Sonifies radius values, live or to a file.

    Args:
        mapper: Radius-to-frequency mapping. A default one is made if omitted.
        sample_rate: Samples per second.
        volume: 0.0-1.0 master gain.
        enabled: Set False to make every method a no-op without removing the
            calls from the loop.

    Example:
        >>> audio = AudioFeedback()
        >>> tone = audio.synthesize(440.0, 0.01)
        >>> tone.dtype, len(tone)
        (dtype('float32'), 441)
    """

    def __init__(
        self,
        mapper: Optional[ToneMapper] = None,
        sample_rate: int = 44100,
        volume: float = 0.35,
        enabled: bool = True,
    ) -> None:
        self.mapper = mapper or ToneMapper()
        self.sample_rate = int(sample_rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.enabled = bool(enabled)

        self._phase = 0.0          # carried across chunks to avoid clicks
        self._backend = self._select_backend()
        self._history: List[Tuple[float, float]] = []   # (radius, frequency)

    # ------------------------------------------------------------------
    @staticmethod
    def _select_backend() -> str:
        """Pick the best available playback backend, or 'none'."""
        try:
            import sounddevice  # noqa: F401
            return "sounddevice"
        except Exception:
            pass
        try:
            import simpleaudio  # noqa: F401
            return "simpleaudio"
        except Exception:
            pass
        return "none"

    @property
    def backend(self) -> str:
        """Which playback backend is in use: sounddevice, simpleaudio or none."""
        return self._backend

    @property
    def can_play(self) -> bool:
        """True when live playback is actually possible."""
        return self.enabled and self._backend != "none"

    # ------------------------------------------------------------------
    def synthesize(
        self,
        frequency: float,
        duration: float,
        *,
        continuous: bool = False,
    ) -> np.ndarray:
        """Render a sine tone as float32 samples in [-1, 1].

        Args:
            frequency: Hertz.
            duration: Seconds.
            continuous: Carry the phase over from the previous call. Chunks
                played back to back must do this -- restarting the sine at
                zero every chunk produces an audible click at each boundary.
        """
        count = max(1, int(self.sample_rate * duration))
        step = 2.0 * math.pi * frequency / self.sample_rate
        start = self._phase if continuous else 0.0
        phases = start + step * np.arange(count, dtype=np.float64)
        if continuous:
            self._phase = float((start + step * count) % (2.0 * math.pi))

        wave_data = np.sin(phases) * self.volume

        # Short fades at both ends stop the speaker cone snapping.
        fade = min(count // 8, int(self.sample_rate * 0.005))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade)
            wave_data[:fade] *= ramp
            wave_data[-fade:] *= ramp[::-1]

        return wave_data.astype(np.float32)

    def update(self, radius: float, duration: float = 0.05) -> Optional[float]:
        """Sonify one radius reading. Returns the frequency played, or None.

        Safe to call every frame: when no backend exists it records the
        mapping and returns without touching any device.
        """
        if not self.enabled:
            return None
        freq = self.mapper.frequency(radius)
        self._history.append((float(radius), freq))
        if self._backend == "none":
            return freq
        self.play(self.synthesize(freq, duration, continuous=True))
        return freq

    def play(self, samples: np.ndarray) -> None:
        """Send samples to the speaker. A no-op when no backend is available.

        Playback failure is swallowed deliberately: a missing sound card must
        not take down hand tracking.
        """
        if not self.can_play:
            return
        try:
            if self._backend == "sounddevice":
                import sounddevice as sd
                sd.play(samples, self.sample_rate, blocking=False)
            else:
                import simpleaudio as sa
                pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
                sa.play_buffer(pcm, 1, 2, self.sample_rate)
        except Exception as error:  # pragma: no cover - hardware dependent
            print(f"[Audio] Playback failed ({error}); disabling audio output.")
            self._backend = "none"

    # ------------------------------------------------------------------
    def render_wav(
        self,
        radii: Sequence[float],
        path: str = "session_audio.wav",
        note_duration: float = 0.08,
    ) -> str:
        """Render a sequence of radii to a WAV file and return its path.

        This is the path that always works. With no sound card you can still
        export the session and listen to it afterwards, and it is also how the
        mapping gets tested without hardware.

        Raises:
            ValueError: If no radii are supplied.
        """
        if not len(radii):
            raise ValueError("No radius values to render.")

        saved_phase = self._phase
        self._phase = 0.0
        chunks = [
            self.synthesize(self.mapper.frequency(r), note_duration, continuous=True)
            for r in radii
        ]
        self._phase = saved_phase

        samples = np.concatenate(chunks)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(path, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm.tobytes())

        seconds = len(samples) / self.sample_rate
        print(f"[Audio] {len(radii)} values -> {path} ({seconds:.1f}s)")
        return path

    def render_history(self, path: str = "session_audio.wav", note_duration: float = 0.08) -> str:
        """Render everything passed to :meth:`update` so far."""
        return self.render_wav([r for r, _ in self._history], path, note_duration)

    def describe(self) -> str:
        """One line on the audio configuration, for the console banner."""
        scale = self.mapper.scale or "continuous"
        state = self._backend if self.can_play else "file export only"
        return (
            f"audio {state}: {self.mapper.freq_min:.0f}-{self.mapper.freq_max:.0f}Hz, "
            f"{scale} scale"
        )
