"""
Tests for the features added on top of the original tracker.

These run with no camera, no display and no GPU. Where a feature genuinely
needs hardware the test says so and skips, rather than passing on a mock and
implying coverage that does not exist.
"""

from __future__ import annotations

import math
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_feedback import SCALES, AudioFeedback, ToneMapper
from src.gestures import Gesture, GestureRecognizer
from src.kalman import KalmanFilter1D, KalmanLandmarkSet, KalmanPoint2D
from src.onnx_backend import available_providers, describe_runtime, preferred_providers
from src.themes import THEME_NAMES, THEMES, apply_theme, current_theme, next_theme, validate
from src.utils import COLORS, ExponentialMovingAverage
from tests import synthetic_hands as hands


# ======================================================================
# Gestures
# ======================================================================
class TestGestureRecognition:
    @pytest.fixture
    def recognizer(self):
        return GestureRecognizer(hold_frames=1)

    @pytest.mark.parametrize(
        "builder, expected",
        [
            (hands.fist, Gesture.FIST),
            (hands.open_palm, Gesture.OPEN_PALM),
            (hands.peace, Gesture.PEACE),
            (hands.pointing, Gesture.POINTING),
            (hands.thumbs_up, Gesture.THUMBS_UP),
            (hands.thumbs_down, Gesture.THUMBS_DOWN),
            (hands.ok_sign, Gesture.OK),
            (hands.pinch, Gesture.PINCH),
        ],
    )
    def test_each_gesture_is_recognised(self, recognizer, builder, expected):
        assert recognizer.classify(builder()).gesture == expected

    @pytest.mark.parametrize("rotation", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_recognition_survives_rotation(self, recognizer, rotation):
        # Comparing y coordinates -- the usual shortcut -- fails here. Working
        # from wrist distances is what makes this pass.
        assert recognizer.classify(hands.peace(rotation=rotation)).gesture == Gesture.PEACE

    @pytest.mark.parametrize("scale", [40.0, 90.0, 180.0, 300.0])
    def test_recognition_survives_hand_size(self, recognizer, scale):
        # A hand near the camera and one far away must classify the same, which
        # is why every threshold is a ratio rather than a pixel count.
        assert recognizer.classify(hands.ok_sign(scale=scale)).gesture == Gesture.OK

    def test_fist_is_not_mistaken_for_a_pinch(self, recognizer):
        # In a fist the curled index tip comes to rest beside the tucked thumb,
        # so a pure distance test calls it a pinch. It needs an extended index.
        assert recognizer.classify(hands.fist()).gesture == Gesture.FIST

    def test_ok_outranks_pinch(self, recognizer):
        # Both touch thumb to index; the other three fingers decide.
        assert recognizer.classify(hands.ok_sign()).gesture == Gesture.OK
        assert recognizer.classify(hands.pinch()).gesture == Gesture.PINCH

    def test_finger_states_report_each_finger(self, recognizer):
        states = recognizer.finger_states(hands.peace())
        assert states == {"thumb": False, "index": True, "middle": True,
                          "ring": False, "pinky": False}

    def test_hold_frames_suppresses_flicker(self):
        recognizer = GestureRecognizer(hold_frames=3)
        # A single stray frame must not change the reported gesture.
        for _ in range(5):
            recognizer.update(hands.peace())
        assert recognizer.update(hands.fist()).gesture == Gesture.PEACE
        recognizer.update(hands.fist())
        assert recognizer.update(hands.fist()).gesture == Gesture.FIST

    def test_no_hand_resets_state(self):
        recognizer = GestureRecognizer(hold_frames=1)
        recognizer.update(hands.peace())
        assert recognizer.update(None).gesture == Gesture.UNKNOWN

    def test_too_few_landmarks_is_rejected(self, recognizer):
        with pytest.raises(ValueError, match="21 landmarks"):
            recognizer.classify([(0, 0)] * 5)

    def test_invalid_hold_frames_is_rejected(self):
        with pytest.raises(ValueError, match="hold_frames"):
            GestureRecognizer(hold_frames=0)


# ======================================================================
# Kalman
# ======================================================================
class TestKalman:
    @staticmethod
    def _signals(n=600):
        t = np.arange(n)
        step = np.full(n, 120.0)
        step[200:230] = np.linspace(120, 240, 30)
        step[230:] = 240.0
        return {
            "sine": 150 + 60 * np.sin(t / 22.0),
            "move-and-hold": step,
            "drift": 150 + 0.08 * t + 3 * np.sin(t / 3.0),
        }

    @staticmethod
    def _rmse(a, b):
        return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))

    def test_converges_on_a_constant(self):
        f = KalmanFilter1D()
        for _ in range(200):
            f.update(42.0)
        assert f.position == pytest.approx(42.0, abs=1e-6)
        assert f.velocity == pytest.approx(0.0, abs=1e-6)

    def test_first_update_seeds_rather_than_ramping(self):
        # Starting from zero would take many frames to reach the first real
        # reading, which looks like the hand flying in from the corner.
        f = KalmanFilter1D()
        assert f.update(500.0) == pytest.approx(500.0)

    def test_estimates_velocity(self):
        f = KalmanFilter1D()
        for i in range(120):
            f.update(float(i) * 3.0)
        assert f.velocity == pytest.approx(3.0, rel=0.1)

    @pytest.mark.parametrize("name", ["sine", "move-and-hold", "drift"])
    def test_beats_the_ema_it_replaces(self, name):
        truth = self._signals()[name]
        noisy = truth + np.random.default_rng(1).normal(0, 8.0, len(truth))

        ema = ExponentialMovingAverage(0.35)      # the shipped default
        ema_out = [ema.update(v) for v in noisy]
        kal = KalmanFilter1D()                     # the shipped default
        kal_out = [kal.update(v) for v in noisy]

        assert self._rmse(kal_out, truth) < self._rmse(ema_out, truth)

    def test_both_filters_beat_the_raw_signal(self):
        truth = self._signals()["sine"]
        noisy = truth + np.random.default_rng(1).normal(0, 8.0, len(truth))
        kal = KalmanFilter1D()
        assert self._rmse([kal.update(v) for v in noisy], truth) < self._rmse(noisy, truth)

    def test_point_tracks_constant_velocity(self):
        p = KalmanPoint2D()
        for i in range(60):
            p.update((100 + i, 200))
        assert p.speed == pytest.approx(1.0, rel=0.15)

    def test_landmark_set_shape_is_enforced(self):
        bank = KalmanLandmarkSet()
        assert len(bank.update([(i, i, 0.0) for i in range(21)])) == 21
        with pytest.raises(ValueError, match="Expected 21"):
            bank.update([(0, 0, 0)])

    def test_reset_clears_state(self):
        f = KalmanFilter1D()
        for _ in range(50):
            f.update(100.0)
        f.reset()
        assert f.update(7.0) == pytest.approx(7.0)

    @pytest.mark.parametrize("kwargs", [{"process_noise": 0}, {"measurement_noise": -1}])
    def test_invalid_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            KalmanFilter1D(**kwargs)


# ======================================================================
# Themes
# ======================================================================
class TestThemes:
    def teardown_method(self):
        apply_theme("corporate")

    def test_every_theme_defines_every_key(self):
        # A partial theme leaves the previous theme's colours in place, giving
        # a palette that is neither one nor the other.
        validate()

    def test_there_are_four_themes(self):
        assert set(THEME_NAMES) == {"corporate", "cyberpunk", "minimal", "retro"}

    @pytest.mark.parametrize("name", ["corporate", "cyberpunk", "minimal", "retro"])
    def test_applying_mutates_the_shared_palette(self, name):
        # Modules import COLORS by name, so a theme must mutate that dict in
        # place; rebinding it would leave every importer on the old palette.
        apply_theme(name)
        assert current_theme() == name
        assert COLORS["accent"] == THEMES[name]["accent"]
        assert COLORS is not THEMES[name]

    def test_colours_are_valid_bgr_triples(self):
        for name, palette in THEMES.items():
            for key, value in palette.items():
                assert len(value) == 3, f"{name}.{key}"
                assert all(0 <= c <= 255 for c in value), f"{name}.{key} = {value}"

    def test_cycling_returns_to_the_start(self):
        apply_theme("corporate")
        seen = [next_theme() for _ in THEME_NAMES]
        assert seen[-1] == "corporate"
        assert len(set(seen)) == len(THEME_NAMES)

    def test_unknown_theme_lists_the_real_ones(self):
        with pytest.raises(KeyError, match="cyberpunk"):
            apply_theme("does-not-exist")


# ======================================================================
# Audio
# ======================================================================
class TestAudioFeedback:
    def test_endpoints_and_midpoint(self):
        m = ToneMapper(scale=None)
        assert m.frequency(0) == pytest.approx(220.0)
        assert m.frequency(300) == pytest.approx(880.0)
        # Geometric, not linear: the midpoint is one octave up, not 550 Hz.
        assert m.frequency(150) == pytest.approx(440.0)

    def test_out_of_range_radii_clamp(self):
        m = ToneMapper(scale=None)
        assert m.frequency(-500) == pytest.approx(220.0)
        assert m.frequency(9999) == pytest.approx(880.0)

    def test_mapping_is_monotonic(self):
        m = ToneMapper(scale=None)
        freqs = [m.frequency(r) for r in range(0, 301, 10)]
        assert all(b >= a for a, b in zip(freqs, freqs[1:]))

    @pytest.mark.parametrize("scale", list(SCALES))
    def test_snapped_notes_stay_in_range(self, scale):
        m = ToneMapper(scale=scale)
        for radius in range(0, 301, 15):
            assert 219.0 <= m.frequency(radius) <= 881.0

    def test_scale_snapping_quantises(self):
        continuous = {ToneMapper(scale=None).frequency(r) for r in range(0, 301, 5)}
        snapped = {ToneMapper(scale="pentatonic").frequency(r) for r in range(0, 301, 5)}
        assert len(snapped) < len(continuous)

    def test_synthesized_tone_has_the_requested_frequency(self, tmp_path):
        # The real check: render, read the file back, and look at its spectrum.
        audio = AudioFeedback()
        path = str(tmp_path / "tone.wav")
        audio.render_wav([150.0] * 8, path, note_duration=0.2)

        with wave.open(path) as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            rate = handle.getframerate()
            data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)

        segment = data[: rate // 4].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        peak = np.fft.rfftfreq(len(segment), 1 / rate)[spectrum.argmax()]
        assert peak == pytest.approx(AudioFeedback().mapper.frequency(150.0), rel=0.05)

    def test_works_without_a_sound_device(self):
        # No hardware here, so this is the path CI actually exercises.
        audio = AudioFeedback()
        assert audio.update(120.0) is not None
        audio.play(audio.synthesize(440.0, 0.01))   # must not raise

    def test_disabled_is_a_no_op(self):
        assert AudioFeedback(enabled=False).update(120.0) is None

    def test_empty_render_is_rejected(self):
        with pytest.raises(ValueError, match="No radius values"):
            AudioFeedback().render_wav([])

    @pytest.mark.parametrize("kwargs", [
        {"radius_min": 100, "radius_max": 100},
        {"freq_min": 0},
        {"scale": "not-a-scale"},
    ])
    def test_invalid_mapper_arguments_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ToneMapper(**kwargs)


# ======================================================================
# ONNX
# ======================================================================
class TestOnnxBackend:
    def test_describe_runtime_is_a_string(self):
        assert isinstance(describe_runtime(), str)

    def test_cpu_provider_is_always_present_when_installed(self):
        providers = available_providers()
        if not providers:
            pytest.skip("onnxruntime is not installed in this environment.")
        assert "CPUExecutionProvider" in providers

    def test_preferred_order_is_a_permutation_of_available(self):
        if not available_providers():
            pytest.skip("onnxruntime is not installed in this environment.")
        assert sorted(preferred_providers()) == sorted(available_providers())

    def test_missing_model_names_the_path(self):
        from src.onnx_backend import OnnxLandmarkBackend, onnxruntime_available
        if not onnxruntime_available():
            pytest.skip("onnxruntime is not installed in this environment.")
        with pytest.raises(FileNotFoundError, match="nope.onnx"):
            OnnxLandmarkBackend("nope.onnx")
