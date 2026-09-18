"""
End-to-end test: a video file in, tracking and exports out.

There is no camera here, so a clip is synthesised and fed through the same
:class:`~src.video_source.VideoSource` the application uses. MediaPipe runs
for real on every frame -- it simply finds no hands in a synthetic clip, which
is the honest result and still exercises the whole path: source, tracker,
radius calculation, gesture recognition, CSV, dashboard and audio export.

What this does NOT prove is landmark accuracy on a real hand. That needs a
camera and a person, and no amount of synthetic video substitutes for it.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_source import VideoSource


def mediapipe_reason() -> str:
    """Return why MediaPipe cannot run here, or an empty string if it can.

    ``import mediapipe`` succeeds on a bare machine, but building a
    HandLandmarker dlopens the system GL stack, so the failure arrives later
    and as an OSError rather than an ImportError:

        OSError: libEGL.so.1: cannot open shared object file

    A stock GitHub Actions ubuntu runner has no libEGL, so this has to be a
    skip with a readable reason rather than an opaque crash. Installing
    ``libegl1`` and ``libgl1`` -- which the workflow now does -- makes the
    test run for real instead of skipping.
    """
    try:
        from src.hand_tracker import HandTracker

        HandTracker(max_num_hands=1).release()
    except OSError as error:
        return f"MediaPipe cannot initialise: {error}. Install libegl1 and libgl1."
    except ImportError as error:  # pragma: no cover - mediapipe absent
        return f"MediaPipe is not installed: {error}"
    except FileNotFoundError as error:  # pragma: no cover - model absent
        return f"Hand landmark model missing: {error}"
    return ""


#: Evaluated once at import; building a landmarker twice per test is wasteful.
MEDIAPIPE_REASON = mediapipe_reason()
requires_mediapipe = pytest.mark.skipif(
    bool(MEDIAPIPE_REASON), reason=MEDIAPIPE_REASON or "MediaPipe is available"
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short synthetic clip with a moving shape, written to disk."""
    path = str(tmp_path_factory.mktemp("video") / "clip.mp4")
    width, height, fps, frames = 320, 240, 20.0, 40
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(frames):
        frame = np.full((height, width, 3), 30, np.uint8)
        cv2.circle(frame, (40 + i * 6, 120), 30, (90, 160, 220), -1)
        writer.write(frame)
    writer.release()
    assert os.path.getsize(path) > 0
    return path


class TestVideoSource:
    def test_reports_file_metadata(self, clip):
        with VideoSource(clip, realtime=False, loop=False) as source:
            assert source.is_file
            assert source.info.frame_count == 40
            assert source.info.fps == pytest.approx(20.0)
            assert source.info.duration_seconds == pytest.approx(2.0)

    def test_reads_every_frame_then_stops(self, clip):
        with VideoSource(clip, realtime=False, loop=False) as source:
            count = 0
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                assert frame.shape == (240, 320, 3)
                count += 1
        assert count == 40

    def test_looping_does_not_stop(self, clip):
        with VideoSource(clip, realtime=False, loop=True) as source:
            for _ in range(95):          # more than twice the clip
                ok, _ = source.read()
                assert ok

    def test_seek_moves_the_position(self, clip):
        with VideoSource(clip, realtime=False, loop=False) as source:
            source.seek_fraction(0.5)
            source.read()
            assert 15 <= source.position <= 25

    def test_pause_freezes_and_step_advances(self, clip):
        with VideoSource(clip, realtime=False, loop=False) as source:
            source.read()
            source.toggle_pause()
            source.read()
            held = source.position
            source.read()
            assert source.position == held
            source.step()
            source.read()
            assert source.position > held

    def test_a_missing_file_says_so(self):
        with pytest.raises(FileNotFoundError, match="absent.mp4"):
            VideoSource("absent.mp4")

    def test_digit_strings_are_camera_indices(self):
        # "0" from the command line has to mean camera 0, not a file named "0".
        with pytest.raises((RuntimeError, FileNotFoundError)) as excinfo:
            VideoSource("999")
        assert "camera" in str(excinfo.value).lower()


class TestFullPipeline:
    @requires_mediapipe
    def test_clip_runs_through_tracker_and_exports(self, clip, tmp_path):
        """Drive the whole chain over a file, exactly as main.py does."""
        from src.audio_feedback import AudioFeedback
        from src.gestures import GestureRecognizer
        from src.hand_tracker import HandTracker
        from src.kalman import KalmanLandmarkSet
        from src.radius_calculator import RadiusCalculator
        from src.utils import CSVExporter, FPSCounter

        tracker = HandTracker(max_num_hands=2)
        calculators = [RadiusCalculator() for _ in range(2)]
        recognizers = [GestureRecognizer() for _ in range(2)]
        banks = [KalmanLandmarkSet() for _ in range(2)]
        exporter = CSVExporter()
        fps = FPSCounter()
        audio = AudioFeedback()

        frames = 0
        with VideoSource(clip, realtime=False, loop=False) as source:
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                frames += 1
                fps.tick()

                for index in range(tracker.process(frame)):
                    lm3 = tracker.get_landmarks_3d(index)
                    if lm3 is None:
                        continue
                    filtered = banks[index].update(lm3)
                    lm = [(x, y) for x, y, _ in filtered]
                    radii, wrist, status, _ = calculators[index].compute(lm, filtered, True)
                    gesture = recognizers[index].update(lm)
                    audio.update(radii.get("Thumb-Index", 0.0))
                    exporter.record(radii, f"{status}|{gesture.gesture}")

                # Drawing must not raise even with no hands in frame.
                tracker.draw_all(frame, True)

        tracker.release()
        assert frames == 40

        # The clip has no hands, so no rows -- and the exporter must say so
        # rather than writing an empty file or crashing.
        csv_path = str(tmp_path / "out.csv")
        exporter.export(csv_path)

        # With rows of its own the export chain has to work end to end.
        for i in range(30):
            exporter.record({"Thumb-Index": 100.0 + i}, "Open|Peace")
        exporter.export(csv_path)
        assert os.path.exists(csv_path)

        from src.dashboard import build_dashboard
        html = build_dashboard(csv_path, str(tmp_path / "report.html"))
        assert os.path.getsize(html) > 100_000        # Plotly is inlined
        assert "plotly-graph-div" in open(html, encoding="utf-8").read()

        wav = audio.render_wav([100.0, 200.0, 300.0], str(tmp_path / "out.wav"))
        assert os.path.getsize(wav) > 1000

    def test_export_chain_without_mediapipe(self, tmp_path):
        """CSV, dashboard and audio export, with no tracking involved.

        Deliberately not gated on MediaPipe: these are the parts that must keep
        working on a machine with no GL stack, and gating them would have hidden
        a regression behind a skip.
        """
        from src.audio_feedback import AudioFeedback
        from src.dashboard import build_dashboard
        from src.utils import CSVExporter

        exporter = CSVExporter()
        for i in range(40):
            exporter.record({"Thumb-Index": 100.0 + i, "Index-Middle": 60.0 + i / 2},
                            "Open|Peace")
        csv_path = str(tmp_path / "chain.csv")
        exporter.export(csv_path)
        assert os.path.exists(csv_path)

        html = build_dashboard(csv_path, str(tmp_path / "chain.html"))
        assert os.path.getsize(html) > 100_000
        assert "plotly-graph-div" in open(html, encoding="utf-8").read()

        wav = AudioFeedback().render_wav([80.0, 160.0, 240.0], str(tmp_path / "chain.wav"))
        assert os.path.getsize(wav) > 1000


class TestMainCLI:
    def test_parses_defaults(self):
        from main import parse_args
        args = parse_args([])
        assert args.source == "0"
        assert args.filter == "kalman"
        assert args.theme == "corporate"

    def test_parses_a_full_command_line(self):
        from main import parse_args
        args = parse_args(
            ["--source", "clip.mp4", "--theme", "retro", "--filter", "ema",
             "--audio", "--scale", "blues", "--fast", "--no-loop"]
        )
        assert args.source == "clip.mp4"
        assert args.theme == "retro"
        assert args.filter == "ema"
        assert args.audio and args.fast and args.no_loop
        assert args.scale == "blues"

    def test_rejects_an_unknown_theme(self):
        from main import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--theme", "neon"])

    def test_list_providers_exits_cleanly(self, capsys):
        from main import main
        assert main(["--list-providers"]) == 0
        assert "ONNX" in capsys.readouterr().out

    def test_a_missing_video_file_exits_with_an_error(self, capsys):
        from main import main
        assert main(["--source", "definitely-not-here.mp4"]) == 1
        assert "not found" in capsys.readouterr().out.lower()
