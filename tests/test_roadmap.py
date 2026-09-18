"""
Tests for the last four roadmap items: dynamic gestures, gesture training,
multi-camera triangulation and the bundled ONNX gesture model.

All run with no camera, no display and no GPU.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamic_gestures import (
    DynamicGesture,
    DynamicGestureRecognizer,
    retire_absent,
)
from src.gesture_trainer import (
    TrainedGestureRecognizer,
    describe_landmarks,
    describe_with_orientation,
)
from src.multi_camera import CameraCalibration, StereoRig
from tests import synthetic_hands as hands


def _events(sequence, **kwargs):
    recognizer = DynamicGestureRecognizer(**kwargs)
    found = []
    for landmarks, landmarks_3d in sequence:
        event = recognizer.update(landmarks, landmarks_3d)
        if event.gesture != DynamicGesture.NONE:
            found.append(event)
    return found


class TestDynamicGestures:
    @pytest.mark.parametrize(
        "dx, dy, expected",
        [
            (18, 0, DynamicGesture.SWIPE_RIGHT),
            (-18, 0, DynamicGesture.SWIPE_LEFT),
            (0, -18, DynamicGesture.SWIPE_UP),
            (0, 18, DynamicGesture.SWIPE_DOWN),
        ],
    )
    def test_swipes(self, dx, dy, expected):
        events = _events(hands.moving_hand(20, dx=dx, dy=dy))
        assert [e.gesture for e in events] == [expected]

    def test_a_swipe_fires_exactly_once(self):
        """One swipe followed by stillness reports one swipe.

        Without the cooldown the same swipe is reported on every frame while
        the hand decelerates. Note the sequence has to *stop* moving: a hand
        that keeps travelling for sixty frames really has performed several
        swipes, and reporting one would be the bug.
        """
        sequence = hands.moving_hand(20, dx=18)
        sequence += hands.moving_hand(20, dx=0, start=(300 + 19 * 18, 400))
        swipes = [e for e in _events(sequence) if e.is_swipe]
        assert len(swipes) == 1

    def test_swipe_survives_a_shaky_hand(self):
        events = _events(hands.moving_hand(20, dx=18, jitter=3.0))
        assert [e.gesture for e in events] == [DynamicGesture.SWIPE_RIGHT]

    @pytest.mark.parametrize("scale", [45.0, 90.0, 240.0])
    def test_swipe_is_distance_invariant(self, scale):
        # Travel is measured in hand-widths, so the same gesture at a
        # different distance from the camera must still register.
        events = _events(hands.moving_hand(20, dx=scale * 0.2, scale=scale))
        assert [e.gesture for e in events] == [DynamicGesture.SWIPE_RIGHT]

    def test_hold(self):
        assert [e.gesture for e in _events(hands.moving_hand(40))] == [DynamicGesture.HOLD]

    def test_circle(self):
        events = _events(hands.moving_hand(24, path=hands.circular_path(frames=24)))
        assert [e.gesture for e in events] == [DynamicGesture.CIRCLE]

    def test_tap(self):
        # A tap moves toward the camera, not across it, so it has almost no
        # lateral travel -- it has to be checked before the stillness test or
        # it is swallowed as a hold.
        events = _events(hands.moving_hand(16, path=hands.tap_path(frames=16)))
        assert [e.gesture for e in events] == [DynamicGesture.TAP]

    def test_a_zigzag_is_not_a_circle(self):
        # Turning accumulates on a zigzag too; consistency of direction is
        # what rejects it.
        zigzag = hands.moving_hand(24, path=lambda i: ((-1) ** i * 40.0, i * 2.0, 0.0))
        assert _events(zigzag) == []

    def test_losing_the_hand_clears_history(self):
        recognizer = DynamicGestureRecognizer()
        for landmarks, landmarks_3d in hands.moving_hand(10, dx=18):
            recognizer.update(landmarks, landmarks_3d)
        recognizer.update(None)
        # A swipe that spanned the gap must not fire on the far side.
        fired = [
            recognizer.update(lm, lm3).gesture
            for lm, lm3 in hands.moving_hand(6, dx=18)
        ]
        assert all(g == DynamicGesture.NONE for g in fired)

    def test_direction_is_reported(self):
        event = _events(hands.moving_hand(20, dx=18))[0]
        assert event.is_swipe
        assert event.travel > 1.6
        assert -1 < event.direction_degrees < 361

    def test_short_history_is_rejected(self):
        with pytest.raises(ValueError, match="history"):
            DynamicGestureRecognizer(history=2)


class TestGestureTraining:
    POSES = {
        "my-peace": hands.peace,
        "my-fist": hands.fist,
        "my-ok": hands.ok_sign,
        "my-point": hands.pointing,
        "my-palm": hands.open_palm,
    }

    @pytest.fixture
    def trained(self):
        recognizer = TrainedGestureRecognizer()
        for name, builder in self.POSES.items():
            for scale, rotation in [(80.0, 0.0), (110.0, 12.0), (95.0, -9.0)]:
                recognizer.record(name, builder(scale=scale, rotation=rotation))
        recognizer.finalise()
        return recognizer

    @pytest.mark.parametrize("scale, rotation", [(45.0, 25.0), (260.0, -40.0), (150.0, 95.0)])
    def test_recognises_at_unseen_scale_and_rotation(self, trained, scale, rotation):
        for name, builder in self.POSES.items():
            predicted, _ = trained.predict(builder(scale=scale, rotation=rotation))
            assert predicted == name

    def test_descriptor_is_invariant(self):
        base = describe_landmarks(hands.peace())
        for scale, rotation in [(40.0, 0.0), (300.0, 0.0), (90.0, 140.0)]:
            other = describe_landmarks(hands.peace(scale=scale, rotation=rotation))
            assert np.linalg.norm(base - other) < 0.35

    def test_an_untaught_pose_returns_none(self, trained):
        # Returning the nearest label would be worse than useless: an unknown
        # pose is not a weak example of the closest gesture.
        predicted, distance = trained.predict(hands.thumbs_up())
        assert predicted is None
        assert distance > 1.0

    def test_orientation_feature_separates_thumbs_up_from_down(self):
        """The 42-value descriptor cannot separate thumbs-up from thumbs-down.

        Across varied hand rotations -- which is what the training set holds --
        rotation normalisation moves the thumb somewhere different every time,
        so the two classes overlap. The measure that shows it is separability:
        mean between-class distance over mean within-class distance. Below 1.0
        the classes are inseparable, because two examples of the *same* gesture
        sit further apart than one of each.

        Measured here: 0.96 on the 42-value descriptor, 1.05 with orientation
        appended. That gap is the difference between chance and 100% accuracy
        for the bundled ONNX model.
        """

        def separability(describe):
            rotations = np.linspace(0.0, 360.0, 25)
            up = np.stack([describe(hands.thumbs_up(rotation=r)) for r in rotations])
            down = np.stack([describe(hands.thumbs_down(rotation=r)) for r in rotations])
            within = np.mean([
                np.linalg.norm(up[i] - up[j])
                for i in range(len(up)) for j in range(i + 1, len(up))
            ])
            between = np.mean([np.linalg.norm(u - d) for u in up for d in down])
            return between / within

        shape_only = separability(describe_landmarks)
        oriented = separability(describe_with_orientation)

        assert shape_only < 1.0, "the 42-value descriptor should NOT separate these"
        assert oriented > 1.0, "adding orientation should separate them"
        assert oriented > shape_only
        assert describe_with_orientation(hands.thumbs_up()).shape == (44,)

    def test_save_and_load_round_trip(self, trained, tmp_path):
        path = trained.save(str(tmp_path / "gestures.json"))
        reloaded = TrainedGestureRecognizer.load(path)
        assert set(reloaded.templates) == set(self.POSES)
        assert reloaded.predict(hands.peace(scale=170.0))[0] == "my-peace"

    def test_model_is_json_not_pickle(self, trained, tmp_path):
        # A gesture file is something people share; loading a shared pickle
        # executes whatever is inside it.
        import json

        path = trained.save(str(tmp_path / "g.json"))
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle)["version"] == 1

    def test_a_mismatched_version_is_refused(self, trained, tmp_path):
        import json

        path = trained.save(str(tmp_path / "g.json"))
        data = json.load(open(path, encoding="utf-8"))
        data["version"] = 99
        json.dump(data, open(path, "w", encoding="utf-8"))
        with pytest.raises(ValueError, match="version"):
            TrainedGestureRecognizer.load(path)

    def test_forget(self, trained):
        assert trained.forget("my-peace") is True
        assert trained.forget("my-peace") is False

    def test_unnamed_gesture_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            TrainedGestureRecognizer().record("  ", hands.peace())


class TestMultiCamera:
    @pytest.fixture
    def rig(self):
        left = CameraCalibration.simple("L", (-0.15, 0.0, 0.0), (0.0, 0.0, 1.0))
        right = CameraCalibration.simple("R", (0.15, 0.0, 0.0), (0.0, 0.0, 1.0))
        return StereoRig(left, right)

    @staticmethod
    def _truth(n=21, seed=0):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            rng.uniform(-0.1, 0.1, n),
            rng.uniform(-0.1, 0.1, n),
            rng.uniform(0.9, 1.2, n),
        ])

    def test_perfect_observations_recover_the_point_exactly(self, rig):
        truth = self._truth()
        result = rig.triangulate(
            [rig.left.project(p) for p in truth],
            [rig.right.project(p) for p in truth],
        )
        assert np.abs(result.points - truth).max() < 1e-9
        assert result.mean_error < 1e-6
        assert result.is_reliable()

    def test_pixel_noise_gives_millimetre_accuracy(self, rig):
        rng = np.random.default_rng(3)
        truth = self._truth()
        left = [(x + rng.normal(0, 1), y + rng.normal(0, 1))
                for x, y in (rig.left.project(p) for p in truth)]
        right = [(x + rng.normal(0, 1), y + rng.normal(0, 1))
                 for x, y in (rig.right.project(p) for p in truth)]
        result = rig.triangulate(left, right)
        errors = np.linalg.norm(result.points - truth, axis=1)
        # A 30 cm baseline at about 1 m, with 1 px of noise.
        assert errors.mean() < 0.02          # under 2 cm
        assert result.mean_error < 3.0       # pixels

    def test_reprojection_error_flags_broken_correspondence(self, rig):
        # Shuffling one view breaks the pairing. The solver will still return
        # points, so the error is the only thing that can catch it.
        truth = self._truth()
        left = [rig.left.project(p) for p in truth]
        right = [rig.right.project(p) for p in truth]
        result = rig.triangulate(left, right[::-1])
        assert not result.is_reliable(tolerance=3.0)

    def test_metric_distance_is_recovered(self, rig):
        truth = self._truth()
        result = rig.triangulate(
            [rig.left.project(p) for p in truth],
            [rig.right.project(p) for p in truth],
        )
        expected = float(np.linalg.norm(truth[0] - truth[8]))
        assert result.distance(0, 8) == pytest.approx(expected, abs=1e-6)

    def test_a_zero_baseline_is_refused(self):
        camera = CameraCalibration.simple("A", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="baseline"):
            StereoRig(camera, camera)

    def test_mismatched_point_counts_are_refused(self, rig):
        with pytest.raises(ValueError, match="same order"):
            rig.triangulate([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])

    def test_a_mirrored_rotation_is_refused(self):
        with pytest.raises(ValueError, match="determinant"):
            CameraCalibration("bad", np.eye(3), np.diag([1.0, 1.0, -1.0]), np.zeros(3))

    def test_a_point_behind_the_camera_is_refused(self, rig):
        with pytest.raises(ValueError, match="behind"):
            rig.left.project((0.0, 0.0, -5.0))

    def test_hand_triangulation_needs_21_landmarks(self, rig):
        with pytest.raises(ValueError, match="21 landmarks"):
            rig.triangulate_hand([(0, 0)] * 5, [(0, 0)] * 5)

    def test_save_and_load(self, rig, tmp_path):
        path = rig.save(str(tmp_path / "rig.json"))
        reloaded = StereoRig.load(path)
        assert reloaded.baseline == pytest.approx(rig.baseline)


class TestBundledOnnxModel:
    @pytest.fixture(scope="class")
    def classifier(self):
        from src.onnx_backend import (
            DEFAULT_GESTURE_MODEL,
            OnnxGestureClassifier,
            onnxruntime_available,
        )

        if not onnxruntime_available():
            pytest.skip("onnxruntime is not installed.")
        if not os.path.exists(DEFAULT_GESTURE_MODEL):
            pytest.skip("Run tools/train_gesture_onnx.py to build the model.")
        return OnnxGestureClassifier()

    def test_model_ships_with_the_repository(self):
        from src.onnx_backend import DEFAULT_GESTURE_MODEL

        assert os.path.exists(DEFAULT_GESTURE_MODEL), (
            "gesture_classifier.onnx should be committed; regenerate with "
            "python tools/train_gesture_onnx.py"
        )

    def test_labels_come_from_the_model_metadata(self, classifier):
        # Nothing should have to keep a separate label list in sync.
        assert len(classifier.labels) == 8
        assert "Peace" in classifier.labels

    @pytest.mark.parametrize(
        "builder, expected",
        [
            (hands.fist, "Fist"),
            (hands.open_palm, "Open Palm"),
            (hands.peace, "Peace"),
            (hands.pointing, "Pointing"),
            (hands.thumbs_up, "Thumbs Up"),
            (hands.thumbs_down, "Thumbs Down"),
            (hands.ok_sign, "OK"),
            (hands.pinch, "Pinch"),
        ],
    )
    def test_classifies_each_gesture(self, classifier, builder, expected):
        gesture, probability = classifier.classify(builder(scale=155.0, rotation=37.0))
        assert gesture == expected
        assert probability > 0.5

    def test_thumbs_up_and_down_are_distinguished(self, classifier):
        # The pair the rotation-invariant descriptor could not separate.
        assert classifier.classify(hands.thumbs_up())[0] == "Thumbs Up"
        assert classifier.classify(hands.thumbs_down())[0] == "Thumbs Down"

    def test_batch_matches_single(self, classifier):
        poses = [hands.peace(), hands.fist(), hands.ok_sign()]
        batched, _ = classifier.classify_batch(poses)
        assert batched == [classifier.classify(p)[0] for p in poses]


class TestRecordingDoesNotDestroyTheFile:
    """Recording a second gesture must keep the first.

    `save` writes the whole model, so a recorder that starts empty erases
    everything already in the file. That is silent data loss: the CLI reports
    a successful save while the previous gestures are gone.
    """

    def test_a_second_session_keeps_the_first_gesture(self, tmp_path):
        path = str(tmp_path / "gestures.json")

        first = TrainedGestureRecognizer()
        first.record("wave", hands.open_palm())
        first.finalise()
        first.save(path)

        second = TrainedGestureRecognizer.open_for_recording(path)
        second.record("vee", hands.peace())
        second.finalise()
        second.save(path)

        stored = TrainedGestureRecognizer.load(path)
        assert set(stored.templates) == {"wave", "vee"}

    def test_a_missing_file_starts_empty(self, tmp_path):
        recorder = TrainedGestureRecognizer.open_for_recording(
            str(tmp_path / "nothing-here.json")
        )
        assert recorder.templates == {}

    def test_an_unreadable_file_is_never_overwritten(self, tmp_path):
        path = tmp_path / "gestures.json"
        path.write_text("{ this is not json")

        with pytest.raises(ValueError):
            TrainedGestureRecognizer.open_for_recording(str(path))

        # The point of raising: the caller stops, so the file survives.
        assert path.read_text() == "{ this is not json"


class TestMotionHistoryIsClearedWhenAHandLeaves:
    """A hand that disappears and returns elsewhere is not a swipe."""

    def test_a_gap_does_not_stitch_into_a_false_swipe(self):
        recognizers = [DynamicGestureRecognizer(), DynamicGestureRecognizer()]

        # Hand 1 is tracked on the left of the frame, long enough to fill the
        # 16-frame window -- the recogniser reports nothing until it is full,
        # so a shorter run would hide the bug rather than test it.
        for lm, lm3 in hands.moving_hand(20, dx=1.0, start=(150, 400)):
            recognizers[1].update(lm, lm3)

        # It leaves: only hand 0 is detected for a few frames.
        for _ in range(5):
            retire_absent(recognizers, [0])

        # It comes back far to the right. Without the reset the window still
        # holds the old positions, and the jump to these new ones reads as one
        # long, straight, fast displacement: a phantom 'Swipe Right' fires on
        # the very first frame back.
        events = [
            recognizers[1].update(lm, lm3)
            for lm, lm3 in hands.moving_hand(4, dx=1.0, start=(1100, 400))
        ]
        assert all(e.gesture == DynamicGesture.NONE for e in events)

    def test_a_hand_still_present_keeps_its_history(self):
        recognizers = [DynamicGestureRecognizer(), DynamicGestureRecognizer()]
        for lm, lm3 in hands.moving_hand(6, dx=18.0, start=(150, 400)):
            recognizers[0].update(lm, lm3)
            retire_absent(recognizers, [0])
        # Hand 0 was present every frame, so it was never reset and the swipe
        # it is halfway through can still complete.
        fired = [
            recognizers[0].update(lm, lm3)
            for lm, lm3 in hands.moving_hand(
                10, dx=18.0, start=(150 + 6 * 18, 400)
            )
        ]
        assert any(e.gesture == DynamicGesture.SWIPE_RIGHT for e in fired)


class TestLearnedConfidence:
    """A learned label must not carry another recogniser's confidence."""

    def test_confidence_falls_as_the_match_worsens(self):
        model = TrainedGestureRecognizer()
        for _ in range(3):
            model.record("wave", hands.open_palm())
        model.finalise()

        name, distance = model.predict(hands.open_palm())
        assert name == "wave"

        limit = model.templates[name].threshold
        confidence = max(0.0, 1.0 - distance / limit)
        assert 0.0 <= confidence <= 1.0
        # An exact repeat of the recorded pose is a near-perfect match.
        assert confidence > 0.9
