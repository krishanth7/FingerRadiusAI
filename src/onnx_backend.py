"""
onnx_backend.py - Optional ONNX Runtime acceleration.

What this does and does not do, stated plainly, because "GPU acceleration"
is easy to overclaim:

* **A gesture model ships; a landmark detector does not.**
  ``models/gesture_classifier.onnx`` is real, trained and included -- it maps
  21 landmarks to a gesture, and :class:`OnnxGestureClassifier` runs it.
  Landmark *detection* is still MediaPipe's job: its
  ``hand_landmarker.task`` is a bundle of TFLite graphs, cannot be converted
  here, and retraining a detector needs a photographic dataset this
  repository does not have. Point :class:`OnnxLandmarkBackend` at your own
  exported detector to use that path.
* **MediaPipe remains the default and is not slower for having this here.**
  With no model supplied, :func:`available_providers` still reports what your
  ONNX Runtime build could use, and everything else stays on the CPU path.
* **A GPU provider is not a guarantee of more FPS.** On short sequences the
  host-to-device copy can cost more than the inference saves. Hence
  :meth:`OnnxLandmarkBackend.benchmark`: measure on your hardware rather than
  trusting a number from someone else's.

Provider preference is CUDA, then TensorRT, then DirectML on Windows, then
CoreML on macOS, then CPU. Whatever is missing is skipped silently -- ONNX
Runtime raises if you request a provider that is not compiled in, so the list
is filtered against what the installed build actually has.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "OnnxGestureClassifier",
    "DEFAULT_GESTURE_MODEL",
    "onnxruntime_available",
    "available_providers",
    "has_gpu_provider",
    "preferred_providers",
    "describe_runtime",
    "OnnxLandmarkBackend",
    "BenchmarkResult",
]

#: Best first. Filtered against the installed build before use.
PROVIDER_PREFERENCE: List[str] = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",       # DirectML, Windows
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",    # macOS
    "CPUExecutionProvider",
]

#: Providers that actually run on a GPU, for honest reporting.
_GPU_PROVIDERS = {
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
}


def onnxruntime_available() -> bool:
    """True when ``onnxruntime`` can be imported."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def available_providers() -> List[str]:
    """Execution providers this ONNX Runtime build supports. Empty if absent."""
    if not onnxruntime_available():
        return []
    import onnxruntime as ort
    return list(ort.get_available_providers())


def preferred_providers() -> List[str]:
    """Our preference order, filtered to what is installed."""
    have = set(available_providers())
    ordered = [p for p in PROVIDER_PREFERENCE if p in have]
    # Anything the build has that we did not rank still beats nothing.
    ordered += [p for p in available_providers() if p not in ordered]
    return ordered


def has_gpu_provider() -> bool:
    """True when at least one GPU execution provider is installed."""
    return bool(_GPU_PROVIDERS & set(available_providers()))


def describe_runtime() -> str:
    """One line for the console banner."""
    if not onnxruntime_available():
        return "ONNX Runtime not installed (pip install onnxruntime-gpu) - using MediaPipe"
    import onnxruntime as ort
    providers = preferred_providers()
    gpu = "GPU available" if has_gpu_provider() else "CPU only"
    return f"ONNX Runtime {ort.__version__}, {gpu}: {', '.join(providers)}"


@dataclass
class BenchmarkResult:
    """Timing from :meth:`OnnxLandmarkBackend.benchmark`."""

    provider: str
    runs: int
    mean_ms: float
    best_ms: float
    worst_ms: float

    @property
    def fps(self) -> float:
        """Inference-only frames per second; excludes capture and drawing."""
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.provider}: {self.mean_ms:.2f}ms mean "
            f"({self.best_ms:.2f} best, {self.worst_ms:.2f} worst) "
            f"= {self.fps:.1f} inferences/sec over {self.runs} runs"
        )


class OnnxLandmarkBackend:
    """Runs a user-supplied ONNX hand-landmark model.

    Args:
        model_path: Path to a ``.onnx`` file. Nothing is bundled.
        providers: Override the provider order. Defaults to
            :func:`preferred_providers`.
        input_size: Square edge the model expects, in pixels.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If the model file is missing.

    Example:
        >>> from src.onnx_backend import describe_runtime
        >>> isinstance(describe_runtime(), str)
        True
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[Sequence[str]] = None,
        input_size: int = 224,
    ) -> None:
        if not onnxruntime_available():
            raise ImportError(
                "ONNX acceleration needs onnxruntime. Install one of:\n"
                "    pip install onnxruntime        # CPU\n"
                "    pip install onnxruntime-gpu    # CUDA\n"
                "    pip install onnxruntime-directml   # Windows DirectML"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "No model ships with this repository -- export or download one "
                "and pass its path with --onnx-model."
            )

        import onnxruntime as ort

        self.model_path = model_path
        self.input_size = int(input_size)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        chosen = list(providers) if providers else preferred_providers()
        self.session = ort.InferenceSession(model_path, options, providers=chosen)

        # What the session actually got, which can differ from what we asked
        # for: ONNX Runtime falls back silently when a provider cannot load.
        self.active_providers: List[str] = list(self.session.get_providers())
        self.input_name: str = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names: List[str] = [o.name for o in self.session.get_outputs()]

    @property
    def provider(self) -> str:
        """The provider actually in use."""
        return self.active_providers[0] if self.active_providers else "unknown"

    @property
    def on_gpu(self) -> bool:
        """True when the active provider runs on a GPU."""
        return self.provider in _GPU_PROVIDERS

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, convert BGR to RGB, scale to 0-1 and add a batch axis."""
        import cv2

        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        """Run one frame through the model and return its raw outputs."""
        return self.session.run(self.output_names, {self.input_name: self.preprocess(frame)})

    def benchmark(self, runs: int = 50, warmup: int = 5) -> BenchmarkResult:
        """Time the model on random input.

        The first few runs are discarded: a GPU provider compiles kernels on
        the first call, and including that in the mean would understate it
        badly.
        """
        shape = (1, self.input_size, self.input_size, 3)
        dummy = np.random.rand(*shape).astype(np.float32)

        for _ in range(max(0, warmup)):
            self.session.run(self.output_names, {self.input_name: dummy})

        times: List[float] = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            self.session.run(self.output_names, {self.input_name: dummy})
            times.append((time.perf_counter() - start) * 1000.0)

        return BenchmarkResult(
            provider=self.provider,
            runs=len(times),
            mean_ms=sum(times) / len(times),
            best_ms=min(times),
            worst_ms=max(times),
        )

    def describe(self) -> str:
        """One line on the loaded session."""
        return (
            f"ONNX {os.path.basename(self.model_path)} on {self.provider}"
            f"{' (GPU)' if self.on_gpu else ' (CPU)'}, "
            f"input {self.input_shape}, {len(self.output_names)} output(s)"
        )


#: The gesture classifier trained by ``tools/train_gesture_onnx.py``.
DEFAULT_GESTURE_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "gesture_classifier.onnx",
)


class OnnxGestureClassifier:
    """Names a hand pose from its 21 landmarks, using the bundled ONNX model.

    This is the model that ships. It does **not** find landmarks -- MediaPipe
    does that -- it replaces the hand-written rule cascade in
    :mod:`src.gestures` with weights, which means you can retrain it on your
    own recordings instead of editing thresholds.

    Class names are read from the model's own metadata, so nothing has to keep
    a separate label list in sync with the weights.

    Args:
        model_path: Path to the ``.onnx`` file. Defaults to the bundled one.
        providers: Execution provider order; defaults to
            :func:`preferred_providers`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If the model file is missing, with the command that
            regenerates it.

    Example:
        >>> from src.onnx_backend import OnnxGestureClassifier
        >>> from tests.synthetic_hands import peace
        >>> classifier = OnnxGestureClassifier()          # doctest: +SKIP
        >>> classifier.classify(peace())[0]                # doctest: +SKIP
        'Peace'
    """

    def __init__(
        self,
        model_path: str = DEFAULT_GESTURE_MODEL,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if not onnxruntime_available():
            raise ImportError(
                "The ONNX gesture classifier needs onnxruntime:\n"
                "    pip install onnxruntime"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Gesture model not found: {model_path}\n"
                "Regenerate it with:  python tools/train_gesture_onnx.py"
            )

        import onnxruntime as ort

        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path, providers=list(providers) if providers else preferred_providers()
        )
        self.active_providers: List[str] = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name

        # Read the labels through onnxruntime's own metadata view rather than
        # onnx.load(). The two agree -- they read the same metadata_props --
        # but onnxruntime is already a hard requirement here while the onnx
        # package is only needed to *build* a model. Importing it for
        # inference made a CPU-only install fail with ModuleNotFoundError.
        meta = self.session.get_modelmeta().custom_metadata_map
        self.labels: List[str] = [
            label for label in meta.get("labels", "").split(",") if label
        ]
        if not self.labels:
            raise ValueError(
                f"{model_path} carries no label metadata. Regenerate it with "
                "tools/train_gesture_onnx.py."
            )

    @property
    def provider(self) -> str:
        return self.active_providers[0] if self.active_providers else "unknown"

    @property
    def on_gpu(self) -> bool:
        return self.provider in _GPU_PROVIDERS

    def classify(self, landmarks) -> Tuple[str, float]:
        """Return ``(gesture, probability)`` for one hand."""
        names, scores = self.classify_batch([landmarks])
        return names[0], scores[0]

    def classify_batch(self, hands) -> Tuple[List[str], List[float]]:
        """Classify several hands at once.

        Batching matters on a GPU: the per-call transfer dominates a model
        this small, so one call with thirty hands is far cheaper than thirty
        calls with one.
        """
        from src.gesture_trainer import describe_with_orientation

        features = np.stack([describe_with_orientation(h) for h in hands]).astype(np.float32)
        probabilities = self.session.run(None, {self.input_name: features})[0]
        indices = probabilities.argmax(axis=1)
        return (
            [self.labels[i] for i in indices],
            [float(probabilities[row, i]) for row, i in enumerate(indices)],
        )

    def describe(self) -> str:
        """One line for the console banner."""
        return (
            f"ONNX gesture classifier on {self.provider}"
            f"{' (GPU)' if self.on_gpu else ' (CPU)'}, "
            f"{len(self.labels)} classes: {', '.join(self.labels)}"
        )
