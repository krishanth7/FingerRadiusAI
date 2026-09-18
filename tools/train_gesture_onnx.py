"""
train_gesture_onnx.py - Train a gesture classifier and export it to ONNX.

Why this model and not a landmark detector
------------------------------------------
The roadmap item asked for "an ONNX hand model". MediaPipe's
``hand_landmarker.task`` is a bundle of TFLite graphs and cannot be converted
here, and re-training a landmark detector needs a labelled photographic
dataset that this repository does not have and should not invent.

What *can* be trained honestly is the stage after detection: **landmarks to
gesture**. That is a real model, it ships, it runs on ONNX Runtime with GPU
providers, and it is useful -- it replaces a hand-written rule cascade with
something you can retrain on your own recordings.

So the pipeline is: MediaPipe finds the 21 landmarks, this model names the
pose. The split is stated in the README rather than blurred.

Training data
-------------
Poses are generated across a wide range of scales, rotations and jitter, and
labelled by the geometric recogniser in :mod:`src.gestures`. The network is
distilling those rules into weights -- which is exactly what makes it
retrainable on real recordings later: swap the generator for your own labelled
captures and rerun.

Features are the 44-value descriptor from
:func:`src.gesture_trainer.describe_with_orientation`: 42 values normalised
for position, scale and rotation, plus 2 giving the hand's actual orientation.

That last pair is not decoration. Trained on the rotation-invariant 42 alone,
the network scores 100% on six gestures and chance on the other two -- because
a thumbs-up and a thumbs-down are the same shape, and normalising away
rotation deletes the only difference between them.

The network is a 42-32-16-N MLP, trained with plain NumPy: Adam, cross
entropy, no framework. At this size a dependency on PyTorch would be heavier
than the training loop.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gesture_trainer import describe_with_orientation
from src.gestures import Gesture, GestureRecognizer
from tests import synthetic_hands as hands

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_OUT = os.path.join(MODEL_DIR, "gesture_classifier.onnx")

# The graph targets the oldest onnxruntime requirements.txt allows (1.16),
# which supports IR version 9 and well past opset 13.
IR_VERSION = 9
OPSET = 13

BUILDERS = {
    Gesture.FIST: hands.fist,
    Gesture.OPEN_PALM: hands.open_palm,
    Gesture.PEACE: hands.peace,
    Gesture.POINTING: hands.pointing,
    Gesture.THUMBS_UP: hands.thumbs_up,
    Gesture.THUMBS_DOWN: hands.thumbs_down,
    Gesture.OK: hands.ok_sign,
    Gesture.PINCH: hands.pinch,
}


def build_dataset(
    samples_per_class: int = 400, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate descriptors and labels across many scales and rotations."""
    rng = np.random.default_rng(seed)
    recognizer = GestureRecognizer(hold_frames=1)
    labels = sorted(BUILDERS)

    features: List[np.ndarray] = []
    targets: List[int] = []
    disagreements = 0

    for index, name in enumerate(labels):
        builder = BUILDERS[name]
        for _ in range(samples_per_class):
            scale = float(rng.uniform(45.0, 260.0))
            rotation = float(rng.uniform(0.0, 360.0))
            landmarks = builder(scale=scale, rotation=rotation)

            # Jitter every landmark, so the network sees the kind of noise a
            # real detector produces rather than perfect geometry.
            noise = rng.normal(0.0, scale * 0.012, (21, 2))
            noisy = [
                (int(x + noise[i, 0]), int(y + noise[i, 1]))
                for i, (x, y) in enumerate(landmarks)
            ]

            # Label with the geometric recogniser, not the builder's name: if
            # jitter has pushed the pose somewhere else, the honest label is
            # what the rules now say, not what we intended to draw.
            verdict = recognizer.classify(noisy).gesture
            if verdict != name:
                disagreements += 1
                continue

            features.append(describe_with_orientation(noisy))
            targets.append(index)

    print(f"  {len(features)} samples, {len(labels)} classes "
          f"({disagreements} dropped where jitter changed the pose)")
    return np.stack(features), np.asarray(targets, dtype=np.int64), labels


class MLP:
    """A 42-32-16-N classifier trained with Adam, in NumPy."""

    def __init__(self, n_in: int, n_out: int, hidden=(32, 16), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        # He initialisation: the right scale for ReLU layers.
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self.biases = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        activations = [x]
        current = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            current = current @ w + b
            if i < len(self.weights) - 1:
                current = np.maximum(current, 0.0)
            activations.append(current)
        return current, activations

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def train(self, x, y, epochs=300, lr=0.01, batch=64, seed=0) -> List[float]:
        rng = np.random.default_rng(seed)
        n_classes = self.weights[-1].shape[1]
        onehot = np.eye(n_classes)[y]

        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        step = 0
        history = []

        for epoch in range(epochs):
            order = rng.permutation(len(x))
            total = 0.0
            for start in range(0, len(x), batch):
                idx = order[start:start + batch]
                xb, yb = x[idx], onehot[idx]

                logits, acts = self.forward(xb)
                probs = self.softmax(logits)
                total += float(-np.mean(np.sum(yb * np.log(probs + 1e-12), axis=1)))

                delta = (probs - yb) / len(idx)
                grads_w, grads_b = [], []
                for i in reversed(range(len(self.weights))):
                    grads_w.insert(0, acts[i].T @ delta)
                    grads_b.insert(0, delta.sum(axis=0))
                    if i > 0:
                        delta = (delta @ self.weights[i].T) * (acts[i] > 0)

                step += 1
                for i in range(len(self.weights)):
                    for param, grad, m, v in (
                        (self.weights, grads_w, m_w, v_w),
                        (self.biases, grads_b, m_b, v_b),
                    ):
                        m[i] = 0.9 * m[i] + 0.1 * grad[i]
                        v[i] = 0.999 * v[i] + 0.001 * grad[i] ** 2
                        m_hat = m[i] / (1 - 0.9 ** step)
                        v_hat = v[i] / (1 - 0.999 ** step)
                        param[i] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

            history.append(total / max(1, len(x) // batch))
            if (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch + 1:3d}  loss {history[-1]:.5f}")
        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].argmax(axis=1)


def export_onnx(model: MLP, labels: List[str], path: str) -> str:
    """Write the trained weights out as an ONNX graph."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Writing an ONNX graph needs the onnx package:\n"
            "    pip install onnx\n"
            "Running the bundled model does not -- that only needs "
            "onnxruntime."
        ) from exc

    nodes, initializers = [], []
    current = "landmarks"

    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        w_name, b_name = f"W{i}", f"B{i}"
        initializers.append(numpy_helper.from_array(w.astype(np.float32), w_name))
        initializers.append(numpy_helper.from_array(b.astype(np.float32), b_name))
        out = f"gemm{i}"
        nodes.append(helper.make_node("Gemm", [current, w_name, b_name], [out]))
        current = out
        if i < len(model.weights) - 1:
            relu = f"relu{i}"
            nodes.append(helper.make_node("Relu", [current], [relu]))
            current = relu

    nodes.append(helper.make_node("Softmax", [current], ["probabilities"], axis=1))

    graph = helper.make_graph(
        nodes,
        "fingerradius_gesture_classifier",
        [helper.make_tensor_value_info(
            "landmarks", TensorProto.FLOAT, ["batch", model.weights[0].shape[0]])],
        [helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, ["batch", len(labels)])],
        initializer=initializers,
    )
    proto = helper.make_model(
        graph, producer_name="FingerRadiusAI",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    # Pin the IR version as well as the opset. Without this the stamp is
    # whatever the installed onnx package happens to default to -- onnx 1.22
    # writes IR 13 -- and onnxruntime refuses to load a model newer than it
    # understands:
    #     Unsupported model IR version: 13, max supported IR version: 11
    # requirements.txt allows onnxruntime>=1.16, which supports IR 9, so the
    # shipped model targets that floor rather than the build machine.
    proto.ir_version = IR_VERSION
    proto.doc_string = (
        "Maps a 44-value hand descriptor (see "
        "src.gesture_trainer.describe_with_orientation) to gesture probabilities. "
        "This does NOT detect landmarks -- MediaPipe does that."
    )
    # Class names travel with the weights, so nothing has to keep a separate
    # label list in sync with the model.
    entry = proto.metadata_props.add()
    entry.key = "labels"
    entry.value = ",".join(labels)

    onnx.checker.check_model(proto)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(proto, path)

    # Read it back rather than trusting the write: this is the exact property
    # a newer machine cannot check by simply loading the file, because a newer
    # onnxruntime accepts a stamp the supported floor would reject.
    written = onnx.load(path).ir_version
    if written != IR_VERSION:
        raise RuntimeError(
            f"{path} was written with IR version {written}, expected "
            f"{IR_VERSION}. Older onnxruntime installs would refuse it."
        )
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args(argv)

    print("Building dataset...")
    x, y, labels = build_dataset(args.samples)

    rng = np.random.default_rng(1)
    order = rng.permutation(len(x))
    split = int(len(x) * 0.8)
    train_idx, test_idx = order[:split], order[split:]

    print(f"Training on {len(train_idx)}, holding out {len(test_idx)}...")
    model = MLP(x.shape[1], len(labels))
    model.train(x[train_idx], y[train_idx], epochs=args.epochs)

    train_acc = float((model.predict(x[train_idx]) == y[train_idx]).mean())
    test_acc = float((model.predict(x[test_idx]) == y[test_idx]).mean())
    print(f"  train accuracy {train_acc:.4f}   held-out accuracy {test_acc:.4f}")

    path = export_onnx(model, labels, args.out)
    size = os.path.getsize(path)
    print(f"Exported {path} ({size / 1024:.1f} KiB)")

    # Verify the exported graph reproduces the NumPy model, rather than
    # assuming the export was faithful.
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    probs = session.run(None, {"landmarks": x[test_idx].astype(np.float32)})[0]
    onnx_acc = float((probs.argmax(axis=1) == y[test_idx]).mean())
    agreement = float((probs.argmax(axis=1) == model.predict(x[test_idx])).mean())
    print(f"  ONNX Runtime accuracy {onnx_acc:.4f}, agrees with NumPy on {agreement:.2%}")
    if agreement < 1.0:
        print("  [WARN] The exported graph disagrees with the trained model.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
