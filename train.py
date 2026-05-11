"""
Train a small MLP on MNIST (sklearn) and export to ONNX.

Architecture: 784 -> 32 -> 16 -> 10 (ReLU activations, softmax output)
Kept intentionally small so Marabou can verify it within reasonable time.
"""

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import onnx.shape_inference
from onnx import TensorProto
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os


def load_mnist_subset(n_samples=5000):
    """Load a subset of MNIST to keep training fast."""
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    # Use a small subset for quick training
    X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=42)

    # Scale to [0, 1] — same range we'll use in verification constraints
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


def train_mlp(X, y):
    """Train a small MLP classifier."""
    print("Training MLP (784->32->16->10)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=30,
        random_state=42,
        verbose=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    return clf


def export_to_onnx(clf, output_path):
    """
    Manually build an ONNX graph from sklearn MLP weights.
    sklearn MLPClassifier stores weights in clf.coefs_ and biases in clf.intercepts_.
    We emit: Input -> Gemm+ReLU (x N hidden layers) -> Gemm (output logits)
    """
    coefs = clf.coefs_        # list of weight matrices
    intercepts = clf.intercepts_  # list of bias vectors

    nodes = []
    initializers = []
    layer_input = "input"

    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        # sklearn coefs_ shape: (in_features, out_features)
        # ONNX Gemm default: Y = A @ B, so B must be (in, out) — no transpose needed
        W = W.astype(np.float32)
        b = b.astype(np.float32)

        w_name = f"W{i}"
        b_name = f"b{i}"
        initializers.append(numpy_helper.from_array(W, name=w_name))
        initializers.append(numpy_helper.from_array(b, name=b_name))

        gemm_out = f"gemm_out{i}"
        nodes.append(helper.make_node(
            "Gemm", inputs=[layer_input, w_name, b_name], outputs=[gemm_out]
        ))

        # Apply ReLU on all layers except the last (output logits)
        if i < len(coefs) - 1:
            relu_out = f"relu_out{i}"
            nodes.append(helper.make_node("Relu", inputs=[gemm_out], outputs=[relu_out]))
            layer_input = relu_out
        else:
            layer_input = gemm_out

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 784])
    output_tensor = helper.make_tensor_value_info(layer_input, TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        nodes, "mnist_mlp", [input_tensor], [output_tensor], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.doc_string = "Small MNIST MLP (784->32->16->10) for Marabou verification"
    # Shape inference populates value_info for intermediate tensors — required by Marabou's ONNX parser
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved ONNX model to: {output_path}")
    return model


def save_sample_inputs(clf, X, y, out_dir):
    """Save a few correctly classified samples for use in verification."""
    os.makedirs(out_dir, exist_ok=True)
    saved = {}
    for digit in range(10):
        indices = np.where(y == digit)[0]
        for idx in indices:
            sample = X[idx:idx+1]
            pred = clf.predict(sample)[0]
            if pred == digit:
                saved[digit] = (sample, digit)
                break

    samples = np.vstack([v[0] for v in saved.values()])
    labels = np.array([v[1] for v in saved.values()])
    np.save(os.path.join(out_dir, "sample_inputs.npy"), samples)
    np.save(os.path.join(out_dir, "sample_labels.npy"), labels)
    print(f"Saved {len(saved)} sample inputs (one per digit) to {out_dir}/")
    return samples, labels


if __name__ == "__main__":
    X, y, scaler = load_mnist_subset(n_samples=5000)
    clf = train_mlp(X, y)

    os.makedirs("models", exist_ok=True)
    export_to_onnx(clf, "models/model.onnx")

    save_sample_inputs(clf, X, y, "models")
    print("Done.")
