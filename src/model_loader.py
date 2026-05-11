# model_loader.py

"""Load an ONNX model into Marabou and expose input/output variable indices."""

import numpy as np
from maraboupy import Marabou


def load_model(onnx_path: str):
    """
    Load an ONNX model via Marabou and return a NetworkInfo namedtuple.

    Returns a simple namespace with:
        .network     - MarabouNetworkONNX object (used to add constraints and call solve)
        .input_vars  - flat array of Marabou variable indices for the input layer (shape: [n_inputs])
        .output_vars - flat array of Marabou variable indices for the output layer (shape: [n_outputs])
    """
    network = Marabou.read_onnx(onnx_path)

    # inputVars / outputVars are lists of arrays; flatten each to 1-D for convenience
    input_vars = network.inputVars[0].flatten()
    output_vars = network.outputVars[0].flatten()

    return _NetworkInfo(network, input_vars, output_vars)


class _NetworkInfo:
    """Thin wrapper that bundles the Marabou network with flat variable index arrays."""

    def __init__(self, network, input_vars: np.ndarray, output_vars: np.ndarray):
        self.network = network
        self.input_vars = input_vars    # shape (784,) for MNIST MLP
        self.output_vars = output_vars  # shape (10,)  for 10-class output

    def __repr__(self):
        return (
            f"NetworkInfo(inputs={len(self.input_vars)}, "
            f"outputs={len(self.output_vars)})"
        )
