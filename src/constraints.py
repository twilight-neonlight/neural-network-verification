# constraints.py

"""Build input/output constraints for Marabou robustness verification."""

import numpy as np


def apply_input_constraints(network, input_vars: np.ndarray, sample: np.ndarray, epsilon: float):
    """
    Set L∞-ball input bounds: x_i ∈ [sample_i - ε, sample_i + ε] ∩ [0, 1].

    Args:
        network:     MarabouNetworkONNX object
        input_vars:  flat array of Marabou variable indices for the input layer
        sample:      1-D numpy array of the reference input (already scaled to [0, 1])
        epsilon:     perturbation radius
    """
    flat = sample.flatten()
    for var, val in zip(input_vars, flat):
        network.setLowerBound(var, float(np.clip(val - epsilon, 0.0, 1.0)))
        network.setUpperBound(var, float(np.clip(val + epsilon, 0.0, 1.0)))


def apply_output_constraints(network, output_vars: np.ndarray, true_class: int):
    """
    Add the negated robustness property: "there exists some rival class j
    whose logit is >= the true class logit."

    This encodes the SAT query:  output[j] - output[true_class] >= 0
    i.e. addInequality asks Marabou whether  output[true_class] - output[j] <= 0
    holds for any rival j.

    Marabou interprets SAT as "counterexample found" and UNSAT as "property holds."
    We run one query per rival class and return the list of rival indices checked.

    NOTE: call this once per (rival_class, solve) pair — see verify.py.
    This function adds the constraint for a SINGLE rival class at a time.

    Args:
        network:     MarabouNetworkONNX object (constraints are added in-place)
        output_vars: flat array of Marabou variable indices for the output layer
        true_class:  the correct class index that should remain the argmax
    """
    n_classes = len(output_vars)
    rivals = [j for j in range(n_classes) if j != true_class]
    return rivals


def apply_rival_constraint(network, output_vars: np.ndarray, true_class: int, rival_class: int):
    """
    Add the single inequality:  output[true_class] - output[rival_class] <= 0
    which is equivalent to asking whether output[rival_class] >= output[true_class].

    Args:
        network:      MarabouNetworkONNX object
        output_vars:  flat array of output variable indices
        true_class:   index of the class that should be the argmax
        rival_class:  index of the class to compare against
    """
    # addInequality(vars, coefficients, scalar) encodes: sum(coeff_i * var_i) <= scalar
    # We encode: output[true_class] - output[rival_class] <= 0
    network.addInequality(
        [int(output_vars[true_class]), int(output_vars[rival_class])],
        [1.0, -1.0],
        0.0,
    )
