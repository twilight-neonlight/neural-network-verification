# verify.py

"""Run Marabou robustness verification for a given sample and epsilon."""

import os, sys
# Ensure project root is on the path when this file is run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from maraboupy import Marabou

from src.model_loader import load_model
from src.constraints import apply_input_constraints, apply_output_constraints, apply_rival_constraint


def verify_robustness(onnx_path: str, sample: np.ndarray, true_class: int, epsilon: float):
    """
    Verify that the network predicts true_class for all inputs within the L∞-ball
    of radius epsilon around sample.

    We check each rival class j != true_class separately:
      Query: does there exist x' in ball(sample, eps) such that output[j] >= output[true_class]?
      SAT   -> adversarial input found for class j
      UNSAT -> network is robust against class j within this epsilon

    The model is reloaded fresh for each rival query because Marabou does not
    support removing individual inequality constraints after they are added.

    Returns:
        dict with keys:
            'epsilon'        : float
            'true_class'     : int
            'robust'         : bool  (True iff UNSAT for all rivals)
            'counterexample' : np.ndarray or None  (first adversarial input found)
            'sat_rival'      : int or None  (rival class that broke robustness)
            'per_rival'      : list of dicts  (one entry per rival class)
            'total_time'     : float (seconds)
    """
    sample_flat = sample.flatten().astype(np.float32)

    _info = load_model(onnx_path)
    rivals = apply_output_constraints(_info.network, _info.output_vars, true_class)

    per_rival = []
    counterexample = None
    sat_rival = None
    t_start = time.time()

    for rival in rivals:
        info = load_model(onnx_path)
        apply_input_constraints(info.network, info.input_vars, sample_flat, epsilon)
        apply_rival_constraint(info.network, info.output_vars, true_class, rival)

        t0 = time.time()
        # solve() with no filename argument — engine log goes to stdout.
        # We print results after all queries finish so they appear at the end.
        exit_code, vals, _ = info.network.solve(verbose=False)
        elapsed = time.time() - t0

        result = {
            'rival': rival,
            'exit_code': exit_code,
            'time': elapsed,
            'counterexample': None,
        }

        if exit_code == 'sat':
            adv_input = np.array(
                [vals[int(v)] for v in info.input_vars], dtype=np.float32
            )
            result['counterexample'] = adv_input
            if counterexample is None:
                counterexample = adv_input
                sat_rival = rival

        per_rival.append(result)

    total_time = time.time() - t_start
    robust = all(r['exit_code'] == 'unsat' for r in per_rival)

    return {
        'epsilon': epsilon,
        'true_class': true_class,
        'robust': robust,
        'counterexample': counterexample,
        'sat_rival': sat_rival,
        'per_rival': per_rival,
        'total_time': total_time,
    }


def print_result(result: dict):
    """Print a human-readable summary to stderr (avoids being buried in Marabou engine logs)."""
    sep = '=' * 50
    lines = [
        sep,
        f"Digit: {result['true_class']}  |  epsilon: {result['epsilon']}",
        f"Robust: {result['robust']}  |  Total time: {result['total_time']:.2f}s",
        '',
    ]
    for r in result['per_rival']:
        status = 'SAT  <- adversarial example found' if r['exit_code'] == 'sat' else 'UNSAT (robust)'
        lines.append(f"  vs class {r['rival']}: {status}  ({r['time']:.3f}s)")
    lines.append(sep)
    print('\n'.join(lines), file=sys.stderr)


if __name__ == '__main__':
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ONNX_PATH    = os.path.join(BASE, 'models', 'model.onnx')
    SAMPLES_PATH = os.path.join(BASE, 'models', 'sample_inputs.npy')
    LABELS_PATH  = os.path.join(BASE, 'models', 'sample_labels.npy')
    RESULTS_PATH = os.path.join(BASE, 'results', 'verification_results.txt')

    samples = np.load(SAMPLES_PATH)
    labels  = np.load(LABELS_PATH)

    # Verify digit 0 with epsilon=0.01
    sample     = samples[0]
    true_class = int(labels[0])
    epsilon    = 0.01

    result = verify_robustness(ONNX_PATH, sample, true_class, epsilon)

    # Engine log has already been printed above; now print our summary
    print_result(result)

    # Also persist to file so the result survives the noisy log
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        f.write(f"digit={result['true_class']}  epsilon={result['epsilon']}\n")
        f.write(f"robust={result['robust']}  total_time={result['total_time']:.2f}s\n\n")
        for r in result['per_rival']:
            f.write(f"vs class {r['rival']}: {r['exit_code']}  ({r['time']:.3f}s)\n")
    print(f"\nResults saved to {RESULTS_PATH}", file=sys.stderr)
