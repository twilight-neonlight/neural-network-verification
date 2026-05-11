# test.py

"""
Entry point for Marabou verification.
Run with: wsl -d Ubuntu-22.04 -- /home/<user>/.venvs/nnv/bin/python test.py
See README.md for full setup instructions.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.verify import verify_robustness, print_result

BASE         = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH    = os.path.join(BASE, 'models', 'model.onnx')
SAMPLES_PATH = os.path.join(BASE, 'models', 'sample_inputs.npy')
LABELS_PATH  = os.path.join(BASE, 'models', 'sample_labels.npy')
RESULTS_PATH = os.path.join(BASE, 'results', 'verification_results.txt')

# Verify the first three digits (0, 1, 2) at two epsilon levels
DIGITS   = [0, 1, 2]
EPSILONS = [0.01, 0.05]

if __name__ == '__main__':
    samples = np.load(SAMPLES_PATH)
    labels  = np.load(LABELS_PATH)

    all_results = []
    for digit_idx in DIGITS:
        for eps in EPSILONS:
            true_class = int(labels[digit_idx])
            result = verify_robustness(ONNX_PATH, samples[digit_idx], true_class, epsilon=eps)
            print_result(result)
            all_results.append(result)

    # Persist all results to file
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        for r in all_results:
            f.write(f"digit={r['true_class']}  epsilon={r['epsilon']}\n")
            f.write(f"robust={r['robust']}  total_time={r['total_time']:.2f}s\n")
            for pr in r['per_rival']:
                f.write(f"  vs class {pr['rival']}: {pr['exit_code']}  ({pr['time']:.3f}s)\n")
            f.write('\n')

    print(f"Results saved to {RESULTS_PATH}", file=sys.stderr)
