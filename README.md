# neural-network-verification

Neural network robustness verification using Marabou on an MNIST MLP model.

---

## Overview

This project uses **Marabou**, an SMT-based neural network verification tool, to formally verify the local robustness of a small MLP trained on MNIST. Given an input image and a perturbation radius ε, Marabou checks whether every input within the L∞-ball of radius ε around that image is classified as the same digit.

- **UNSAT**: no adversarial example exists within the ε-ball — the network is provably robust
- **SAT**: an adversarial example was found — the network is not robust at this ε

---

## Project Structure

```
neural-network-verification/
  src/
    __init__.py       : Makes src/ a Python package
    model_loader.py   : Load ONNX model into Marabou, expose input/output variable indices
    constraints.py    : Build L∞-ball input bounds and rival-class output constraints
    verify.py         : Run Marabou queries per rival class, collect SAT/UNSAT results
  models/
    model.onnx        : Trained MNIST MLP in ONNX format
    sample_inputs.npy : One correctly-classified sample per digit (10 total)
    sample_labels.npy : Corresponding digit labels
  results/
    verification_results.txt : Persisted verification output
  train.py            : Train MNIST MLP and export to ONNX
  test.py             : Entry point — runs full verification pipeline via Marabou
  requirements.txt
  report.pdf
```

---

## Model

| Architecture | Dataset | Test Accuracy |
|---|---|---|
| MLP (784 → 32 → 16 → 10, ReLU) | MNIST | 92.0% |

The model is intentionally small so Marabou can verify it within seconds per query. It is trained with scikit-learn's `MLPClassifier` and exported to ONNX with explicit shape inference so Marabou's ONNX parser can resolve intermediate tensor shapes.

---

## Setup

### Requirements

- **Windows**: Python 3.x with `.venv` (for `train.py` only)
- **WSL (Ubuntu 22.04)**: required for Marabou (`test.py`)

Marabou must be built from source inside WSL. The Python packages in `requirements.txt` cover everything except Marabou itself.

### 1. Install Python dependencies

```bash
# Windows (.venv) — for train.py
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# WSL — for test.py
python3 -m venv ~/.venvs/nnv
~/.venvs/nnv/bin/pip install -r requirements.txt
```

### 2. Build and register Marabou (WSL only)

```bash
# Clone and build Marabou
git clone https://github.com/NeuralNetworkVerification/Marabou.git ~/Marabou
cd ~/Marabou
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON
make -j$(nproc)

# Register maraboupy so it is importable from the project venv
echo "$HOME/Marabou" > ~/.venvs/nnv/lib/python3.10/site-packages/marabou.pth
```

---

## Usage

### Train the model

```bash
# Windows
.venv\Scripts\python train.py

# WSL
~/.venvs/nnv/bin/python train.py
```

Trains the MLP on 5 000 MNIST samples and saves `models/model.onnx` and `models/sample_inputs.npy`.

### Run verification

```bash
# WSL (Marabou required)
wsl -d Ubuntu-22.04 -- bash -c \
  "cd /mnt/c/Users/<user>/neural-network-verification && \
   ~/.venvs/nnv/bin/python test.py 2>&1 1>/dev/null"
```

Redirecting stdout hides Marabou's internal engine log; results appear on stderr. They are also saved to `results/verification_results.txt`.

---

## Verification Approach

For each query:

1. **Input constraints**: bound each of the 784 input pixels to `[x_i − ε, x_i + ε] ∩ [0, 1]`
2. **Output constraint**: ask whether any rival class `j` satisfies `output[j] ≥ output[true_class]`
3. Run Marabou — **UNSAT** means no such input exists (robust); **SAT** returns a counterexample

Each rival class is checked in a separate query because Marabou does not support removing individual constraints after they are added.

---

## Results

Digits 0, 1, 2 verified at ε = 0.01 and ε = 0.05:

| Digit | ε | Robust | SAT rival(s) | Total time |
|---|---|---|---|---|
| 0 | 0.01 | ✓ | — | 0.44s |
| 0 | 0.05 | ✗ | class 2 | 45.85s |
| 1 | 0.01 | ✓ | — | 0.53s |
| 1 | 0.05 | ✗ | class 2, 3, 8 | 32.45s |
| 2 | 0.01 | ✓ | — | 0.34s |
| 2 | 0.05 | ✓ | — | 0.55s |

All three digits are robust at ε = 0.01. At ε = 0.05, digits 0 and 1 are not robust — adversarial examples exist — while digit 2 remains robust even at the larger radius.

---

## Code Explanation

### `train.py`

Loads a 5 000-sample MNIST subset via scikit-learn's `fetch_openml`, trains a small `MLPClassifier` (784 → 32 → 16 → 10, ReLU, 30 iterations), and manually constructs an ONNX graph from the learned weights using `onnx.helper`. `onnx.shape_inference.infer_shapes` is called before saving to populate intermediate tensor shapes, which is required by Marabou's ONNX parser.

### `src/model_loader.py`

Calls `Marabou.read_onnx()` and returns a `_NetworkInfo` object that bundles the network with flat arrays of input variable indices (shape `(784,)`) and output variable indices (shape `(10,)`).

### `src/constraints.py`

- `apply_input_constraints`: sets per-pixel lower/upper bounds for the L∞-ball, clipped to `[0, 1]`
- `apply_rival_constraint`: adds the inequality `output[true] − output[rival] ≤ 0` via `network.addInequality`, encoding the negated robustness property

### `src/verify.py`

Iterates over the 9 rival classes, reloading a fresh network for each query. Calls `network.solve(verbose=False)` and extracts the adversarial input from the satisfying assignment when `exit_code == 'sat'`. Results are printed to stderr (to avoid being buried in Marabou's engine log) and saved to `results/verification_results.txt`.

### `test.py`

Thin wrapper over `verify_robustness`. Verifies digits 0, 1, 2 at ε = 0.01 and ε = 0.05 and persists all results.
