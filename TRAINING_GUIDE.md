# Gomoku AI Training Guide

This guide explains how to set up and train the Deep Learning Gomoku agent locally, specifically targeting systems with an NVIDIA GPU (e.g., RTX 3050).

## 1. Environment Setup

Ensure you have NVIDIA drivers installed on your system.
This project uses `uv` for dependency management. The training dependencies (like `torch`) are isolated in an optional `train` dependency group to keep the production runtime lean.

To install all dependencies (including training tools), run:

```bash
uv sync --all-extras
```

This installs `torch`, `onnxruntime`, and other necessary ML libraries.

## 2. Understanding the Architecture

- **Encoder (`ml/encoder.py`)**: Converts the 15x15 Gomoku board into a $3 \times 15 \times 15$ tensor.
  - Channel 0: Current player's stones.
  - Channel 1: Opponent's stones.
  - Channel 2: Color to play.
  It also handles data augmentation by generating 8-fold dihedral symmetries (rotations and flips) of the board state.
- **Model (`ml/model.py`)**: A PyTorch Dual-Head CNN (`PolicyValueNet`).
  - **Trunk**: Convolutional layers with residual blocks (`ResBlock`).
  - **Policy Head**: Outputs 225 logits corresponding to move probabilities.
  - **Value Head**: Outputs a single scalar evaluation in the range $[-1, 1]$.

## 3. Running Training

The training script (`ml/train.py`) runs self-play simulations and updates the neural network using a combined loss function:

$$ \mathcal{L} = (z - v)^2 - \pi^\top \log p + c \Vert\theta\Vert^2 $$

Where:
- $(z - v)^2$ is the Mean Squared Error of the value prediction $v$ against the actual game outcome $z$.
- $-\pi^\top \log p$ is the Cross-Entropy loss between the target policy $\pi$ and the predicted probabilities $p$.
- $c \Vert\theta\Vert^2$ is the L2 weight regularization term.

To start training, simply run:

```bash
uv run python -m ml.train
```

The script will automatically detect if CUDA is available and train the model on your GPU. Once completed, it saves the PyTorch weights to `gomoku_model.pt`.

## 4. Exporting to ONNX

To deploy the model efficiently without heavy PyTorch dependencies on the backend, we convert it to an ONNX model.

Run the export script:

```bash
uv run python -m ml.export_onnx
```

This will load `gomoku_model.pt` and output the optimized model to `weights/gomoku_model.onnx`.

## 5. Integrating with the Backend

The FastAPI backend uses `DeepLearningAI` (`app/game/dl_ai.py`).
- Upon startup, it looks for `weights/gomoku_model.onnx`.
- If found, it loads the model into an ONNX Runtime `InferenceSession` for extremely fast CPU/GPU inference.
- **Graceful Fallback**: If the `weights/` directory is empty or the model is missing, the backend seamlessly degrades to using the heuristic Minimax agent (`GomokuAI`), ensuring the game is always playable.
