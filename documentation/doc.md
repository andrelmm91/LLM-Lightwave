# LLM Lightwave: Complex-Valued Neural Architecture

**LLM Lightwave** is a photonic-inspired language model that utilizes **Complex Numbers** ($\mathbb{C}$) and **Multi-Layer Cascaded Interference** to achieve efficient, phase-aware text generation.

## Core Technology: Complex-Valued Neural Networks (CVNNs)

Unlike standard LLMs which operate entirely in the real-number domain, Lightwave leverages the rich dynamics of complex arithmetic.

### 1. Complex-Valued Embeddings
Tokens are projected into a complex space ($a + bi$). 
- **Magnitude**: Represents the "intensity" or "strength" of a feature.
- **Phase**: Encodes "timing," "order," and "relational" context naturally.

### 2. Multi-Head Modulator
Traditional Self-Attention has $O(N^2)$ complexity. Lightwave uses a **Multi-Head Modulator**:
- Calculates how the current token modulates the accumulated historical field (`z_cache`).
- Uses **Relative Positional Bias** for sequence awareness without the heavy cost of standard attention.

### 3. Evolutionary Incremental Step
The model is a **Dynamical System** where each token acts as an impulse that evolves the internal complex state. This transition from static embeddings to evolving fields allows for constant-time inference.

---

## Multi-Layer Photonic Architecture

The model uses a **Deep Cascaded Architecture** to simulate multiple stages of lightwave propagation.

`Input → Embedding → [ Layer 1 ] → [ Layer 2 ] → ... → [ Layer L ] → Readout`

### Photonic Interference Layer
Each layer implements a physical interaction stage:
1.  **Interference**: The current state interacts with the global historical field.
2.  **Modulation**: Uses either a Neural Modulator or a Fixed Quantum Wave (QRW).
3.  **Non-Linearity**: Applies `tanh` activation (simulating photonic saturable absorbers).
4.  **Normalization**: Maintains field intensity via a learnable `norm_scale`.

---

## Modes of Operation

### 1. Neural Modulator (`--mode neural`)
Learned multi-head interaction patterns. Best for capturing complex linguistic nuances.

### 2. Quantum Wave Modulator (`--mode wave`)
Strictly follows a **Discrete-time Quantum Random Walk (QRW)**:
$\begin{pmatrix} u \\ v \end{pmatrix}_{next} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix} \begin{pmatrix} u_{past} \\ v_{curr} \end{pmatrix}$
This mode has **Zero** trainable parameters in the modulator core, relying on pure unitary interference.

---

## Command Line Interface (CLI)

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--train` | Start training | False |
| `--generate` | Generate text | False |
| `--load` | Load checkpoint before running | False |
| `--checkpoint [FILE]` | Specific checkpoint file to load | `model.pth` |
| `--mode [neural\|wave]` | Modulator architecture | `neural` |
| `--layers [N]` | Number of cascaded layers | 4 |
| `--epochs [N]` | Training epochs | 15 |
| `--steps [N]` | Steps per epoch | 200 |
| `--prompt "[TEXT]"` | Prompt for generation | (TinyStories intro) |

### Example Commands
- **Train Depth-8 Wave Model**: `python llm_light.py --train --mode wave --layers 8`
- **Generate from Specific Version**: `python llm_light.py --generate --load --checkpoint model --prompt "there was a dog"`

---

## Checkpointing
The script automatically saves two files after each epoch:
- `model.pth`: The latest state for easy resuming.
- `model_[mode]_[timestamp].pth`: A versioned archive for history tracking and comparison.

The `load_checkpoint` logic automatically detects the saved mode and layer count, re-configuring the script to match the saved architecture perfectly.