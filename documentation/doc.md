# LLM Lightwave: Complex-Valued Neural Architecture

**LLM Lightwave** is a photonic-inspired language model that utilizes **Complex Numbers** ($\mathbb{C}$) and **Multi-Layer Cascaded Interference** to achieve efficient, phase-aware text generation.

## Core Technology: Complex-Valued Neural Networks (CVNNs)

Unlike standard LLMs which operate entirely in the real-number domain, Lightwave leverages the rich dynamics of complex arithmetic.

### 1. Complex-Valued Embeddings
Tokens are projected into a complex space ($a + bi$) with an increased dimensionality of **16** (8 real + 8 imaginary pairs by default).
- **Magnitude**: Represents the "intensity" or "strength" of a feature.
- **Phase**: Encodes "timing," "order," and "relational" context naturally.
- **Dimension Expansion**: The use of 16-dimensional embeddings allows the model to handle higher linguistic complexity compared to earlier versions.

### 2. Mish Activation (Complex Domain)
Lightwave utilizes the **Mish** activation function ($x \cdot \tanh(\text{softplus}(x))$) applied independently to the real and imaginary components.
- **Smooth Gradient**: Unlike `tanh`, Mish avoids early saturation, preserving gradient information during deep propagation.
- **Dynamic Response**: Provides a more nuanced non-linear mapping for photonic field interactions.

### 3. Multi-Head Modulator
Traditional Self-Attention has $O(N^2)$ complexity. Lightwave uses a **Multi-Head Modulator**:
- Calculates how the current token modulates the accumulated historical field (`z_cache`).
- Uses **Relative Positional Bias** for sequence awareness without the heavy cost of standard attention.

### 4. Continuous Evolutionary Step
The model is a **Dynamical System** where each token acts as an impulse that evolves the internal complex state. This transition from static embeddings to evolving fields allows for constant-time inference.

---

## Multi-Layer Photonic Architecture

The model uses a **Deep Cascaded Architecture** to simulate multiple stages of lightwave propagation.

`Input → Embedding → [ Layer 1 ] → [ Layer 2 ] → ... → [ Layer L ] → Readout`

#### Photonic Interference Layer
Each layer implements a physical interaction stage:
1.  **Interference**: The current state interacts with the global historical field.
2.  **Modulation**: Uses either a Neural Modulator or a Fixed Quantum Wave (QRW).
3.  **Learnable Coupling**: The coupling strength is a per-layer learnable parameter ($\alpha$), initialized at 0.12, allowing the model to adaptively tune interference levels.
4.  **Non-Linearity**: Applies **Mish** activation (simulating advanced photonic saturable absorbers).
5.  **Normalization**: Maintains field intensity via a learnable `norm_scale`.

---

## Modes of Operation

### 1. Neural Modulator (`--mode neural`)
Learned multi-head interaction patterns. Best for capturing complex linguistic nuances.

### 2. Quantum Wave Modulator (`--mode wave`)
Strictly follows a **Discrete-time Quantum Random Walk (QRW)**:
$\begin{pmatrix} u \\ v \end{pmatrix}_{next} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix} \begin{pmatrix} u_{past} \\ v_{curr} \end{pmatrix}$
This mode has **Zero** trainable parameters in the modulator core, relying on pure unitary interference.

### 3. Beam Search Decoding
Beyond greedy decoding, Lightwave now supports **Beam Search**:
- Maintains multiple top-k hypotheses (beams) during generation.
- Corrects potential local errors by exploring multiple high-probability paths simultaneously.
- Configurable via `--beam` and `--beam_width`.

### 4. Validation Perplexity Tracking
The model now includes automated performance monitoring:
- **Dataset Split**: 95/5 train/val split is performed automatically.
- **Perplexity ($PPL$):** Tracked after each epoch on held-out validation data ($PPL = e^{loss}$).
- **Best Model Saving**: Automatically saves the model with the lowest validation perplexity as `best_model_valppl_*.pt`.

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
| `--beam` | Use beam search decoding | False |
| `--beam_width [N]` | Number of beams for search | 5 |
| `--test` | Run final validation & tests | False |

### Example Commands

#### Training
- **Quick Verification**: `python llm_light.py --train --epochs 1 --steps 5`
  > [!NOTE]
  > Use this to verify data loading, training steps, and the validation loop without waiting for a full run.
- **Standard Neural Training**: `python llm_light.py --train --mode neural --layers 4 --epochs 15`
- **Deep Wave Architecture**: `python llm_light.py --train --mode wave --layers 12`
  > [!TIP]
  > Wave mode uses fixed unitary matrices, requiring much less VRAM but benefiting from deeper cascaded layers.

#### Generation
- **Standard (Greedy)**: `python llm_light.py --generate --prompt "The princess found a"`
- **Beam Search (Default Width 5)**: `python llm_light.py --generate --beam --prompt "In a small house"`
- **Precision Beam Search**: `python llm_light.py --generate --beam --beam_width 10 --prompt "Once upon a time"`
- **Load Best Model & Generate**: `python llm_light.py --generate --load --checkpoint best_model_valppl_15.39.pt --beam`

#### Evaluation & Testing
- **Standalone Evaluation**: `python llm_light.py --test --load --checkpoint model.pth`
  > [!NOTE]
  > This runs a full validation perplexity check and generates text for 5 distinct test prompts to assess qualitative quality.
---

## Checkpointing
The script automatically saves two files after each epoch:
- `model.pth`: The latest state for easy resuming.
- `model_[mode]_[timestamp].pth`: A versioned archive for history tracking and comparison.
- `best_model_valppl_*.pt`: The model state that achieved the lowest validation perplexity during training.

The `load_checkpoint` logic automatically detects the saved mode and layer count, re-configuring the script to match the saved architecture perfectly.