# LLM Lightwave: Complex-Valued Neural Architecture

**LLM Lightwave** is a photonic-inspired language model that utilizes **Complex Numbers** ($\mathbb{C}$) and **Multi-Layer Cascaded Interference** to achieve efficient, phase-aware text generation.

## Core Technology: Complex-Valued Neural Networks (CVNNs)

Unlike standard LLMs which operate entirely in the real-number domain, Lightwave leverages the rich dynamics of complex arithmetic.

### 1. Complex-Valued Embeddings
Tokens are projected into a complex space ($a + bi$) with an increased dimensionality of **16** (8 real + 8 imaginary pairs by default).
- **Magnitude**: Represents the "intensity" or "strength" of a feature.
- **Phase**: Encodes "timing," "order," and "relational" context naturally.
- **Dimension Expansion**: The use of 16-dimensional embeddings allows the model to handle higher linguistic complexity compared to earlier versions.
- **Positional Handling**: Instead of a hard sequence limit (`MAX_N`), the model uses a **Relative Positional Window** (defined by `REL_MAX_DIST`). It maintains a unique, high-resolution positional bias for relative distances up to 64 tokens, after which biases are smoothly clamped.

### 2. Mish Activation (Complex Domain)
Lightwave utilizes the **Mish** activation function ($x \cdot \tanh(\text{softplus}(x))$) applied independently to the real and imaginary components.
- **Smooth Gradient**: Unlike `tanh`, Mish avoids early saturation, preserving gradient information during deep propagation.
- **Dynamic Response**: Provides a more nuanced non-linear mapping for photonic field interactions.
- **Unitary Efficiency**: Extremely low-parameter "Wave" mode using Quantum Random Walks.
- **Deep Adaptive Interference**: Multi-stage coupling that learns unique field dynamics for every evolution step.
- **Dynamic Evolution**: A stateful, temporal architecture that treats memory as a field.

### 3. Multi-Head Modulator (Attention Upgrade)
Traditional Self-Attention has $O(N^2)$ complexity. Lightwave uses an optimized **Complex Dot-Product Attention** mechanism:
- **Dot-Product Scores**: Calculates interaction strength using `real(conj(Q) * K)` per head.
- **Positional Scores**: Incorporates **Relative Positional Attention Bias** directly into the attention weights.
- **Evolutionary context**: Instead of a global attention matrix, it modulates the current state as a function of the weighted historical field, maintaining high efficiency while capturing strong relational signals.

### 4. Temporal Evolution Loop (Internal Steps $M$)
To simulate the deep state refinement of photonic systems, each interference layer evolves the complex field **$M$ times** per token.
- **Internal Propagation**: Allows the model to iteratively refine the modulation results before passing them to the next cascaded stage.
- **Configurable Complexity**: The depth of refinement can be tuned via the `--M` flag without increasing the physical number of layers.

### 5. Continuous Evolutionary Step
The model is a **Dynamical System** where each token acts as an impulse that evolves the internal complex state. This transition from static embeddings to evolving fields allows for constant-time inference.

---

## Multi-Layer Photonic Architecture

The model uses a **Deep Cascaded Architecture** to simulate multiple stages of lightwave propagation.

`Input → Embedding → [ Layer 1 ] → [ Layer 2 ] → ... → [ Layer L ] → Readout`

#### Photonic Interference Layer
Each layer implements a physical interaction stage:
1.  **Interference**: The current state interacts with the global historical field.
2.  **Modulation**: Uses either a Neural Modulator or a Fixed Quantum Wave (QRW).
3.  **Triple-Adaptive Coupling**: The coupling strength is optimized across **three dimensions**:
    -   **Epoch Adaptive**: Learned throughout training via backpropagation.
    -   **Step Adaptive**: Each of the $M$ evolution steps in a layer has its own unique coupling $\alpha_m$.
    -   **Layer Adaptive**: Every cascaded layer maintains its own independent coupling profile, allowing the model to learn different interference logic at different depths.
4.  **Non-Linearity**: Applies **Mish** activation (simulating advanced photonic saturable absorbers).
5.  **Normalization**: Uses **Complex LayerNorm** (independent LayerNorm on real/imag) followed by a learnable `norm_scale` to maintain field stability.
6.  **Modulation Dynamics: Amplitude & Phase**: The architecture performs spatio-temporal modulation to learn linguistic patterns:
    -   **Amplitude Modulation**: Managed by the **Triple-Adaptive Coupling**. The model learns an optimal mixing ratio ($\alpha$) for every $M$ step and Layer, gating the signal strength between the modulated candidate and the existing field.
    -   **Phase Modulation**:
        -   *Wave Mode*: Uses learned phase vectors ($\phi_u, \phi_v$) to perform rotations in the complex plane before interference, allowing for constructive/destructive alignment.
        -   *Neural Mode*: Learned complex weights naturally optimize phase rotations and magnitude scaling simultaneously.

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

### 5. Reinforcement Learning (RLVR)
Lightwave supports post-training self-improvement via **Reinforcement Learning from Verifiable Rewards**:
- **Verifier-Based Rewards**: Responses are analyzed by a `RewardVerifier` which scores coherence, length, and task success.
- **Sparse Optimization**: During RL fine-tuning, only the modulator parameters ($\alpha$ and multi-head projections) are updated, preserving the foundational linguistic knowledge while refining the attention policy.
- **Scaling**: Allows the model to improve on specific tasks or styles without massive additional Supervised Fine-Tuning (SFT) data.

### 6. Parameter-Efficient Techniques (PEFT)
To scale for photonic hardware with memory constraints, Lightwave implements:
- **LoRA (Low-Rank Adaptation)**: Injected into all `MultiHeadModulator` projections. Freezes base weights and trains rank=$k$ matrices to adapt the interference patterns.
- **Simulated 4-bit Quantization**: Bakes in symmetric linear quantization approximations during the forward pass, preparing the model for low-precision optical substrates.

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
| `--rl_train` | Start RLVR fine-tuning loop | False |
| `--lora` | Enable Low-Rank Adaptation | False |
| `--lora_rank [N]` | Rank of LoRA adapters | 4 |
| `--quant` | Enable 4-bit quantization simulation | False |
| `--M [N]` | Number of internal evolution steps per layer | 8 |

### Example Commands

#### Training
- **Quick Verification**: `python llm_light.py --train --epochs 1 --steps 5`
  > [!NOTE]
  > Use this to verify data loading, training steps, and the validation loop without waiting for a full run.
- **Standard Neural Training**: `python llm_light.py --train --mode neural --layers 4 --epochs 15`
- **Deep Wave Architecture**: `python llm_light.py --train --mode wave --layers 12`
  > [!TIP]
  > Wave mode uses fixed unitary matrices, requiring much less VRAM but benefiting from deeper cascaded layers.

#### Self-Improvement (RLVR)
- **Quick RL Verification**: `python llm_light.py --rl_train --steps 2 --prompt "Once upon a time"`
  > [!NOTE]
  > Use this to verify the reward logic and policy gradient update without a full fine-tuning run.
- **Standard RL Fine-tuning**: `python llm_light.py --rl_train --load --checkpoint model.pth --steps 100`

#### Efficiency & Scaling
- **LoRA Fine-tuning**: `python llm_light.py --train --lora --lora_rank 8 --load`
  > [!TIP]
  > This is the recommended way to adapt a pre-trained model to new data with minimal VRAM.
- **Quantization-Aware Training**: `python llm_light.py --train --quant`
  > [!NOTE]
  > Simulates 4-bit weights to prepare the model for specialized photonic inference engines.
- **Wave LoRA RLVR**: `python llm_light.py --train --mode wave --layers 12 --lora --quant --rl_train --steps 100`
  > [!TIP]
  > This is the recommended way to adapt a pre-trained model to new data with minimal VRAM.

#### Generation
- **Standard (Greedy)**: `python llm_light.py --generate --prompt "The princess found a"`
- **Beam Search (Default Width 5)**: `python llm_light.py --generate --beam --prompt "In a small house"`
- **Precision Beam Search**: `python llm_light.py --generate --beam --beam_width 10 --prompt "Once upon a time"`
- **Load Best Model & Generate**: `python llm_light.py --generate --load --checkpoint model.pth --beam --prompt "Once upon a time"`

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