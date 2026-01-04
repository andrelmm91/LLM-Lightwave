# LLM Lightwave: Complex-Valued Neural Architecture

The model is now training on the TinyStories dataset. Below is an overview of the technology behind this model and the benefits it offers.

## Core Technology: Complex-Valued Neural Networks (CVNNs)

Unlike standard Large Language Models (LLMs) which operate entirely in the real-number domain, **LLM Lightwave** utilizes **Complex Numbers** ($\mathbb{C}$) for its internal state and weights.

### 1. Complex-Valued Embeddings
Tokens are projected into a complex space ($a + bi$). This allows each dimension to carry two distinct pieces of information:
*   **Magnitude**: Often corresponding to the "strength" or "presence" of a feature.
*   **Phase**: Representing the "timing," "order," or "relational" context of the feature.

### 2. Multi-Head Modulator vs. Self-Attention
Traditional Transformers use "Self-Attention," which compares every token to every other token ($O(N^2)$ complexity). 
**Lightwave** uses a **Multi-Head Modulator**:
*   Instead of a massive attention matrix, it calculates how the current token "modulates" the accumulated history (`z_cache`).
*   It uses a **weighted mean-aggregation** with **Relative Positional Bias**, which is mathematically much lighter than standard attention while still maintaining a global memory.

### 3. Evolutionary Incremental Step
The model is designed as a **Dynamical System**. Each token acts as an impulse that evolves the previous internal state. This is more akin to a **State Space Model (SSM)** or a sophisticated **Recurrent Neural Network (RNN)**, but with the expressive power of complex dynamics.

---

## Key Benefits

| Benefit | Description |
| :--- | :--- |
| **Efficiency** | The model is extremely lightweight. It achieves sequence-aware generation with a fraction of the parameters of a standard Transformer. |
| **Phase-Awareness** | Complex numbers naturally represent periodic and structural patterns in language (like grammar and rhythm) more efficiently through phase-shifting. |
| **Constant-Time Inference** | Because it evolves the state incrementally, generating the next token is extremely fast and doesn't require re-processing the entire sequence history like some traditional models. |
| **Memory Efficiency** | The `z_cache` approach stores historical context as a compressed complex state rather than a growing set of KD/V vectors. |

## Modes: Neural vs. Quantum Wave
I have implemented two distinct modes for the core dynamics:

### 1. Neural Modulator (`--mode neural`)
*   **Mechanism**: Learned multi-head attention-like interaction.
*   **Trainable Parameters**: ~8,000 in the modulator.

### 2. Quantum Wave Modulator (`--mode wave`)
*   **Mechanism**: Strictly follows your QRW equation:
    $\begin{pmatrix} u \\ v \end{pmatrix}_{next} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix} \begin{pmatrix} u_{past} \\ v_{curr} \end{pmatrix}$
*   **Trainable Parameters**: **ZERO**. Only Embeddings/Readout are trained.

## Usage
- **Neural**: `python llm_light.py --train --mode neural`
- **Quantum Wave**: `python llm_light.py --train --mode wave`

## Code Deep Dive: Interference with Neighbour States
The "interference" or interaction between states (neighbouring tokens) is handled in the [MultiHeadModulator](file:///c:/Users/andre/OneDrive/Documentos/dev/LLM_lightwave/llm_light.py#88-139) class. 

In this architecture, the current token state (`z_curr`) "interferes" with the entire history of past states (`z_past`) to determine the next evolutionary nudge.

```python
# llm_light.py:L107-123
# 1. Broaden current state to match history length
z_curr_rep = z_curr.unsqueeze(0).expand(past_len, -1) 

# 2. Concatenate every history state with the current state (The Interference)
features_c = torch.cat([z_curr_rep, z_past], dim=-1)

# 3. Process through multiple heads (Interference patterns)
for head in self.heads:
    h = torch.tanh(head(features)) # Nonlinear interaction
    head_outs.append(h)

# 4. Aggregate findings (The resulting modulation)
combined_mean = combined.mean(dim=0) 
```

### Automated Tests
- Run `python llm_light.py --mode wave --train --epochs 1` to verify it can learn with a fixed wave core.
- Compare the loss curves (neural vs. wave).

## Command Line Interface

Command Line Interface: You can now run the script with specific flags:
python llm_light.py --train : Start training from scratch.
python llm_light.py --train --load : Resume training from model.pth.
python llm_light.py --generate --load : Generate text using a saved model.
python llm_light.py --prompt "Your text here" --load : Generate with a custom prompt.