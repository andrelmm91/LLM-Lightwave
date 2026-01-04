# LLM Lightwave: Complex-Valued Neural Architecture

I have successfully debugged and optimized the [llm_light.py](file:///c:/Users/andre/OneDrive/Documentos/dev/LLM_lightwave/llm_light.py) script. The model is now training on the TinyStories dataset. Below is an overview of the technology behind this model and the benefits it offers.

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