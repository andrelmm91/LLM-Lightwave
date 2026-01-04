import torch
import torch.nn.functional as F
import numpy as np

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
MAX_N = 256          # increased context length potential
M = 8                # evolution steps per new token
H = 4                # multi-head
COUPLING = 0.12
EPS = 1e-8
VOCAB_SIZE = 32
READOUT_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
#  Vocabulary (same)
# ────────────────────────────────────────────────────────────────
vocab = list(" abcdefghijklmnopqrstuvwxyz012345")
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def idx_to_text(indices):
    return "".join(idx_to_char.get(int(i), "?") for i in indices).rstrip()

# ────────────────────────────────────────────────────────────────
#  Modules
# ────────────────────────────────────────────────────────────────
class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len=MAX_N):
        super().__init__()
        # Complex-valued learnable positional embeddings
        self.pe = torch.nn.Parameter(
            0.2 * torch.randn(max_len, dtype=torch.complex64)
        )

    def forward(self, pos: int):
        return self.pe[pos % MAX_N]

class MultiHeadModulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(2, 2) for _ in range(H)
        ])
        self.out_proj = torch.nn.Linear(H*2, 2)

    def forward(self, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros(2, device=device, dtype=torch.complex64)
            
        features = torch.cat([z_curr.repeat(z_past.size(0), 1), z_past], dim=-1)
        head_outputs = []
        for head in self.heads:
            h = head(features)
            phase_diff = torch.atan2(h[:,1], h[:,0])
            amp_ratio = torch.norm(h, dim=-1) / (torch.norm(z_curr) + EPS)
            mod = torch.tanh(phase_diff) * amp_ratio
            head_outputs.append(mod.unsqueeze(-1))
        combined = torch.cat(head_outputs, dim=-1)
        return self.out_proj(combined.mean(dim=0, keepdim=True)).squeeze()

class ReadoutHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, VOCAB_SIZE)

    def forward(self, z):
        features = torch.stack([z.real, z.imag], dim=-1)
        return self.proj(features)

# Instantiate modules
pos_enc = LearnedPositionalEncoding().to(device)
modulator = MultiHeadModulator().to(device)
readout = ReadoutHead().to(device)

# ────────────────────────────────────────────────────────────────
#  Incremental evolution step with positional encoding
# ────────────────────────────────────────────────────────────────
def incremental_evolve_step(new_z_candidate, z_cache, current_position: int):
    # Add learned positional encoding
    pe = pos_enc(current_position)
    candidate_with_pos = new_z_candidate + pe

    if not z_cache:
        z_new = torch.tanh(candidate_with_pos.real) + 1j * torch.tanh(candidate_with_pos.imag)
        max_int = torch.max(torch.abs(z_new)**2)
        if max_int > 0:
            z_new *= 1.0 / torch.sqrt(max_int)
        return z_new

    z_past = torch.stack(z_cache)
    curr = candidate_with_pos.unsqueeze(0)

    # Multi-head modulation
    mod_factor = modulator(curr, z_past)  # [2] complex

    # Simplified interference aggregation from cache
    interf = COUPLING * mod_factor * z_past.mean()

    z_new = candidate_with_pos + interf

    # Nonlinearity + normalization
    z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
    max_int = torch.max(torch.abs(z_new)**2)
    if max_int > 0:
        z_new *= 1.0 / torch.sqrt(max_int)

    return z_new

# ────────────────────────────────────────────────────────────────
#  Full autoregressive generation with KV-cache + pos enc
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt_indices, max_new_tokens=80, temperature=0.85):
    prompt_len = len(prompt_indices)
    z_cache = []
    generated = prompt_indices.clone()

    # 1. Prefill: process prompt tokens with positions 0...prompt_len-1
    for i in range(prompt_len):
        # Start with small random + position
        candidate = 0.4 * torch.randn((), device=device, dtype=torch.complex64)
        z = incremental_evolve_step(candidate, z_cache[:i], current_position=i)
        z_cache.append(z)

    # 2. Generate new tokens
    current_pos = prompt_len
    for _ in range(max_new_tokens):
        prev_z = z_cache[-1]
        candidate = prev_z + 0.18 * torch.randn_like(prev_z)

        z_new = incremental_evolve_step(candidate, z_cache, current_position=current_pos)

        logits = readout(z_new) / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = probs.argmax()

        generated = torch.cat([generated, next_token.unsqueeze(0)])
        z_cache.append(z_new)
        current_pos += 1

        if next_token.item() == char_to_idx[' ']:  # early stop example
            break

    return generated

# ────────────────────────────────────────────────────────────────
#  Demo run
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "once upon a time in a land far away"
    prompt_idx = torch.tensor(
        [char_to_idx.get(c.lower(), 0) for c in prompt],
        device=device
    )

    print("Prompt:", repr(prompt))

    generated_idx = generate(prompt_idx, max_new_tokens=100)
    generated_text = idx_to_text(generated_idx)

    print("\nGenerated continuation (with learned positional encodings):")
    print("  " + repr(generated_text))