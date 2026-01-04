import torch
import torch.nn.functional as F
import numpy as np

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
MAX_N = 128
M = 8                # steps per new token (shallow "layer")
H = 4                # number of heads
COUPLING = 0.12
EPS = 1e-8
VOCAB_SIZE = 32
READOUT_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
#  Vocabulary (same as before)
# ────────────────────────────────────────────────────────────────
vocab = list(" abcdefghijklmnopqrstuvwxyz012345")
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def idx_to_text(indices):
    return "".join(idx_to_char.get(int(i), "?") for i in indices).rstrip()

# ────────────────────────────────────────────────────────────────
#  Multi-head modulation + readout
# ────────────────────────────────────────────────────────────────
class MultiHeadModulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(4, 2) for _ in range(H)  # simple projection per head (input: 2+2=4)
        ])
        self.out_proj = torch.nn.Linear(H, 2)      # combine heads

    def forward(self, z_curr, z_past):
        # z_curr: [1,2], z_past: [k,2] where k = past length
        features = torch.cat([z_curr.repeat(z_past.size(0), 1), z_past], dim=-1)
        
        head_outputs = []
        for head in self.heads:
            h = head(features)                  # [k,2]
            phase_diff = torch.atan2(h[:,1], h[:,0])
            amp_ratio = torch.norm(h, dim=-1) / (torch.norm(z_curr) + EPS)
            mod = torch.tanh(phase_diff) * amp_ratio
            head_outputs.append(mod.unsqueeze(-1))
        
        combined = torch.cat(head_outputs, dim=-1)     # [k, H]
        return self.out_proj(combined.mean(dim=0, keepdim=True))  # [1,2]

class ReadoutHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, VOCAB_SIZE)

    def forward(self, z):
        features = torch.stack([z.real, z.imag], dim=-1)
        return self.proj(features)

modulator = MultiHeadModulator().to(device)
readout = ReadoutHead().to(device)

# ────────────────────────────────────────────────────────────────
#  Incremental evolution step (with KV-cache)
# ────────────────────────────────────────────────────────────────
def incremental_evolve_step(new_z_candidate, z_cache):
    # z_cache: list of previous complex states [t-1, ..., 0]
    if not z_cache:
        # First token: no past → minimal evolution
        z_new = torch.tanh(new_z_candidate.real) + 1j * torch.tanh(new_z_candidate.imag)
        max_int = torch.max(torch.abs(z_new)**2)
        if max_int > 0:
            z_new *= 1.0 / torch.sqrt(max_int)
        return z_new

    z_past_complex = torch.stack(z_cache)               # [past_len, ]
    curr_complex = new_z_candidate.unsqueeze(0)         # [1, ]
    
    # Prepare features for modulator (Real, Imag)
    z_past_feat = torch.stack([z_past_complex.real, z_past_complex.imag], dim=-1)
    curr_feat = torch.stack([curr_complex.real, curr_complex.imag], dim=-1)

    # Multi-head modulation score (scalarized for simplicity)
    mod_factor = modulator(curr_feat, z_past_feat).squeeze()  # [2] → use as complex weight

    # Weighted interference from cache (average style)
    # Weighted interference from cache (average style)
    # Note: Using complex z_past for interference summation
    interf = COUPLING * mod_factor[0] * z_past_complex.mean()   # simplified aggregation

    z_new = new_z_candidate + interf

    # Nonlinearity + norm
    z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
    max_int = torch.max(torch.abs(z_new)**2)
    if max_int > 0:
        z_new *= 1.0 / torch.sqrt(max_int)

    return z_new

# ────────────────────────────────────────────────────────────────
#  Full autoregressive generation
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt_indices, max_new_tokens=40, temperature=0.8):
    # Initialize from prompt
    z_cache = []
    generated = prompt_indices.clone()

    # Prefill phase: evolve prompt tokens incrementally
    for i in range(len(prompt_indices)):
        candidate = 0.5 * torch.randn((), device=device, dtype=torch.complex64)
        z = incremental_evolve_step(candidate, z_cache[:i])
        z_cache.append(z)

    # Generate new tokens
    for _ in range(max_new_tokens):
        # Candidate from previous (simple noise + residual)
        prev_z = z_cache[-1]
        candidate = prev_z + 0.15 * torch.randn_like(prev_z)

        z_new = incremental_evolve_step(candidate, z_cache)

        logits = readout(z_new) / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = probs.argmax()

        generated = torch.cat([generated, next_token.unsqueeze(0)])
        z_cache.append(z_new)

        if next_token.item() == char_to_idx[' ']:  # early stop heuristic
            break

    return generated

# ────────────────────────────────────────────────────────────────
#  Demo
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = "the quick brown fox"
    prompt_idx = torch.tensor([char_to_idx.get(c, 0) for c in prompt.lower()], device=device)

    print("Prompt:", repr(prompt))

    generated_idx = generate(prompt_idx, max_new_tokens=60)
    generated_text = idx_to_text(generated_idx)

    print("Generated continuation:")
    print("  " + repr(generated_text))