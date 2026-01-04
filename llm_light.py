import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import random
import os

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
MAX_N = 512
M = 32                # evolution steps per token
H = 4
COUPLING = 0.12
EPS = 1e-8
VOCAB_SIZE = 96      # increased for real text (basic ASCII + common chars)
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 25          # adjust according to your compute
LR = 6e-4
REL_MAX_DIST = 64    # for relative positional bias clipping
DATA_PATH = "TinyStories-train.txt"  # ← download from HF!

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────────
#  Load real TinyStories text
# ────────────────────────────────────────────────────────────────
def load_tinystories(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TinyStories-train.txt not found!\n"
            "Download from: https://huggingface.co/datasets/roneneldan/TinyStories\n"
            "(use TinyStories-train.txt ~1-2GB)"
        )
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded TinyStories text: {len(text):,} characters")
    return text

full_text = load_tinystories()

# Simple character-level tokenizer (good enough for toy)
chars = sorted(list(set(full_text)))
VOCAB_SIZE = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def get_random_chunk(length=SEQ_LEN):
    start = random.randint(0, len(full_text) - length - 1)
    chunk = full_text[start:start + length]
    return torch.tensor([char_to_idx[c] for c in chunk], dtype=torch.long, device=DEVICE)

# ────────────────────────────────────────────────────────────────
#  Modules
# ────────────────────────────────────────────────────────────────
class RelativePositionalBias(torch.nn.Module):
    """Trainable relative positional bias (complex-valued)"""
    def __init__(self, max_rel_dist=REL_MAX_DIST):
        super().__init__()
        # Bias for relative distances -2*max ... +2*max
        size = 2 * max_rel_dist + 1
        self.bias = torch.nn.Parameter(torch.randn(size, dtype=torch.complex64) * 0.02)

    def get_bias(self, rel_dist):
        # rel_dist can be tensor of shape [...]
        idx = rel_dist + REL_MAX_DIST
        idx = torch.clamp(idx, 0, len(self.bias) - 1)
        return self.bias[idx]

class MultiHeadModulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList([torch.nn.Linear(4, 2) for _ in range(H)])
        self.out_proj = torch.nn.Linear(H*2, 2)
        self.rel_bias = RelativePositionalBias()

    def forward(self, curr_pos: int, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros(2, device=DEVICE, dtype=torch.float32)

        past_len = z_past.size(0)
        rel_dists = torch.arange(curr_pos - past_len, curr_pos, device=DEVICE)  # [past_len]
        rel_biases = self.rel_bias.get_bias(rel_dists)  # [past_len] complex

        # Features: (Real, Imag) for both curr and past
        z_curr_feat = torch.stack([z_curr.real, z_curr.imag], dim=-1) # [1, 2]
        z_past_feat = torch.stack([z_past.real, z_past.imag], dim=-1) # [past_len, 2]
        
        # Expand curr to match past
        features = torch.cat([z_curr_feat.repeat(past_len, 1), z_past_feat], dim=-1) # [past_len, 4]
        
        head_outs = [torch.tanh(head(features)) for head in self.heads] # List of [past_len, 2]
        combined = torch.cat(head_outs, dim=-1).mean(dim=0, keepdim=True) # [1, H*2] (Mean over past_len)

        mod_out = self.out_proj(combined).squeeze() # [2] (Real, Imag vector)
        
        # Combine mod_out (float vector) with rel_biases (complex tensor)
        # Convert mod_out to complex scalar
        mod_complex = torch.complex(mod_out[0], mod_out[1])
        
        # Add average relative bias (simple aggregation)
        final_mod = mod_complex + rel_biases.mean()
        
        return final_mod

class ReadoutHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, VOCAB_SIZE)

    def forward(self, z):
        features = torch.stack([z.real, z.imag], dim=-1)
        return self.proj(features)

# Instantiate
modulator = MultiHeadModulator().to(DEVICE)
readout = ReadoutHead().to(DEVICE)

optimizer = optim.Adam(
    list(modulator.parameters()) + list(readout.parameters()),
    lr=LR, betas=(0.9, 0.98), weight_decay=1e-5
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

# ────────────────────────────────────────────────────────────────
#  Forward pass (teacher forcing, incremental)
# ────────────────────────────────────────────────────────────────
def forward_incremental(indices: torch.Tensor):
    seq_len = indices.size(0)
    z_cache = []
    logits_list = []

    for i in range(seq_len):
        target_idx = indices[i]
        candidate = 0.35 * torch.randn((), device=DEVICE, dtype=torch.complex64)

        z_new = incremental_evolve_step(candidate, z_cache, i)

        if i < seq_len - 1:
            logits = readout(z_new)
            logits_list.append(logits)

        z_cache.append(z_new)

    return torch.stack(logits_list)

# Reuse incremental_evolve_step from previous version (adapted)
def incremental_evolve_step(candidate, z_cache, pos):
    if z_cache:
        z_past = torch.stack(z_cache)
        # Modulator now returns a complex scalar
        mod_factor = modulator(pos, candidate.unsqueeze(0), z_past)
        interf = COUPLING * mod_factor * z_past.mean()
        z_new = candidate + interf
    else:
        z_new = candidate

    z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
    max_int = torch.max(torch.abs(z_new)**2)
    if max_int > 0:
        z_new = z_new * (1.0 / torch.sqrt(max_int))
    return z_new

# ────────────────────────────────────────────────────────────────
#  Training Loop
# ────────────────────────────────────────────────────────────────
def train():
    print("Starting training on TinyStories...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

        for _ in tqdm(range(BATCH_SIZE * 4), desc=f"Epoch {epoch+1}/{EPOCHS}"):  # many steps per epoch
            seq = get_random_chunk(SEQ_LEN)
            logits = forward_incremental(seq)  # [seq_len-1, VOCAB_SIZE]
            target = seq[1:]

            loss = F.cross_entropy(logits, target, reduction='mean')
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(list(modulator.parameters()) + list(readout.parameters()), 1.0)
        optimizer.step()
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Avg loss: {total_loss/(BATCH_SIZE*4):.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Generation (same as before, greedy)
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new=120, temperature=0.9):
    prompt_idx = torch.tensor([char_to_idx.get(c, 0) for c in prompt.lower()], device=DEVICE)
    z_cache = []
    generated = prompt_idx.clone()

    # Prefill
    for i in range(len(prompt_idx)):
        candidate = 0.35 * torch.randn((), device=DEVICE, dtype=torch.complex64)
        z = incremental_evolve_step(candidate, z_cache, i)
        z_cache.append(z)

    # Generate
    current_pos = len(prompt_idx)
    for _ in range(max_new):
        prev = z_cache[-1]
        candidate = prev + 0.15 * torch.randn_like(prev)
        z_new = incremental_evolve_step(candidate, z_cache, current_pos)

        logits = readout(z_new) / temperature
        next_token = F.softmax(logits, dim=-1).argmax()

        generated = torch.cat([generated, next_token.unsqueeze(0)])
        z_cache.append(z_new)
        current_pos += 1

        if next_token.item() == char_to_idx.get(' ', 0):
            break

    return "".join(idx_to_char.get(i.item(), "?") for i in generated)

# ────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n" + "="*80)
    prompt = "Once upon a time there was a little"
    print(f"Prompt: {repr(prompt)}")
    generated = generate(prompt)
    print(f"Generated: {repr(generated)}")