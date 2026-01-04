import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import random
import os
import tiktoken 

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
MAX_N = 512
M = 8                # evolution steps per token
H = 4
COUPLING = 0.12
EPS = 1e-8
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 15
LR = 4e-4
REL_MAX_DIST = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────────
#  Real tokenizer: tiktoken GPT-2 style
# ────────────────────────────────────────────────────────────────
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab  # ~50257

def get_random_chunk(length=SEQ_LEN):
    """Sample random contiguous chunk from TinyStories text"""
    start = random.randint(0, len(full_text) - length - 1)
    chunk = full_text[start:start + length]
    tokens = torch.tensor(tokenizer.encode(chunk, disallowed_special=()), dtype=torch.long, device=DEVICE)
    if len(tokens) < length:
        tokens = F.pad(tokens, (0, length - len(tokens)), value=tokenizer.eot_token)
    return tokens[:length]

# ────────────────────────────────────────────────────────────────
#  Load TinyStories (download manually from HF)
# ────────────────────────────────────────────────────────────────
# DATA_PATH = "debug_data.txt"  # for quick testing
DATA_PATH = "TinyStories-train.txt"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Download TinyStories-train.txt from:\n"
        "https://huggingface.co/datasets/roneneldan/TinyStories\n"
        "(or TinyStoriesV2-GPT4-train.txt for GPT-4 only version)"
    )

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    full_text = f.read()

print(f"Loaded TinyStories: {len(full_text):,} characters | ~{len(full_text)//4:,} tokens")

# ────────────────────────────────────────────────────────────────
#  Modules (same as before + token embedding)
# ────────────────────────────────────────────────────────────────
class TokenEmbedding(torch.nn.Module):
    """Project token ids to complex domain"""
    def __init__(self, vocab_size, dim=2):
        super().__init__()
        self.dim = dim
        self.embed = torch.nn.Embedding(vocab_size, dim * 2)  # real + imag

    def forward(self, ids):
        emb = self.embed(ids)           # [..., dim*2]
        return emb[..., :self.dim] + 1j * emb[..., self.dim:]  # complex

class RelativePositionalBias(torch.nn.Module):
    def __init__(self, max_rel_dist=REL_MAX_DIST):
        super().__init__()
        size = 2 * max_rel_dist + 1
        self.bias = torch.nn.Parameter(torch.randn(size, dtype=torch.complex64) * 0.02)

    def get_bias(self, rel_dist):
        idx = rel_dist + REL_MAX_DIST
        idx = torch.clamp(idx, 0, len(self.bias) - 1)
        return self.bias[idx]

class MultiHeadModulator(torch.nn.Module):
    def __init__(self, dim=2, heads=H):
        super().__init__()
        self.dim = dim
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(dim * 2 * 2, dim * 2) for _ in range(heads)
        ])
        self.out_proj = torch.nn.Linear(heads * dim * 2, dim * 2)
        self.rel_bias = RelativePositionalBias()

    def forward(self, curr_pos: int, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros(self.dim, device=DEVICE, dtype=torch.complex64)

        past_len = z_past.size(0)
        
        # Prepare inputs
        z_curr_rep = z_curr.unsqueeze(0).expand(past_len, -1) # [N, dim]
        
        # Concatenate features: [N, 2*dim] complex
        features_c = torch.cat([z_curr_rep, z_past], dim=-1)
        
        # View as real: [N, 2*dim * 2] floats
        # view_as_real adds a last dim of 2. Flatten that.
        features = torch.view_as_real(features_c).flatten(start_dim=-2)
        
        # Heads
        head_outs = []
        for head in self.heads:
            h = torch.tanh(head(features)) # [N, dim*2]
            head_outs.append(h)
            
        # Cat heads: [N, H * dim * 2]
        combined = torch.cat(head_outs, dim=-1)
        
        # Aggregate over past (mean)
        combined_mean = combined.mean(dim=0) # [H * dim * 2]
        
        # Output projection
        out = self.out_proj(combined_mean) # [dim * 2]
        
        # Back to complex
        mod = torch.view_as_complex(out.view(self.dim, 2))
        
        # Bias
        rel_dists = torch.arange(curr_pos - past_len, curr_pos, device=DEVICE)
        biases = self.rel_bias.get_bias(rel_dists) # [N]
        # Expand biases to [dim]? It is scalar complex.
        # Broadcasting handles it.
        
        return mod + biases.mean()

class ReadoutHead(torch.nn.Module):
    def __init__(self, dim=2, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.proj = torch.nn.Linear(dim * 2, vocab_size)

    def forward(self, z):
        # z: [..., dim] complex
        x = torch.view_as_real(z).flatten(start_dim=-2)
        return self.proj(x)

# Instantiate
token_embed = TokenEmbedding(VOCAB_SIZE).to(DEVICE)
pos_bias = RelativePositionalBias().to(DEVICE)
modulator = MultiHeadModulator().to(DEVICE)  # assume same as previous
readout = ReadoutHead().to(DEVICE)           # output to VOCAB_SIZE logits

optimizer = optim.Adam(
    list(token_embed.parameters()) +
    list(pos_bias.parameters()) +
    list(modulator.parameters()) +
    list(readout.parameters()),
    lr=LR
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

# ────────────────────────────────────────────────────────────────
#  Evolution Step
# ────────────────────────────────────────────────────────────────
def incremental_evolve_step(candidate, z_cache, i):
    # Retrieve previous state
    if not z_cache:
        past = torch.empty(0, 2, device=DEVICE, dtype=torch.complex64)
    else:
        past = torch.stack(z_cache) # [N, dim]

    # Evolution: Modulate based on interaction with past
    mod = modulator(i, candidate, past)
    
    return candidate + mod

# ────────────────────────────────────────────────────────────────
#  Forward pass (teacher forcing) - adapted for subword tokens
# ────────────────────────────────────────────────────────────────
def forward_incremental(token_ids: torch.Tensor):
    seq_len = token_ids.size(0)
    z_cache = []
    logits_list = []

    for i in range(seq_len):
        # Embed current token
        candidate = token_embed(token_ids[i:i+1]).squeeze(0)

        z_new = incremental_evolve_step(candidate, z_cache, i)  # from previous version

        if i < seq_len - 1:
            logits = readout(z_new)
            logits_list.append(logits)

        z_cache.append(z_new)

    return torch.stack(logits_list)  # [seq_len-1, VOCAB_SIZE]

# ────────────────────────────────────────────────────────────────
#  Training loop (same structure, real chunks)
# ────────────────────────────────────────────────────────────────
def train():
    print("Training started on TinyStories with real BPE tokenizer...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0

        with tqdm(total=200, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:  # ~200 steps/epoch
            while steps < 200:
                seq = get_random_chunk(SEQ_LEN)
                logits = forward_incremental(seq)
                target = seq[1:]

                loss = F.cross_entropy(logits, target, ignore_index=tokenizer.eot_token)
                loss.backward()
                total_loss += loss.item()
                steps += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.nn.utils.clip_grad_norm_(list(token_embed.parameters()) +
                                       list(pos_bias.parameters()) +
                                       list(modulator.parameters()) +
                                       list(readout.parameters()), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        print(f"Epoch {epoch+1:2d}  Avg loss: {total_loss/steps:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Generation with real tokenizer
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new=180, temperature=0.92):
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), device=DEVICE)
    z_cache = []

    # Prefill
    for i, tok_id in enumerate(prompt_tokens):
        candidate = token_embed(tok_id.unsqueeze(0)).squeeze(0)
        z = incremental_evolve_step(candidate, z_cache, i)
        z_cache.append(z)

    generated_tokens = prompt_tokens.clone()

    current_pos = len(prompt_tokens)
    for _ in range(max_new):
        prev = z_cache[-1]
        candidate = prev + 0.12 * torch.randn_like(prev)  # some noise
        z_new = incremental_evolve_step(candidate, z_cache, current_pos)

        logits = readout(z_new) / temperature
        next_token = F.softmax(logits, dim=-1).argmax()

        generated_tokens = torch.cat([generated_tokens, next_token])
        z_cache.append(z_new)
        current_pos += 1

        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(generated_tokens.tolist())

# ────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n" + "="*90)
    test_prompt = "Once upon a time there was a little girl who loved to"
    print(f"Test prompt: {test_prompt!r}")
    generated = generate(test_prompt)
    print(f"Generated continuation:\n{generated}")