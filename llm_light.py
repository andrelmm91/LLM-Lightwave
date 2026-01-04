import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import random
import os
import tiktoken 
import argparse
from datetime import datetime

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
STEPS_PER_EPOCH = 200
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
DATA_PATH = "TinyStories-train.txt"  # ← place downloaded file here

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

    def forward(self, rel_dist):
        idx = rel_dist + REL_MAX_DIST
        if torch.is_tensor(idx):
            idx = torch.clamp(idx, 0, len(self.bias) - 1)
        else:
            idx = max(0, min(idx, len(self.bias) - 1))
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
        biases = self.rel_bias(rel_dists) # [N]
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

class QuantumWaveModulator(torch.nn.Module):
    """
    Fixed Unitary Modulator based on Discrete-time Quantum Random Walk.
    Matrix: 1/sqrt(2) * [[1, i], [i, 1]]
    """
    def __init__(self):
        super().__init__()
        # Coin matrix
        self.register_buffer("coin", torch.tensor([
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.0 + 1.0j, 1.0 + 0.0j]
        ], dtype=torch.complex64) / (2.0**0.5))

    def forward(self, curr_pos, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros_like(z_curr)
        u_prev = z_past[-1, 0]
        v_curr = z_curr[1]
        state = torch.stack([u_prev, v_curr])
        out = (self.coin * COUPLING) @ state
        return out

class PhotonicInterferenceLayer(torch.nn.Module):
    """One full interference layer/block"""
    def __init__(self, mode="neural"):
        super().__init__()
        self.mode = mode
        if mode == "neural":
            self.modulator = MultiHeadModulator()
        else:
            self.modulator = QuantumWaveModulator()
        self.norm_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, candidate, z_cache, pos):
        if not z_cache:
            z_new = candidate
        else:
            z_past = torch.stack(z_cache)
            mod_factor = self.modulator(pos, candidate, z_past)
            if self.mode == "wave":
                z_new = mod_factor + (1 - COUPLING) * candidate
            else:
                z_new = candidate + COUPLING * mod_factor

        # Photonic-style non-linearity and normalization
        z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
        max_int = torch.max(torch.abs(z_new)**2)
        if max_int > 0:
            z_new = z_new * (self.norm_scale / torch.sqrt(max_int + EPS))
        return z_new

class MultiLayerPhotonicNN(torch.nn.Module):
    def __init__(self, mode="neural", num_layers=4):
        super().__init__()
        self.mode = mode
        self.token_embed = TokenEmbedding(VOCAB_SIZE)
        self.pos_bias = RelativePositionalBias()
        self.layers = torch.nn.ModuleList([PhotonicInterferenceLayer(mode=mode) for _ in range(num_layers)])
        self.readout = ReadoutHead()

    def forward_incremental(self, token_ids):
        seq_len = token_ids.size(0)
        z_cache = []
        logits_list = []

        for i in range(seq_len):
            z_current = self.token_embed(token_ids[i:i+1]).squeeze(0)
            # Add relative positional bias
            z_current = z_current + self.pos_bias(i)
            
            for layer in self.layers:
                z_current = layer(z_current, z_cache, i)
            if i < seq_len - 1:
                logits_list.append(self.readout(z_current))
            z_cache.append(z_current)
        return torch.stack(logits_list)

# Model placeholder
model = None
optimizer = None
scheduler = None
current_mode = "neural" # default

def init_model(mode="neural", num_layers=4):
    global model, optimizer, scheduler, current_mode
    current_mode = mode
    model = MultiLayerPhotonicNN(mode=mode, num_layers=num_layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

def save_checkpoint(path=None):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'mode': current_mode,
        'num_layers': len(model.layers)
    }
    
    torch.save(checkpoint, "model.pth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dynamic_path = f"model_{current_mode}_{timestamp}.pth"
    torch.save(checkpoint, dynamic_path)
    print(f"Checkpoints saved: model.pth and {dynamic_path}")

def load_checkpoint(path="model.pth"):
    global current_mode, optimizer, scheduler, model
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return False
    
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    
    saved_mode = checkpoint.get('mode', 'neural')
    saved_layers = checkpoint.get('num_layers', 4)

    # Re-initialize model to match saved architecture if necessary
    if saved_mode != current_mode or len(model.layers) != saved_layers:
        print(f"Switching to mode {saved_mode} with {saved_layers} layers...")
        current_mode = saved_mode
        init_model(mode=saved_mode, num_layers=saved_layers)

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    print(f"Checkpoint loaded: {path} (Mode: {saved_mode}, Layers: {saved_layers})")
    return True

# ────────────────────────────────────────────────────────────────
#  Training loop (same structure, real chunks)
# ────────────────────────────────────────────────────────────────
def train():
    print("Training started on TinyStories with real BPE tokenizer...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        steps = 0

        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            while steps < STEPS_PER_EPOCH:
                seq = get_random_chunk(SEQ_LEN)
                logits = model.forward_incremental(seq)
                target = seq[1:]

                loss = F.cross_entropy(logits, target, ignore_index=tokenizer.eot_token)
                loss.backward()
                total_loss += loss.item()
                steps += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        print(f"Epoch {epoch+1:2d}  Avg loss: {total_loss/steps:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")
        save_checkpoint()

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Generation with real tokenizer
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new=180, temperature=0.92):
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), device=DEVICE)
    z_cache = []

    # Process prompt tokens
    logits = None
    for i in range(len(prompt_tokens)):
        token_id = prompt_tokens[i:i+1]
        z_curr = model.token_embed(token_id).squeeze(0)
        z_curr = z_curr + model.pos_bias(i)
        
        for layer in model.layers:
            z_curr = layer(z_curr, z_cache, i)
        z_cache.append(z_curr)
        logits = model.readout(z_curr)

    generated_tokens = prompt_tokens.clone()
    current_pos = len(prompt_tokens)

    for _ in range(max_new):
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens = torch.cat([generated_tokens, next_token])
        if next_token.item() == tokenizer.eot_token:
            break

        # Feed prediction back through the layers
        z_curr = model.token_embed(next_token).squeeze(0)
        z_curr = z_curr + model.pos_bias(current_pos)
        
        for layer in model.layers:
            z_curr = layer(z_curr, z_cache, current_pos)
        z_cache.append(z_curr)
        logits = model.readout(z_curr)
        current_pos += 1

    return tokenizer.decode(generated_tokens.tolist())

# ────────────────────────────────────────────────────────────────
#  Run
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Lightwave: Complex-Valued LLM")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--generate", action="store_true", help="Generate text")
    parser.add_argument("--load", action="store_true", help="Load checkpoint before running")
    parser.add_argument("--checkpoint", type=str, default="model.pth", help="Path to specific checkpoint file")
    parser.add_argument("--mode", type=str, choices=["neural", "wave"], default="neural", help="Modulator mode")
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH, help="Steps per epoch")
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a little girl who loved to", help="Prompt for generation")
    args = parser.parse_args()

    current_mode = args.mode
    EPOCHS = args.epochs
    STEPS_PER_EPOCH = args.steps
    
    # Initialize the model correctly before potentially loading
    init_model(mode=current_mode, num_layers=args.layers)

    if args.load:
        load_checkpoint(args.checkpoint)

    if args.train:
        train()

    if args.generate or not args.train:
        print("\n" + "="*90)
        print(f"Test prompt: {args.prompt!r}")
        generated = generate(args.prompt)
        print(f"Generated continuation:\n{generated}")