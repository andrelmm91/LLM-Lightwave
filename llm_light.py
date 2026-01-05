import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
from tqdm import tqdm
import random
import os
import tiktoken 
import argparse
from datetime import datetime

# ────────────────────────────────────────────────────────────────
#  Utils
# ────────────────────────────────────────────────────────────────
def mish(x):
    """Mish activation function: x * tanh(softplus(x))"""
    return x * torch.tanh(F.softplus(x))

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
M = 8                # Internal evolution steps (temporal depth) per layer
H = 4                # Number of parallel attention heads in the modulator
INITIAL_COUPLING = 0.12      # Initial coupling strength between field and candidate
SEQ_LEN = 128        # Context window length for training samples
EPOCHS = 15          # Total training passes over the dataset
STEPS_PER_EPOCH = 200 # number of weight updates per epoch
LR = 4e-4            # Initial learning rate for Adam optimizer
EMBED_DIM = 16       # Total dimension: 8 real + 8 imaginary pairs (for 16D CVNN)
BEAM_WIDTH = 5       # Number of candidate beams maintained in beam search
REL_MAX_DIST = 64    # Max relative distance for unique positional biases
GRAD_CLIP = 1.0      # Maximum gradient norm to prevent explosions
LAYERS_DEFAULT = 4   # Default number of cascaded interference stages
VAL_STEPS = 100      # Number of validation chunks to sample each epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ────────────────────────────────────────────────────────────────
#  Real tokenizer: tiktoken GPT-2 style
# ────────────────────────────────────────────────────────────────
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab  # ~50257

def get_random_chunk(text, length=SEQ_LEN):
    """Sample random contiguous chunk from given text"""
    start = random.randint(0, len(text) - length - 1)
    chunk = text[start:start + length]
    tokens = torch.tensor(tokenizer.encode(chunk, disallowed_special=()), dtype=torch.long, device=DEVICE)
    if len(tokens) < length:
        tokens = F.pad(tokens, (0, length - len(tokens)), value=tokenizer.eot_token)
    return tokens[:length]

# ────────────────────────────────────────────────────────────────
#  Load TinyStories (download manually from HF)
# ────────────────────────────────────────────────────────────────
# DATA_PATH = "debug_data.txt"  # for quick testing
DATA_PATH = "tinyStories-train.txt"  # ← place downloaded file here

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Download TinyStories-train.txt from:\n"
        "https://huggingface.co/datasets/roneneldan/TinyStories\n"
        "(or TinyStoriesV2-GPT4-train.txt for GPT-4 only version)"
    )

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    full_text = f.read()

# Rough split: last 5% for validation
split_idx = int(len(full_text) * 0.95)
train_text = full_text[:split_idx]
val_text = full_text[split_idx:]

print(f"Loaded TinyStories: {len(full_text):,} characters")
print(f"Train: {len(train_text):,} | Val: {len(val_text):,}")

# ────────────────────────────────────────────────────────────────
#  Modules (same as before + token embedding)
# ────────────────────────────────────────────────────────────────
class TokenEmbedding(torch.nn.Module):
    """Project token ids to complex domain"""
    def __init__(self, vocab_size, dim=EMBED_DIM):
        super().__init__()
        self.dim = dim
        self.embed = torch.nn.Embedding(vocab_size, dim)  # real part
        self.imag_embed = torch.nn.Embedding(vocab_size, dim)  # imaginary part

    def forward(self, ids):
        real = self.embed(ids)
        imag = self.imag_embed(ids)
        return real + 1j * imag  # shape: [..., dim]

class RelativePositionalBias(torch.nn.Module):
    def __init__(self, max_rel_dist=REL_MAX_DIST, dim=EMBED_DIM):
        super().__init__()
        self.dim = dim
        size = 2 * max_rel_dist + 1
        # Vector bias per position: [size, dim]
        self.bias = torch.nn.Parameter(torch.randn(size, dim, dtype=torch.complex64) * 0.02)

    def forward(self, rel_dist):
        idx = rel_dist + REL_MAX_DIST
        if torch.is_tensor(idx):
            idx = torch.clamp(idx, 0, len(self.bias) - 1)
        else:
            idx = max(0, min(idx, len(self.bias) - 1))
        return self.bias[idx]

class MultiHeadModulator(torch.nn.Module):
    def __init__(self, dim=EMBED_DIM, heads=H):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Projections for Q, K, V
        self.q_proj = torch.nn.Linear(dim * 2, dim * 2)
        self.k_proj = torch.nn.Linear(dim * 2, dim * 2)
        self.v_proj = torch.nn.Linear(dim * 2, dim * 2)
        self.out_proj = torch.nn.Linear(dim * 2, dim * 2)
        
        self.rel_bias = RelativePositionalBias(dim=dim)

    def forward(self, curr_pos: int, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros(self.dim, device=DEVICE, dtype=torch.complex64)

        past_len = z_past.size(0)
        
        # 1. Project to Q, K, V (using real views for Linear layers)
        q_real = torch.view_as_real(z_curr).flatten()
        q_proj = self.q_proj(q_real)
        
        k_past_real = torch.view_as_real(z_past).flatten(start_dim=-2)
        k_proj = self.k_proj(k_past_real)
        
        v_past_real = torch.view_as_real(z_past).flatten(start_dim=-2)
        v_proj = self.v_proj(v_past_real)
        
        # 2. Back to complex and split heads
        q_c = torch.view_as_complex(q_proj.view(self.dim, 2))
        k_c = torch.view_as_complex(k_proj.view(-1, self.dim, 2))
        v_c = torch.view_as_complex(v_proj.view(-1, self.dim, 2))
        
        q_h = q_c.view(self.heads, self.head_dim)          # [H, d]
        k_h = k_c.view(past_len, self.heads, self.head_dim) # [N, H, d]
        v_h = v_c.view(past_len, self.heads, self.head_dim) # [N, H, d]
        
        # 3. Upgrade A: Dot-product style scores
        # scores: real(conj(Q) * K) summed over head_dim
        # Broadcast conj(q_h) across past tokens
        dot_scores = torch.real((torch.conj(q_h).unsqueeze(0) * k_h).sum(dim=-1)) # [N, H]
        
        # 4. Upgrade B: Relative positional attention bias
        rel_dists = torch.arange(curr_pos - past_len, curr_pos, device=DEVICE)
        rel_biases_c = self.rel_bias(rel_dists) # [N, dim]
        rel_biases_h = rel_biases_c.view(past_len, self.heads, self.head_dim)
        
        rel_scores = torch.real((torch.conj(q_h).unsqueeze(0) * rel_biases_h).sum(dim=-1)) # [N, H]
        
        # 5. Combine and Softmax
        final_scores = (dot_scores + rel_scores) / math.sqrt(self.head_dim)
        weights = F.softmax(final_scores, dim=0) # [N, H]
        
        # 6. Weighted sum of values
        # weights: [N, H] -> [N, H, 1]
        out_h = (weights.unsqueeze(-1) * v_h).sum(dim=0) # [H, d]
        
        # 7. Final Projection
        out_c = out_h.view(self.dim)
        out_proj = self.out_proj(torch.view_as_real(out_c).flatten())
        
        return torch.view_as_complex(out_proj.view(self.dim, 2))

class ReadoutHead(torch.nn.Module):
    def __init__(self, dim=EMBED_DIM, vocab_size=VOCAB_SIZE):
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

    def forward(self, curr_pos, z_curr, z_past, coupling=INITIAL_COUPLING):
        if z_past.numel() == 0:
            return torch.zeros_like(z_curr)
        u_prev = z_past[-1, 0]
        v_curr = z_curr[1]
        state = torch.stack([u_prev, v_curr])
        out = (self.coin * coupling) @ state
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
        self.coupling = torch.nn.Parameter(torch.tensor(INITIAL_COUPLING))
        self.norm_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, candidate, z_cache, pos, num_steps=M):
        z_new = candidate
        
        # Internal temporal evolution loop
        for _ in range(num_steps):
            if not z_cache:
                # First token: no interference, just modulation? 
                # Or we can skip loop. Let's keep it simple:
                pass
            else:
                z_past = torch.stack(z_cache)
                if self.mode == "wave":
                    mod_factor = self.modulator(pos, z_new, z_past, coupling=self.coupling)
                    z_new = mod_factor + (1 - self.coupling) * z_new
                else:
                    mod_factor = self.modulator(pos, z_new, z_past)
                    z_new = z_new + self.coupling * mod_factor

            # Photonic-style non-linearity (Mish) and normalization
            z_new = mish(z_new.real) + 1j * mish(z_new.imag)
            # Complex LayerNorm (real and imag independently) + learned scaling
            z_real = F.layer_norm(z_new.real, (z_new.real.size(-1),))
            z_imag = F.layer_norm(z_new.imag, (z_new.imag.size(-1),))
            z_new = (z_real + 1j * z_imag) * self.norm_scale
            
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
                z_current = layer(z_current, z_cache, i, num_steps=M)
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
#  Validation perplexity
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate():
    model.eval()
    total_loss = 0.0
    num_samples = 0

    for _ in range(VAL_STEPS):
        seq = get_random_chunk(val_text)
        logits = model.forward_incremental(seq)
        target = seq[1:]
        loss = F.cross_entropy(logits, target, ignore_index=tokenizer.eot_token, reduction='sum')
        total_loss += loss.item()
        num_samples += target.numel()

    avg_loss = total_loss / num_samples
    perplexity = math.exp(avg_loss)
    return perplexity, avg_loss

# ────────────────────────────────────────────────────────────────
#  Training loop (same structure, real chunks)
# ────────────────────────────────────────────────────────────────
def train():
    print(f"Training started on TinyStories | Val steps: {VAL_STEPS}")
    best_val_ppl = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        steps = 0

        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch+1}/{EPOCHS} Train") as pbar:
            while steps < STEPS_PER_EPOCH:
                seq = get_random_chunk(train_text)
                logits = model.forward_incremental(seq)
                target = seq[1:]

                loss = F.cross_entropy(logits, target, ignore_index=tokenizer.eot_token)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                
                total_train_loss += loss.item()
                steps += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        train_loss = total_train_loss / STEPS_PER_EPOCH

        # Validation
        val_ppl, val_loss = validate()
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_path = f"best_model_valppl_{val_ppl:.2f}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → New best model saved: {save_path}")

        save_checkpoint()

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Final Test after training
# ────────────────────────────────────────────────────────────────
def run_final_tests():
    print("\n" + "="*100)
    print("Final Test Results after training")
    print("="*100)

    val_ppl, val_loss = validate()
    print(f"Final Validation Perplexity: {val_ppl:.2f}  (lower = better)")
    print(f"Final Validation Loss: {val_loss:.4f}\n")

    test_prompts = [
        "Once upon a time there was a little girl who loved to",
        "In a dark forest lived an old wizard who could",
        "The brave knight took his sword and went to",
        "A small dragon woke up from a long sleep and",
        "Deep under the sea there lived a curious mermaid who"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt!r}")
        generated = beam_search_generate(prompt)
        print(f"Generated:\n{generated}\n{'-'*80}")

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
            z_curr = layer(z_curr, z_cache, i, num_steps=M)
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
            z_curr = layer(z_curr, z_cache, current_pos, num_steps=M)
        z_cache.append(z_curr)
        logits = model.readout(z_curr)
        current_pos += 1

    return tokenizer.decode(generated_tokens.tolist())

# ────────────────────────────────────────────────────────────────
#  Beam Search Generation
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def beam_search_generate(prompt: str, beam_width=BEAM_WIDTH, max_new=180, temperature=0.9):
    model.eval()
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), device=DEVICE)
    prompt_len = len(prompt_tokens)

    # Beam element: (sequence_tensor, log_prob, z_cache_list)
    # Start with prefilling the cache for the prompt
    z_cache_init = []
    logits = None
    for i in range(prompt_len):
        token_id = prompt_tokens[i:i+1]
        z_curr = model.token_embed(token_id).squeeze(0)
        z_curr = z_curr + model.pos_bias(i)
        for layer in model.layers:
            z_curr = layer(z_curr, z_cache_init, i, num_steps=M)
        z_cache_init.append(z_curr)
        logits = model.readout(z_curr)

    beams = [(prompt_tokens.clone(), 0.0, z_cache_init)]

    for step in range(max_new):
        new_beams = []
        for seq, logp, z_cache in beams:
            # Get logits for the LAST token in the current sequence
            # Wait, the way incremental works: we need the z_cache to produce logits for NEXT token
            # z_cache[-1] already went through layers, so readout(z_cache[-1]) gives next token logits
            logits = model.readout(z_cache[-1]) / temperature
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(beam_width)

            for k in range(beam_width):
                new_token = top_ids[k:k+1]
                new_logp = logp + torch.log(top_probs[k] + 1e-10).item()
                
                # Check if it was already EOT
                if seq[-1].item() == tokenizer.eot_token:
                    new_beams.append((seq, logp, z_cache))
                    continue

                new_seq = torch.cat([seq, new_token])
                
                # Update cache for the new token
                current_pos = len(seq)
                z_curr = model.token_embed(new_token).squeeze(0)
                z_curr = z_curr + model.pos_bias(current_pos)
                
                new_z_cache = z_cache.copy()
                for layer in model.layers:
                    z_curr = layer(z_curr, new_z_cache, current_pos, num_steps=M)
                new_z_cache.append(z_curr)
                
                new_beams.append((new_seq, new_logp, new_z_cache))

        # Keep top beam_width, penalizing length slightly or just sorting
        # Avoid duplicate sequences if any (though unlikely here)
        unique_beams = {}
        for s, l, c in new_beams:
            s_tuple = tuple(s.tolist())
            if s_tuple not in unique_beams or unique_beams[s_tuple][0] < l:
                unique_beams[s_tuple] = (l, s, c)
        
        beams = sorted(unique_beams.values(), key=lambda x: x[0], reverse=True)[:beam_width]
        beams = [(s, l, c) for l, s, c in beams]

        # Early stop if ALL beams ended with EOT
        if all(b[0][-1].item() == tokenizer.eot_token for b in beams):
            break

    # Return best sequence
    best_seq = beams[0][0]
    return tokenizer.decode(best_seq.tolist())

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
    parser.add_argument("--layers", type=int, default=LAYERS_DEFAULT, help="Number of layers")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH, help="Steps per epoch")
    parser.add_argument("--prompt", type=str, default="Once upon a time there was a little girl who loved to", help="Prompt for generation")
    parser.add_argument("--beam", action="store_true", help="Use beam search for generation")
    parser.add_argument("--beam_width", type=int, default=BEAM_WIDTH, help="Beam width")
    parser.add_argument("--test", action="store_true", help="Run final test suite")
    parser.add_argument("--M", type=int, default=M, help="Internal evolution steps per layer")
    args = parser.parse_args()

    current_mode = args.mode
    EPOCHS = args.epochs
    STEPS_PER_EPOCH = args.steps
    M = args.M
    
    # Initialize the model correctly before potentially loading
    init_model(mode=current_mode, num_layers=args.layers)

    if args.load:
        load_checkpoint(args.checkpoint)

    if args.train:
        train()
        run_final_tests()

    if args.test and not args.train:
        run_final_tests()

    if args.generate or (not args.train and not args.test):
        print("\n" + "="*90)
        print(f"Test prompt: {args.prompt!r}")
        if args.beam:
            print(f"Generating with Beam Search (width={args.beam_width})...")
            generated = beam_search_generate(args.prompt, beam_width=args.beam_width)
        else:
            print("Generating with Greedy Decoding...")
            generated = generate(args.prompt)
        print(f"Generated continuation:\n{generated}")