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

def simulate_quantization(weight, bits=4):
    """Simulate symmetric linear quantization during training"""
    if bits is None:
        return weight
    qmin = -(2**(bits - 1))
    qmax = 2**(bits - 1) - 1
    
    # Scale based on max absolute value
    max_val = torch.max(torch.abs(weight))
    if max_val == 0:
        return weight
    
    scale = max_val / qmax
    q_weight = torch.round(weight / (scale + 1e-8))
    q_weight = torch.clamp(q_weight, qmin, qmax)
    return q_weight * scale

class LoRALinear(torch.nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA) and simulated quantization"""
    def __init__(self, in_features, out_features, rank=4, alpha=16, use_lora=False, quant_bits=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.use_lora = use_lora
        self.quant_bits = quant_bits
        
        # Base weights (frozen if use_lora is True)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        
        if use_lora:
            # Low-rank matrices: B (out x rank), A (rank x in)
            self.lora_A = torch.nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
            self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
            self.scaling = alpha / rank
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    def forward(self, x):
        # 1. Simulate quantization on weights
        w = simulate_quantization(self.weight, bits=self.quant_bits)
        b = simulate_quantization(self.bias, bits=self.quant_bits)
        
        # 2. Main linear pass
        out = F.linear(x, w, b)
        
        # 3. LoRA path: out += (x @ A.T @ B.T) * scaling
        if self.use_lora and self.lora_A is not None:
            lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
            out = out + lora_out * self.scaling
            
        return out

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
M = 8                # Internal evolution steps (temporal depth) per layer
H = 4                # Number of parallel attention heads in the modulator
INITIAL_COUPLING = 0.5      # Initial coupling strength between field and candidate
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
# DATA_PATH = "./training dataset/debug_data.txt"  # for quick testing
DATA_PATH = "./training dataset/tinyStories-train.txt"  # place downloaded file here

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
    def __init__(self, dim=EMBED_DIM, heads=H, use_lora=False, lora_rank=4, quant_bits=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        
        # Projections for Q, K, V
        self.q_proj = LoRALinear(dim * 2, dim * 2, rank=lora_rank, use_lora=use_lora, quant_bits=quant_bits)
        self.k_proj = LoRALinear(dim * 2, dim * 2, rank=lora_rank, use_lora=use_lora, quant_bits=quant_bits)
        self.v_proj = LoRALinear(dim * 2, dim * 2, rank=lora_rank, use_lora=use_lora, quant_bits=quant_bits)
        self.out_proj = LoRALinear(dim * 2, dim * 2, rank=lora_rank, use_lora=use_lora, quant_bits=quant_bits)
        
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
    def __init__(self, dim=EMBED_DIM, vocab_size=VOCAB_SIZE, use_lora=False, quant_bits=None):
        super().__init__()
        self.proj = LoRALinear(dim * 2, vocab_size, use_lora=use_lora, quant_bits=quant_bits)

    def forward(self, z):
        # z: [..., dim] complex
        x = torch.view_as_real(z).flatten(start_dim=-2)
        return self.proj(x)

class QuantumWaveModulator(torch.nn.Module):
    """
    Enhanced Unitary Modulator based on Discrete-time Quantum Random Walk.
    Features learnable phase shifts for increased expressivity.
    """
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        # Coin matrix (Fixed Unitary)
        self.register_buffer("coin", torch.tensor([
            [1.0 + 0.0j, 0.0 + 1.0j],
            [0.0 + 1.0j, 1.0 + 0.0j]
        ], dtype=torch.complex64) / (2.0**0.5))
        
        # Learnable phase shifts (vectors)
        self.phase_u = torch.nn.Parameter(torch.zeros(dim))
        self.phase_v = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, curr_pos, z_curr, z_past_stacked, coupling=INITIAL_COUPLING):
        if z_past_stacked is None:
            return torch.zeros_like(z_curr)
            
        # 1. Capture the 'previous' state from the historical field
        # In a multi-layer cascade, we often look at the immediate predecessor
        u_prev = z_past_stacked[-1] # [dim]
        v_curr = z_curr             # [dim]
        
        # 2. Apply learnable phase rotations before interference
        # exp(i * phase)
        u_prev = u_prev * torch.exp(1j * self.phase_u)
        v_curr = v_curr * torch.exp(1j * self.phase_v)
        
        # 3. Apply Unitary Coin (Interference)
        state = torch.stack([u_prev, v_curr], dim=0) # [2, dim]
        out = (self.coin * coupling) @ state # [2, dim]
        
        return out[0] # [dim]

class PhotonicInterferenceLayer(torch.nn.Module):
    """One full interference layer/block"""
    def __init__(self, mode="neural", use_lora=False, lora_rank=4, quant_bits=None):
        super().__init__()
        self.mode = mode
        if mode == "neural":
            self.modulator = MultiHeadModulator(use_lora=use_lora, lora_rank=lora_rank, quant_bits=quant_bits)
        else:
            self.modulator = QuantumWaveModulator(dim=EMBED_DIM)
        self.coupling = torch.nn.Parameter(torch.full((M,), INITIAL_COUPLING))
        self.norm_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, candidate, z_cache, pos, num_steps=M):
        z_new = candidate
        
        # Optimization: stack history once per token, not M times
        z_past_stacked = torch.stack(z_cache) if z_cache else None
        
        # Internal temporal evolution loop
        for m in range(num_steps):
            if z_past_stacked is None:
                # First token: no interference
                pass
            else:
                c = self.coupling[m] # Multi-stage adaptive coupling
                if self.mode == "wave":
                    mod_factor = self.modulator(pos, z_new, z_past_stacked, coupling=c)
                    z_new = mod_factor + (1 - c) * z_new
                else:
                    mod_factor = self.modulator(pos, z_new, z_past_stacked)
                    z_new = z_new + c * mod_factor

            # Photonic-style non-linearity (Mish) and normalization
            z_new = mish(z_new.real) + 1j * mish(z_new.imag)
            # Complex LayerNorm (real and imag independently) + learned scaling
            z_real = F.layer_norm(z_new.real, (z_new.real.size(-1),))
            z_imag = F.layer_norm(z_new.imag, (z_new.imag.size(-1),))
            z_new = (z_real + 1j * z_imag) * self.norm_scale
            
        return z_new

class MultiLayerPhotonicNN(torch.nn.Module):
    def __init__(self, num_layers=LAYERS_DEFAULT, mode="neural", use_lora=False, lora_rank=4, quant_bits=None):
        super().__init__()
        self.num_layers = num_layers
        self.mode = mode
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.quant_bits = quant_bits
        
        self.token_embed = TokenEmbedding(VOCAB_SIZE)
        self.pos_bias = RelativePositionalBias()
        self.layers = torch.nn.ModuleList([
            PhotonicInterferenceLayer(mode=mode, use_lora=use_lora, lora_rank=lora_rank, quant_bits=quant_bits) for _ in range(num_layers)
        ])
        self.readout = ReadoutHead(use_lora=use_lora, quant_bits=quant_bits)

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

def init_model(mode="neural", num_layers=LAYERS_DEFAULT, use_lora=False, lora_rank=4, quant_bits=None):
    global model, optimizer, scheduler, current_mode
    current_mode = mode # Keep current_mode updated globally
    model = MultiLayerPhotonicNN(num_layers=num_layers, mode=mode, use_lora=use_lora, lora_rank=lora_rank, quant_bits=quant_bits).to(DEVICE)
    print(f"Photonic Model Initialized (Mode: {mode}, Layers: {num_layers}, LoRA: {use_lora} (rank {lora_rank}), Quant: {quant_bits}-bit)")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

def save_checkpoint(path=None):
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'mode': model.mode,
        'num_layers': model.num_layers,
        'use_lora': model.use_lora,
        'lora_rank': model.lora_rank,
        'quant_bits': model.quant_bits
    }
    
    torch.save(checkpoint, "model.pth")
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        dynamic_path = f"model_{current_mode}_{timestamp}.pth"
        torch.save(checkpoint, dynamic_path)
        print(f"Checkpoints saved: model.pth and {dynamic_path}")
    else:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

def load_checkpoint(path="model.pth"):
    global current_mode, optimizer, scheduler, model
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return False
    
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    
    # Handle both full checkpoints and raw state dicts
    if 'model_state' in checkpoint:
        model_state = checkpoint['model_state']
        saved_mode = checkpoint.get('mode', 'neural')
        saved_layers = checkpoint.get('num_layers', 4)
        saved_lora = checkpoint.get('use_lora')
        saved_rank = checkpoint.get('lora_rank', 4)
        saved_quant = checkpoint.get('quant_bits', None)
    else:
        # Assume it's a raw state dict
        model_state = checkpoint
        saved_mode = model.mode
        saved_layers = model.num_layers
        saved_lora = model.use_lora
        saved_rank = model.lora_rank
        saved_quant = model.quant_bits
    
    # If LoRA was active, we must re-filter the optimizer before loading its state
    if saved_lora:
        for name, param in model.named_parameters():
            if "lora_" in name or "coupling" in name or "norm_scale" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

    if 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    if 'scheduler_state' in checkpoint:
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
    print(f"Long training | Max epochs: {EPOCHS} | Steps per epoch: {STEPS_PER_EPOCH}")
    best_val_ppl = float('inf')

    # If LoRA is enabled, freeze base weights and only train adapters
    # Note: We still keep 'coupling' and 'norm_scale' trainable by default 
    # as they are small and central to the photonic mechanism.
    if any(p.requires_grad and "lora_" in n for n, p in model.named_parameters()):
        print("LoRA training detected: Freezing non-LoRA parameters (except couplings/norms)...")
        for name, param in model.named_parameters():
            if "lora_" in name or "coupling" in name or "norm_scale" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Re-initialize optimizer for trainable params only
        global optimizer, scheduler
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        # Re-initialize scheduler to point to the correct optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.05)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        steps = 0

        with tqdm(total=STEPS_PER_EPOCH, desc=f"Epoch {epoch}/{EPOCHS} Train") as pbar:
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
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            save_path = f"best_model_valppl_{val_ppl:.2f}.pt"
            # Use the unified save_checkpoint logic to include metadata
            save_checkpoint(path=save_path)
            print(f"  → New best model saved: {save_path}")

        save_checkpoint()

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Reinforcement Learning for Self-Improvement (RLVR)
# ────────────────────────────────────────────────────────────────

class RewardVerifier:
    """Objective verifier for sequence quality"""
    @staticmethod
    def get_reward(text, prompt):
        reward = 0.0
        # 1. Length reward: Encourage moderate length (e.g. 50-150 chars)
        length = len(text)
        if 50 < length < 300:
            reward += 1.0
        
        # 2. Non-repetition: Penalty for repetitive chunks
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            reward += unique_ratio * 2.0
            
        # 3. Task Success: Does it actually continue the prompt?
        # Simple heuristic: Does it contain new words not in the prompt?
        prompt_words = set(prompt.lower().split())
        new_words = set(words) - prompt_words
        if len(new_words) > 5:
            reward += 1.5
            
        # 4. Termination: Bonus for ending with punctuation
        if text.strip()[-1] in ".!?":
            reward += 1.0
            
        return reward

def rl_train(steps=100, samples_per_step=4):
    print(f"RLVR Fine-tuning started | Steps: {steps} | Sparse Updates")
    model.train()
    
    # Sparse Optimization: Freeze Embedding and Readout, only update Modulators
    for name, param in model.named_parameters():
        if "modulator" in name or "coupling" in name or "norm_scale" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    rl_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR * 0.5)
    
    for step in range(steps):
        # 1. Sample a random prompt from train text
        prompt_raw = get_random_chunk(train_text, length=64)
        prompt = tokenizer.decode(prompt_raw.tolist())
        
        batch_loss = 0.0
        total_reward = 0.0
        
        for _ in range(samples_per_step):
            # 2. Rollout: Generate with tracking gradients
            # We need a modified generate that tracks log_probs
            trajectory_tokens, log_probs = rollout(prompt, max_new=50)
            generated_text = tokenizer.decode(trajectory_tokens.tolist())
            
            # 3. Verify: Get reward
            reward = RewardVerifier.get_reward(generated_text, prompt)
            total_reward += reward
            
            # 4. Policy Gradient (REINFORCE)
            # Loss = - Reward * sum(log_probs)
            loss = -reward * sum(log_probs)
            batch_loss += loss / samples_per_step
            
        if samples_per_step > 0:
            rl_optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            rl_optimizer.step()
            
        if (step + 1) % 5 == 0:
            print(f"RL Step {step+1:3d} | Avg Reward: {total_reward/samples_per_step:.2f} | Loss: {batch_loss.item():.4f}")

    # Restore all parameters to trainable for standard SFT if needed later
    for param in model.parameters():
        param.requires_grad = True
    
    save_path = f"model_rl_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"RL Fine-tuning finished. Saved to {save_path}")

def rollout(prompt, max_new=50, temperature=0.9):
    """Generate a sequence while tracking log-probabilities for RL"""
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, disallowed_special=()), device=DEVICE)
    z_cache = []
    
    # Process prompt
    for i in range(len(prompt_tokens)):
        token_id = prompt_tokens[i:i+1]
        z_curr = model.token_embed(token_id).squeeze(0)
        z_curr = z_curr + model.pos_bias(i)
        for layer in model.layers:
            z_curr = layer(z_curr, z_cache, i, num_steps=M)
        z_cache.append(z_curr)
        logits = model.readout(z_curr)

    trajectory = []
    log_probs = []
    current_pos = len(prompt_tokens)
    
    for _ in range(max_new):
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        next_token = dist.sample()
        
        trajectory.append(next_token)
        log_probs.append(dist.log_prob(next_token))
        
        if next_token.item() == tokenizer.eot_token:
            break
            
        z_curr = model.token_embed(next_token.unsqueeze(0)).squeeze(0)
        z_curr = z_curr + model.pos_bias(current_pos)
        for layer in model.layers:
            z_curr = layer(z_curr, z_cache, current_pos, num_steps=M)
        z_cache.append(z_curr)
        logits = model.readout(z_curr)
        current_pos += 1
        
    return torch.stack(trajectory), torch.stack(log_probs)

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
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, disallowed_special=()), device=DEVICE)
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
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, disallowed_special=()), device=DEVICE)
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
    parser.add_argument("--rl_train", action="store_true", help="Start RLVR self-improvement training")
    parser.add_argument("--lora", action="store_true", help="Use Low-Rank Adaptation (LoRA) for fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--quant", action="store_true", help="Enable simulated 4-bit quantization")
    args = parser.parse_args()

    current_mode = args.mode
    EPOCHS = args.epochs
    STEPS_PER_EPOCH = args.steps
    
    # Initialize the model correctly before potentially loading
    quant_bits = 4 if args.quant else None
    init_model(mode=current_mode, num_layers=args.layers, use_lora=args.lora, lora_rank=args.lora_rank, quant_bits=quant_bits)

    if args.load:
        load_checkpoint(args.checkpoint)

    if args.train:
        train()
        run_final_tests()

    if args.rl_train:
        rl_train(steps=args.steps)
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