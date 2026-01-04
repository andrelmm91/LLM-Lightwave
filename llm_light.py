import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR



# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
MAX_N = 256
M = 8                # evolution steps per token
H = 4                # multi-head count
COUPLING = 0.12
EPS = 1e-8
VOCAB_SIZE = 32
READOUT_DIM = 32
BATCH_SIZE = 8
SEQ_LEN = 64         # training sequence length
EPOCHS = 100
LR = 0.008
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on device: {DEVICE}")

# ────────────────────────────────────────────────────────────────
#  Vocabulary
# ────────────────────────────────────────────────────────────────
vocab = list(" abcdefghijklmnopqrstuvwxyz012345")
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def text_to_indices(text: str, seq_len: int = SEQ_LEN) -> torch.Tensor:
    text = (text.lower() * (seq_len // len(text) + 2))[:seq_len]
    return torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long, device=DEVICE)

def indices_to_text(indices) -> str:
    return "".join(idx_to_char.get(int(i), "?") for i in indices).rstrip()

# ────────────────────────────────────────────────────────────────
#  Modules
# ────────────────────────────────────────────────────────────────
class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len=MAX_N):
        super().__init__()
        self.pe = torch.nn.Parameter(0.25 * torch.randn(max_len, dtype=torch.complex64, device=DEVICE))

    def forward(self, pos: torch.Tensor):
        return self.pe[pos % MAX_N]

class MultiHeadModulator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList([torch.nn.Linear(4, 2) for _ in range(H)])
        self.out_proj = torch.nn.Linear(H*2, 2)

    def forward(self, z_curr, z_past):
        if z_past.numel() == 0:
            return torch.zeros(2, device=DEVICE, dtype=torch.float32)
        
        # Prepare features: concatenate current and past (Real, Imag)
        # z_curr: [1, 2] (real, imag)
        # z_past: [k, 2] (real, imag)
        # Check inputs: if they are complex, split them. If they are already split (dim=-1=2), use as is.
        # The caller (see below) might pass complex tensor or split tensor?
        # User code passed: `modulator(candidate.unsqueeze(0), z_past)` where they are complex.
        # So we MUST split them here.
        
        z_curr_feat = torch.stack([z_curr.real, z_curr.imag], dim=-1) # [1, 2]
        z_past_feat = torch.stack([z_past.real, z_past.imag], dim=-1) # [k, 2]
        
        features = torch.cat([z_curr_feat.repeat(z_past.size(0), 1), z_past_feat], dim=-1) # [k, 4]
        
        head_outs = [torch.tanh(head(features)) for head in self.heads] # List of [k, 2]
        combined = torch.cat(head_outs, dim=-1) # [k, H*2]
        
        return self.out_proj(combined.mean(dim=0, keepdim=True)).squeeze()

class ReadoutHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, VOCAB_SIZE)

    def forward(self, z):
        features = torch.stack([z.real, z.imag], dim=-1)
        return self.proj(features)

# ────────────────────────────────────────────────────────────────
#  Core components
# ────────────────────────────────────────────────────────────────
pos_enc = LearnedPositionalEncoding().to(DEVICE)
modulator = MultiHeadModulator().to(DEVICE)
readout = ReadoutHead().to(DEVICE)

optimizer = optim.Adam(
    list(pos_enc.parameters()) + list(modulator.parameters()) + list(readout.parameters()),
    lr=LR, betas=(0.9, 0.98), weight_decay=1e-5
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.1)

# ────────────────────────────────────────────────────────────────
#  Incremental forward pass (teacher forcing)
# ────────────────────────────────────────────────────────────────
def forward_incremental(indices: torch.Tensor):
    """
    Teacher-forcing forward pass
    Returns logits for each position (except first)
    """
    seq_len = indices.size(0)
    z_cache = []
    logits_list = []

    for i in range(seq_len):
        # Target token at position i is used as teacher
        target_idx = indices[i]
        candidate = 0.4 * torch.randn((), device=DEVICE, dtype=torch.complex64)

        # Add positional encoding
        pe = pos_enc(torch.tensor(i, device=DEVICE))
        candidate = candidate + pe              # Out-of-place add to make autograd happy

        # Evolve
        if z_cache:
            z_past = torch.stack(z_cache)
            mod_factor_tuple = modulator(candidate.unsqueeze(0), z_past) # [2] (real, imag)
            mod_factor = torch.complex(mod_factor_tuple[0], mod_factor_tuple[1])
            interf = COUPLING * mod_factor * z_past.mean()
            z_new = candidate + interf
        else:
            z_new = candidate

        # Nonlinearity + norm
        z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)
        max_int = torch.max(torch.abs(z_new)**2)
        if max_int > 0:
            z_new = z_new * (1.0 / torch.sqrt(max_int))

        # Collect logits (predict NEXT token)
        if i < seq_len - 1:
            logits = readout(z_new)
            logits_list.append(logits)

        z_cache.append(z_new)

    return torch.stack(logits_list)  # [seq_len-1, VOCAB_SIZE]

# ────────────────────────────────────────────────────────────────
#  Training Loop
# ────────────────────────────────────────────────────────────────
def train():
    simple_text = (
        "once upon a time in a land far far away there lived a brave little mouse "
        "who loved to explore the big wide world and meet new friends under the shining sun "
        "and bright stars every night "
    )

    print("Starting training...")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

        # Create batch-like data (repetition + shift for teacher forcing)
        indices = text_to_indices(simple_text, SEQ_LEN * BATCH_SIZE)
        indices = indices.view(BATCH_SIZE, SEQ_LEN)

        for b in range(BATCH_SIZE):
            seq = indices[b]
            logits = forward_incremental(seq)          # [seq_len-1, VOCAB]
            target = seq[1:]                           # next tokens

            loss = F.cross_entropy(logits, target, reduction='mean')
            loss.backward(retain_graph=(b < BATCH_SIZE-1))  # accumulate gradients
            total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        avg_loss = total_loss / BATCH_SIZE
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training finished.")

# ────────────────────────────────────────────────────────────────
#  Generation after training (greedy)
# ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new=120, temperature=0.85):
    prompt_idx = torch.tensor([char_to_idx.get(c.lower(), 0) for c in prompt], device=DEVICE)
    z_cache = []
    generated = prompt_idx.clone()

    # Prefill
    for i in range(len(prompt_idx)):
        candidate = 0.4 * torch.randn((), device=DEVICE, dtype=torch.complex64)
        pe = pos_enc(torch.tensor(i, device=DEVICE))
        pe = pos_enc(torch.tensor(i, device=DEVICE))
        candidate = candidate + pe

        z_new = incremental_evolve_step(candidate, z_cache, i)  # reuse helper from previous
        z_cache.append(z_new)

    # Generate
    current_pos = len(prompt_idx)
    for _ in range(max_new):
        prev = z_cache[-1]
        candidate = prev + 0.18 * torch.randn_like(prev)
        pe = pos_enc(torch.tensor(current_pos, device=DEVICE))
        pe = pos_enc(torch.tensor(current_pos, device=DEVICE))
        candidate = candidate + pe

        z_new = incremental_evolve_step(candidate, z_cache, current_pos)

        logits = readout(z_new) / temperature
        next_token = F.softmax(logits, dim=-1).argmax()

        generated = torch.cat([generated, next_token.unsqueeze(0)])
        z_cache.append(z_new)
        current_pos += 1

        if next_token.item() == char_to_idx[' ']:
            break

    return indices_to_text(generated)

# Helper (from previous versions - for generation)
def incremental_evolve_step(candidate, z_cache, pos):
    pe = pos_enc(torch.tensor(pos, device=DEVICE))
    pe = pos_enc(torch.tensor(pos, device=DEVICE))
    candidate = candidate + pe

    if not z_cache:
        z_new = torch.tanh(candidate.real) + 1j * torch.tanh(candidate.imag)
    else:
        z_past = torch.stack(z_cache)
        mod_factor_tuple = modulator(candidate.unsqueeze(0), z_past)
        mod_factor = torch.complex(mod_factor_tuple[0], mod_factor_tuple[1])
        interf = COUPLING * mod_factor * z_past.mean()
        z_new = candidate + interf
        z_new = torch.tanh(z_new.real) + 1j * torch.tanh(z_new.imag)

    max_int = torch.max(torch.abs(z_new)**2)
    if max_int > 0:
        z_new *= 1.0 / torch.sqrt(max_int)

    return z_new

# ────────────────────────────────────────────────────────────────
#  Run everything
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()

    print("\n" + "="*70)
    prompt = "once upon a time"
    print(f"Prompt: {repr(prompt)}")
    generated = generate(prompt)
    print(f"Generated: {repr(generated)}")