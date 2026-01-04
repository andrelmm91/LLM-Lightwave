import torch
import torch.nn.functional as F
import numpy as np

# ────────────────────────────────────────────────────────────────
#  Hyperparameters - scaled up
# ────────────────────────────────────────────────────────────────
N = 128             # sequence length (longer context demo)
M = 12              # more evolution steps (deeper "layers")
COUPLING = 0.10
EPS = 1e-8
VOCAB_SIZE = 32     # tiny vocab for demo
READOUT_DIM = 32    # readout projects to vocab logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ────────────────────────────────────────────────────────────────
#  Tiny artificial vocabulary & encoding
# ────────────────────────────────────────────────────────────────
vocab = list(" abcdefghijklmnopqrstuvwxyz012345")  # 0-25 letters + space + digits + specials
char_to_idx = {c: i for i, c in enumerate(vocab)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def text_to_indices(text: str, n: int = N) -> torch.Tensor:
    text = (text.lower() + " " * n)[:n]
    return torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long, device=device)

def indices_to_text(indices: torch.Tensor) -> str:
    return "".join(idx_to_char.get(i.item(), "?") for i in indices).rstrip()

# ────────────────────────────────────────────────────────────────
#  Readout head (learnable)
# ────────────────────────────────────────────────────────────────
class ReadoutHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, VOCAB_SIZE, bias=True)  # real+imag → logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N] complex → stack real/imag → [N, 2]
        features = torch.stack([z.real, z.imag], dim=-1)    # [N, 2]
        logits = self.proj(features)                        # [N, VOCAB_SIZE]
        return logits

readout = ReadoutHead().to(device)

# ────────────────────────────────────────────────────────────────
#  Evolution (same as before, now with larger N/M)
# ────────────────────────────────────────────────────────────────
def evolve(u0: torch.Tensor, v0: torch.Tensor):
    u = [u0.clone()]
    v = [v0.clone()]

    for m in range(M):
        u_left  = torch.cat([torch.zeros(1, device=device), u[-1][:-1]])
        v_left  = torch.cat([torch.zeros(1, device=device), v[-1][:-1]])
        u_right = torch.cat([u[-1][1:], torch.zeros(1, device=device)])
        v_right = torch.cat([v[-1][1:], torch.zeros(1, device=device)])

        z_curr = u[-1] + 1j * v[-1]
        z_left  = u_left + 1j * v_left
        z_right = u_right + 1j * v_right

        # Modulation
        phase_diff_u = torch.angle(z_left) - torch.angle(z_curr)
        amp_ratio_u  = torch.abs(z_left) / (torch.abs(z_curr) + EPS)
        mod_u = torch.tanh(phase_diff_u) * amp_ratio_u

        phase_diff_v = torch.angle(z_right) - torch.angle(z_curr)
        amp_ratio_v  = torch.abs(z_right) / (torch.abs(z_curr) + EPS)
        mod_v = torch.tanh(phase_diff_v) * amp_ratio_v

        interf_u = COUPLING * mod_u * z_left
        interf_v = COUPLING * mod_v * z_right

        u_new = u[-1] + interf_u
        v_new = v[-1] + interf_v

        # Nonlinearity
        u_new = torch.tanh(u_new.real) + 1j * torch.tanh(u_new.imag)
        v_new = torch.tanh(v_new.real) + 1j * torch.tanh(v_new.imag)

        # Normalize max intensity = 1
        z = u_new + 1j * v_new
        max_int = torch.max(torch.abs(z)**2)
        if max_int > 0:
            scale = 1.0 / torch.sqrt(max_int)
            u_new *= scale
            v_new *= scale

        u.append(u_new)
        v.append(v_new)

    return u[-1] + 1j * v[-1]

# ────────────────────────────────────────────────────────────────
#  Reverse optimization with readout head
# ────────────────────────────────────────────────────────────────
def reverse_optimize_with_readout(target_text: str, lr=0.08, steps=1800):
    target_indices = text_to_indices(target_text)
    target_onehot = F.one_hot(target_indices, num_classes=VOCAB_SIZE).float()

    # Learnable initial complex embedding
    initial_z = 0.3 * torch.randn(N, device=device, dtype=torch.complex64)
    initial_z.requires_grad_(True)

    optimizer = torch.optim.Adam([initial_z, *readout.parameters()], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.4)

    losses = []

    for it in range(steps):
        optimizer.zero_grad()

        z_final = evolve(initial_z.real, initial_z.imag)
        logits = readout(z_final)               # [N, VOCAB_SIZE]

        loss = F.cross_entropy(logits, target_indices, reduction='mean')
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if it % 200 == 0:
            print(f"Step {it:4d} | CE loss: {loss.item():.4f} | lr: {optimizer.param_groups[0]['lr']:.5f}")

    # Results
    with torch.no_grad():
        z_final = evolve(initial_z.real, initial_z.imag)
        logits = readout(z_final)
        pred_probs = F.softmax(logits / 0.8, dim=-1)   # mild temperature
        pred_indices = pred_probs.argmax(dim=-1)
        recovered_text = indices_to_text(pred_indices)

        initial_indices = ((torch.angle(initial_z) + torch.pi) / (2 * torch.pi) * VOCAB_SIZE).round().long().clamp(0, VOCAB_SIZE-1)
        initial_text_guess = indices_to_text(initial_indices)

    return {
        "target_text": target_text,
        "recovered_text": recovered_text,
        "initial_text_guess": initial_text_guess,
        "final_loss": losses[-1],
        "loss_trend": losses[::100]  # every 100 steps
    }

# ────────────────────────────────────────────────────────────────
#  Demo run
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    target = "the quick brown fox jumps over the lazy dog  "[:N]
    print("Target (next-token shifted sequence):")
    print("  " + repr(target))

    result = reverse_optimize_with_readout(target)

    print("\nFinal Results after optimization:")
    print(f"  Target:     {repr(result['target_text'])}")
    print(f"  Recovered:  {repr(result['recovered_text'])}")
    print(f"  Initial guess: {repr(result['initial_text_guess'])}")
    print(f"  Final CE loss: {result['final_loss']:.4f}")
    print("  Loss trend (every 100 steps):", [f"{l:.4f}" for l in result['loss_trend']])