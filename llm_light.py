import torch
import torch.nn.functional as F
import numpy as np

# ────────────────────────────────────────────────────────────────
#  Hyperparameters
# ────────────────────────────────────────────────────────────────
N = 32              # sequence length
M = 6               # number of evolution steps
COUPLING = 0.10     # base coupling strength
EPS = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────
#  Encoding helper functions
# ────────────────────────────────────────────────────────────────
def text_to_complex(text: str, n: int = N) -> torch.Tensor:
    """Very simple char → complex encoding (for demo)"""
    text = (text + " " * n)[:n]  # pad/truncate
    indices = torch.tensor([ord(c) % 100 for c in text], dtype=torch.float32, device=device)
    amps = indices / 100.0
    phases = 2 * torch.pi * (indices / 100.0) - torch.pi
    return amps * torch.exp(1j * phases)

def complex_to_text(z: torch.Tensor) -> str:
    """Naive phase → char decoding"""
    phases = torch.angle(z)
    indices = ((phases + torch.pi) / (2 * torch.pi) * 100).round().long().cpu().tolist()
    chars = [chr(i % 128) for i in indices]
    return "".join(chars).rstrip()

# ────────────────────────────────────────────────────────────────
#  Core forward evolution (differentiable!)
# ────────────────────────────────────────────────────────────────
def evolve(u0: torch.Tensor, v0: torch.Tensor, return_all=False):
    u = [u0.clone()]
    v = [v0.clone()]

    for m in range(M):
        # Absorbing boundaries → pad with zeros
        u_left = torch.cat([torch.zeros(1, device=device), u[-1][:-1]])
        v_left = torch.cat([torch.zeros(1, device=device), v[-1][:-1]])
        u_right = torch.cat([u[-1][1:], torch.zeros(1, device=device)])
        v_right = torch.cat([v[-1][1:], torch.zeros(1, device=device)])

        # Per-neighbor phase/amplitude modulation
        z_left = u_left + 1j * v_left
        z_curr = u[-1] + 1j * v[-1]
        phase_diff_u = torch.angle(z_left) - torch.angle(z_curr)
        amp_ratio_u = torch.abs(z_left) / (torch.abs(z_curr) + EPS)
        mod_u = torch.tanh(phase_diff_u) * amp_ratio_u

        z_right = u_right + 1j * v_right
        phase_diff_v = torch.angle(z_right) - torch.angle(z_curr)
        amp_ratio_v = torch.abs(z_right) / (torch.abs(z_curr) + EPS)
        mod_v = torch.tanh(phase_diff_v) * amp_ratio_v

        # Interference terms
        interf_u = COUPLING * mod_u * z_left
        interf_v = COUPLING * mod_v * z_right

        # Residual + interference
        u_new = u[-1] + interf_u
        v_new = v[-1] + interf_v

        # Nonlinearity (tanh on real & imag parts)
        u_new = torch.tanh(u_new.real) + 1j * torch.tanh(u_new.imag)
        v_new = torch.tanh(v_new.real) + 1j * torch.tanh(v_new.imag)

        # Intensity normalization (max |z|^2 = 1)
        z = u_new + 1j * v_new
        max_int = torch.max(torch.abs(z)**2)
        if max_int > 0:
            scale = 1.0 / torch.sqrt(max_int)
            u_new = u_new * scale
            v_new = v_new * scale

        u.append(u_new)
        v.append(v_new)

    if return_all:
        return u, v
    return u[-1], v[-1]

# ────────────────────────────────────────────────────────────────
#  Reverse optimization demo
# ────────────────────────────────────────────────────────────────
def reverse_optimize(target_text: str, lr=0.05, steps=800):
    target_z = text_to_complex(target_text)

    # Learnable initial state (what we optimize)
    initial_z = text_to_complex(" " * N) + 0.1 * torch.randn(N, device=device, dtype=torch.complex64)
    initial_z.requires_grad_(True)

    optimizer = torch.optim.Adam([initial_z], lr=lr)

    losses = []

    for it in range(steps):
        optimizer.zero_grad()

        u0 = initial_z.real
        v0 = initial_z.imag

        u_final, v_final = evolve(u0, v0)
        pred_z = u_final + 1j * v_final

        # loss = F.mse_loss(pred_z, target_z)
        # Fix: MSE for complex numbers manually
        loss = torch.mean(torch.abs(pred_z - target_z)**2)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if it % 100 == 0:
            print(f"Step {it:4d} | Loss: {loss.item():.6f}")

    # Final results
    with torch.no_grad():
        u_final, v_final = evolve(initial_z.real, initial_z.imag)
        pred_final = u_final + 1j * v_final
        recovered_text = complex_to_text(pred_final)
        initial_text = complex_to_text(initial_z)

    return {
        "target": target_text,
        "recovered_after_evolution": recovered_text,
        "optimized_initial_text": initial_text,
        "final_loss": losses[-1],
        "loss_history": losses
    }

# ────────────────────────────────────────────────────────────────
#  Run demo
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    target = "hello world this is a test   "[:N]
    print("Target (desired after evolution):", repr(target))

    result = reverse_optimize(target, lr=0.08, steps=1200)

    print("\nResults:")
    print("  Target:                  ", repr(result["target"]))
    print("  Recovered after evolution:", repr(result["recovered_after_evolution"]))
    print("  Optimized initial text:  ", repr(result["optimized_initial_text"]))
    print(f"  Final MSE loss: {result['final_loss']:.6f}")