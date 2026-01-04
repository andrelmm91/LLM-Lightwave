import torch
class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, dim=2):
        super().__init__()
        self.dim = dim # Need to store dim!
        self.embed = torch.nn.Embedding(vocab_size, dim * 2) 

    def forward(self, ids):
        dim = self.dim
        emb = self.embed(ids)
        # Verify shapes
        print(f"Emb shape: {emb.shape}, dim: {dim}")
        return emb[..., :dim] + 1j * emb[..., dim:]

te = TokenEmbedding(100, 2)
ids = torch.tensor([1, 2], dtype=torch.long)
try:
    out = te(ids)
    print("Success:", out.shape, out.dtype)
except Exception as e:
    import traceback
    traceback.print_exc()
