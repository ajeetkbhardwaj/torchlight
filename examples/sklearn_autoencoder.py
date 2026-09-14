"""
Class-conditioned autoencoder on sklearn's handwritten digits.

The encoder squeezes a 64-d image into an 8-d latent, the class label is
embedded into an 8-d conditioning vector (``nn.Embedding``), and the
decoder reconstructs from ``cat(latent, condition)``.  This shows how the
embedding + concatenation primitives compose into a real architecture.

Usage:
    PYTHONPATH=src python examples/sklearn_autoencoder.py
"""

import numpy as np
from sklearn.datasets import load_digits

import torchlight as tl
from torchlight import nn
from torchlight.nn import functional as F
from torchlight.optim import Adam


class CondAutoencoder(nn.Module):
    """64 -> 32 -> 8 latent; decoder cond on the class embedding -> 64."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 8),
        )
        # One 8-d embedding per digit class.
        self.labels = nn.Embedding(10, 8)
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid(),
        )

    def forward(self, x, y):
        z = self.encoder(x)             # (B, 8) latent
        c = self.labels(y.view(y.shape[0]))   # (B, 8) class condition
        h = tl.cat([z, c], dim=1)       # (B, 16)
        return self.decoder(h)


def main():
    # 1. Data ---------------------------------------------------------------
    digits = load_digits()
    X = (digits.data / 16.0).astype(np.float32)   # greyscale -> [0, 1]
    y = digits.target.astype(np.float32)
    n = int(0.8 * len(X))
    X_tr, y_tr, X_te, y_te = X[:n], y[:n], X[n:], y[n:]

    ds = tl.data.TensorDataset(tl.tensor(X_tr), tl.tensor(y_tr))
    loader = tl.data.DataLoader(ds, batch_size=64, shuffle=True, seed=0)
    print(f"train {n} | test {len(X) - n} | 8-d latent + 8-d class condition")

    # 2. Model + optimizer ----------------------------------------------------
    model = CondAutoencoder()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 3. Training loop ---------------------------------------------------------
    for epoch in range(15):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb, yb)
            loss = F.mse_loss(out, xb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.to_numpy().reshape(-1)[0]))

        recon = model(tl.tensor(X_te), tl.tensor(y_te)).to_numpy()
        err = float(np.mean((recon - X_te) ** 2))
        print(f"epoch {epoch + 1:2d}  ae loss {np.mean(losses):.5f}  "
              f"recon MSE {err:.5f}")

    # 4. Show one reconstruction ----------------------------------------------
    idx = 42
    before = X_te[idx].reshape(8, 8)
    after = recon[idx].reshape(8, 8)
    print(f"\nlabel {int(y_te[idx])}  =>  |before - after| = "
          f"{np.abs(before - after).mean():.4f}")


if __name__ == "__main__":
    main()