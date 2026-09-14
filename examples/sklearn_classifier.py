"""
Train an MLP on sklearn's Breast Cancer Wisconsin dataset with the full
deep-learning toolkit: BatchNorm + Dropout regularization, a cosine LR
schedule, and gradient clipping.

Usage:
    PYTHONPATH=src python examples/sklearn_classifier.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torchlight as tl
from torchlight import nn
from torchlight.nn import functional as F
from torchlight.nn.utils import clip_grad_norm_
from torchlight.optim import Adam, CosineAnnealingLR


class MLP(nn.Module):
    """30 -> 64 -> 32 -> 1 architecturally with BN + dropout."""

    def __init__(self, in_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # 1. Data ------------------------------------------------------------
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    ds = tl.data.TensorDataset(tl.tensor(X_tr), tl.tensor(y_tr))
    loader = tl.data.DataLoader(ds, batch_size=32, shuffle=True, seed=0)
    print(f"train {X_tr.shape[0]} | test {X_te.shape[0]} | {X.shape[1]} features")

    # 2. Model, optimizer, scheduler ---------------------------------------
    model = MLP(X.shape[1])
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    # 3. Training loop ------------------------------------------------------
    for epoch in range(40):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.to_numpy().reshape(-1)[0]))
        scheduler.step()

        model.eval()
        preds = (model(tl.tensor(X_te)).to_numpy() > 0).astype(int).ravel()
        acc = accuracy_score(y_te, preds)
        print(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.4f}  "
              f"lr {optimizer.lr:.2e}  test acc {acc:.2%}")


if __name__ == "__main__":
    main()