"""
Classify sklearn's handwritten-digit (8x8) images with a convolutional
network built only from torchlight's autograd Convolutions.

Architecture: Conv2d(1->8) -> ReLU -> MaxPool2d(2)
              Conv2d(8->16) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(10)

Usage:
    PYTHONPATH=src python examples/sklearn_digits_conv.py
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torchlight as tl
from torchlight import nn
from torchlight.nn import functional as F
from torchlight.optim import Adam


class ConvDigits(nn.Module):
    """Small CNN for 8x8 greyscale digit images."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # (B, 8, 4, 4)
        x = self.pool2(F.relu(self.conv2(x)))   # (B, 16, 2, 2)
        x = x.view(x.shape[0], 16 * 2 * 2)
        return self.fc(x)


def main():
    # 1. Data -------------------------------------------------------------
    digits = load_digits()
    X = digits.data.reshape(-1, 1, 8, 8).astype(np.float32) / 16.0
    y = digits.target.astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    ds = tl.data.TensorDataset(tl.tensor(X_tr), tl.tensor(y_tr))
    loader = tl.data.DataLoader(ds, batch_size=64, shuffle=True, seed=0)
    print(f"{len(digits.images)} digits of 8x8 px | "
          f"train {X_tr.shape[0]} | test {X_te.shape[0]}")

    # 2. Model + optimizer --------------------------------------------------
    model = ConvDigits()
    optimizer = Adam(model.parameters(), lr=3e-3)

    # 3. Training loop -------------------------------------------------------
    for epoch in range(8):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.to_numpy().reshape(-1)[0]))

        model.eval()
        logits = model(tl.tensor(X_te)).to_numpy()
        acc = accuracy_score(y_te, logits.argmax(1))
        print(f"epoch {epoch + 1}  loss {np.mean(losses):.4f}  test acc {acc:.2%}")


if __name__ == "__main__":
    main()