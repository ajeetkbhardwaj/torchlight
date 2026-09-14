# Tutorial 7: Real-World Datasets (scikit-learn)

Torchlight is not just for synthetic problems — it drops into the same loop
you'd use with scikit-learn data.  This tutorial walks through the three
ready-to-run examples in `examples/`, each pairing a real sklearn dataset
with a distinct deep-learning architecture.

> Requires `pip install scikit-learn`.

## 1. MLP on Breast Cancer (`examples/sklearn_classifier.py`)

A classic binary classification task: 30 numeric features, 2 classes.  The
model is a regularization-heavy MLP that uses nearly every training
technique torchlight ships:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torchlight as tl
from torchlight import nn
from torchlight.nn import functional as F
from torchlight.optim import Adam, CosineAnnealingLR
from torchlight.nn.utils import clip_grad_norm_

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data).astype(np.float32)
y = data.target.astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = nn.Sequential(
    nn.Linear(30, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 1),
)
optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

ds = tl.data.TensorDataset(tl.tensor(X_tr), tl.tensor(y_tr))
loader = tl.data.DataLoader(ds, batch_size=32, shuffle=True, seed=0)

for epoch in range(40):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
    scheduler.step()
    model.eval()
    # evaluate...
```

Highlights: **BatchNorm** for stable training, **Dropout** for
regularization, a **cosine LR schedule**, and **gradient clipping** to bound
update size — reaches ~97% test accuracy in under 10 seconds on a laptop.

## 2. CNN on handwritten digits (`examples/sklearn_digits_conv.py`)

`load_digits()` gives 8×8 greyscale images, so we treat each sample as a
`(1, 8, 8)` tensor (channel, height, width) and use a proper conv net:

```python
from torchlight import nn
from torchlight.nn import functional as F

class ConvDigits(nn.Module):
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
```

The `Conv2d` layer here is torchlight's own autograd convolution (see
`src/torchlight/autograd/convolutions.py`), so gradients flow straight
through convolution → pooling → dense.

## 3. Class-conditioned autoencoder (`examples/sklearn_autoencoder.py`)

An autoencoder learns to compress an image to an 8-d latent and rebuild it.
We condition the decoder on the digit's class using an **Embedding** — a
learnable lookup table — and concatenate latent + condition with `tl.cat`:

```python
from torchlight import nn

class CondAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 8),
        )
        self.labels = nn.Embedding(10, 8)   # one 8-d vector per digit class
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid(),
        )

    def forward(self, x, y):
        z = self.encoder(x)                  # (B, 8)
        c = self.labels(y.view(y.shape[0]))  # (B, 8)
        h = tl.cat([z, c], dim=1)            # (B, 16)
        return self.decoder(h)
```

Note that targets from the DataLoader have shape `(B, 1)`, so we flatten
before the embedding lookup.  Reconstructed images converge to a mean
absolute error of a few percent, which you can compare by eye.

## Which example should you read first?

| Goal                                   | Example                          |
| -------------------------------------- | -------------------------------- |
| Standard tabular ML + training tricks  | `sklearn_classifier.py`        |
| Convolutional networks on images       | `sklearn_digits_conv.py`       |
| Embeddings, latent spaces, generative  | `sklearn_autoencoder.py`       |