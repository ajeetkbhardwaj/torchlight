"""
Sentiment classifier on real user reviews.

Dataset: UCI "Sentiment Labelled Sentences" -- 3000 real, human-labelled
IMDb / Amazon / Yelp reviews (one labelled sentence each).  Downloaded once
into ``./data`` when missing.

Model: Embedding -> masked-mean pooling -> MLP head (binary logits).  Built
entirely from torchlight primitives; nothing beyond numpy + scikit-learn.

After training a checkpoint is saved to ``./out`` (state dict ``.npz`` +
``meta.json``).  Run ``post.py`` afterwards for metrics, a confusion matrix,
a plot and sample predictions.

Environment knob (optional): TORCHLIGHT_EPOCHS  (default: 8)

In Colab (GPU runtime or not):

    !git clone <your-repo-url> torchlight && cd torchlight && pip install -e .
    !python projects/sentiment-classifier/train.py
    !python projects/sentiment-classifier/post.py

    (optionally) import os; os.environ["TORCHLIGHT_DEVICE"] = "cuda"

Usage:
    PYTHONPATH=src python projects/sentiment-classifier/train.py
"""

import json
import os
import re
import sys
import urllib.request
import zipfile

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(os.path.join(_SRC, "torchlight")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torchlight as tl
from torchlight import nn
from torchlight.nn import functional as F
from torchlight.optim import Adam
from torchlight.utils.serialization import save_state

# ---------------------------------------------------------------------------
# Data: UCI "Sentiment Labelled Sentences"
# ---------------------------------------------------------------------------
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
)
MAX_VOCAB = 10_000   # keep only the most frequent words
MAX_LEN = 64         # truncate / pad every review to this many tokens
SEED = 0
EPOCHS = int(os.environ.get("TORCHLIGHT_EPOCHS", 8))
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def download_dataset() -> list[tuple[str, int]]:
    """Return ``[(text, label)]`` for the three labelled-review files."""
    cache = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache, exist_ok=True)
    zip_path = os.path.join(cache, "sentiment_labelled_sentences.zip")
    if not os.path.exists(zip_path):
        print(f"[data] downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    pairs: list[tuple[str, int]] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if "__MACOSX" in name or name.startswith("._"):
                continue
            if name.endswith("_labelled.txt"):
                for line in z.read(name).decode("utf-8", "replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit("\t", 1)
                    if len(parts) == 2:
                        pairs.append((parts[0].strip(), int(parts[1])))
    return pairs


def build_vocab(pairs: list[tuple[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text, _ in pairs:
        for w in tokenize(text):
            counts[w] = counts.get(w, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_VOCAB]
    return {w: i + 2 for i, (w, _) in enumerate(top)}  # 0=PAD, 1=UNK


def encode(pairs, vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return token ids (N, MAX_LEN), masks (N, MAX_LEN), labels (N,)."""
    N = len(pairs)
    ids = np.zeros((N, MAX_LEN), dtype=np.float32)
    masks = np.zeros((N, MAX_LEN), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.float32)
    for i, (text, label) in enumerate(pairs):
        labels[i] = label
        toks = [vocab.get(w, 1) for w in tokenize(text)][:MAX_LEN]
        ids[i, : len(toks)] = toks
        masks[i, : len(toks)] = 1.0
    return ids, masks, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SentimentNet(nn.Module):
    """Embedding -> masked-mean pooling -> MLP head.

    Mean pooling over the *real* tokens (the mask zeroes the padding), so
    review length is handled naturally with a single forward pass per batch.
    """

    def __init__(self, vocab_size, emb=128, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(emb, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        pooled = F.masked_mean(self.embed(x), mask, dim=1)  # (B, emb)
        return self.net(pooled)  # (B, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def prepare():
    """Download, shuffle, split (fixed seed), build vocab and encode.

    ``post.py`` imports and calls this same function, so the evaluation set is
    guaranteed to match the one used during training.
    """
    pairs = download_dataset()
    rng = np.random.RandomState(SEED)
    rng.shuffle(pairs)
    n_tr = int(0.8 * len(pairs))
    vocab = build_vocab(pairs[:n_tr])
    tr_ids, tr_mask, tr_y = encode(pairs[:n_tr], vocab)
    te_ids, te_mask, te_y = encode(pairs[n_tr:], vocab)
    return {
        "pairs": pairs,
        "n_tr": n_tr,
        "vocab": vocab,
        "tr_ids": tr_ids,
        "tr_mask": tr_mask,
        "tr_y": tr_y,
        "te_ids": te_ids,
        "te_mask": te_mask,
        "te_y": te_y,
    }


def main():
    d = prepare()
    pairs, vocab = d["pairs"], d["vocab"]
    tr_ids, tr_mask, tr_y = d["tr_ids"], d["tr_mask"], d["tr_y"]
    te_ids, te_mask, te_y = d["te_ids"], d["te_mask"], d["te_y"]
    rng = np.random.RandomState(SEED)
    print(
        f"[data] {len(pairs)} reviews | vocab {len(vocab)} | "
        f"train {d['n_tr']} | test {len(pairs) - d['n_tr']}"
    )

    model = SentimentNet(len(vocab) + 2)
    optimizer = Adam(model.parameters(), lr=1e-3)
    batch_size = 64
    history = []

    def batches(ids, mask, y, shuffle=True):
        idx = rng.permutation(len(y)) if shuffle else np.arange(len(y))
        for i in range(0, len(y), batch_size):
            sel = idx[i : i + batch_size]
            yield (
                tl.tensor(ids[sel]),
                tl.tensor(mask[sel]),
                tl.tensor(y[sel, None]),
            )

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, mb, yb in batches(tr_ids, tr_mask, tr_y):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb, mb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.to_numpy().reshape(-1)[0]))

        model.eval()
        logits = model(tl.tensor(te_ids), tl.tensor(te_mask)).to_numpy()
        acc = float(((logits.reshape(-1) > 0).astype(int) == te_y.astype(int)).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "test_acc": acc})
        print(f"epoch {epoch + 1:2d}  loss {np.mean(losses):.4f}  test acc {acc:.2%}")

    # A few sanity check spellings.
    model.eval()
    for text in [
        "this restaurant was absolutely amazing, best meal i have had in years",
        "worst customer service i have ever experienced, do not buy anything here",
        "the battery life is fine but the screen keeps flickering",
    ]:
        toks = np.zeros((1, MAX_LEN), dtype=np.float32)
        m = np.zeros((1, MAX_LEN), dtype=np.float32)
        ws = tokenize(text)[:MAX_LEN]
        toks[0, : len(ws)] = [vocab.get(w, 1) for w in ws]
        m[0, : len(ws)] = 1.0
        p = float(model(tl.tensor(toks), tl.tensor(m)).to_numpy().reshape(-1)[0])
        print(f"  {p:+.2f}  {text}")

    # Save the checkpoint for post.py (inference + metrics).
    os.makedirs(OUT_DIR, exist_ok=True)
    save_state(model, os.path.join(OUT_DIR, "sentiment_model.npz"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(
            {"vocab": vocab, "max_len": MAX_LEN, "vocab_size": len(vocab) + 2, "epochs": EPOCHS},
            fh,
        )
    with open(os.path.join(OUT_DIR, "history.json"), "w") as fh:
        json.dump(history, fh)
    print(f"[save] checkpoint -> {OUT_DIR}/sentiment_model.npz + meta.json")


if __name__ == "__main__":
    main()