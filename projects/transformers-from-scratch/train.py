"""
A miniature GPT built from scratch, trained on real Shakespeare text.

Dataset: tiny Shakespeare (~1.1M characters of genuine public-domain plays),
downloaded once into ``./data``.  Character-level language model: every token
is a character, so the model learns spelling, vocabulary and simple rhythm.

Model: decoder-only transformer with 2 heads of block-wise causal
self-attention, LayerNorm + residual MLP blocks, and learned token/position
embeddings -- all assembled from torchlight primitives (``Linear``,
``Embedding``, ``LayerNorm``, ``softmax``, matmul).  No attention module
exists in torchlight; the multi-head attention is built inline.

After training a checkpoint is saved to ``./out`` (state dict ``.npz`` +
``meta.json``).  Run ``post.py`` afterwards for validation metrics, a learning
curve, and fresh sample generations.

Environment knob (optional): TORCHLIGHT_ITERS  (default: 1200)

In Colab:

    !git clone <your-repo-url> torchlight && cd torchlight && pip install -e .
    !python projects/transformers-from-scratch/train.py
    !python projects/transformers-from-scratch/post.py

Usage:
    PYTHONPATH=src python projects/transformers-from-scratch/train.py
"""

import json
import os
import sys
import urllib.request

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
# Hyper-parameters
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
BLOCK = 64        # context window (in characters)
N_EMBD = 64       # embedding width
N_HEAD = 4        # attention heads (head dim = N_EMBD / N_HEAD = 16)
N_LAYER = 2
BATCH = 64
ITERS = int(os.environ.get("TORCHLIGHT_ITERS", 1200))
LR = 3e-3
EVAL_EVERY = 300
SEED = 0
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_text() -> str:
    cache = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"[data] downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Model (from-scratch decoder-only transformer)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Scaled dot-product attention over a causal window.

    Q/K/V come from one ``Linear`` (split into heads), attention is
    ``softmax(Q K^T / sqrt(d_head) + mask) @ V``, exactly like GPT.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        # 1/sqrt(d_head) as a (1, 1) constant so it broadcasts over any batch.
        self.scale = tl.tensor(np.array([[self.head ** -0.5]], dtype=np.float32))

    def forward(self, x, mask):
        b, t, _ = x.shape
        q, k, v = tl.chunk(self.qkv(x), 3, dim=-1)  # each (B, T, n_embd)

        def heads(z):
            return (
                z.contiguous()
                .view(b, t, self.n_head, self.head)
                .permute(0, 2, 1, 3)               # (B, n_head, T, head)
            )

        q, k, v = heads(q), heads(k), heads(v)
        scores = q @ k.permute(0, 1, 3, 2) * self.scale  # (B, nH, T, T)
        attn = F.softmax(scores + mask, dim=-1)
        out = attn @ v                                   # (B, nH, T, head)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, self.n_head * self.head)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm transformer block: attn + residual, MLP + residual."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK, N_EMBD)
        # Register each block so Module traversal (and Adam) can see them.
        self.blocks = []
        for i in range(N_LAYER):
            blk = Block(N_EMBD, N_HEAD)
            self.add_module(f"blocks.{i}", blk)
            self.blocks.append(blk)
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, x):
        b, t = x.shape
        tok = self.token_emb(x)  # (B, T, n_embd)
        pos = self.pos_emb(tl.tensor(np.arange(t, dtype=np.float32)))  # (T, n_embd)
        h = tok + pos
        mask = tl.tensor(np.triu(np.full((t, t), -1e9), k=1).astype(np.float32))
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_f(h))  # (B, T, vocab)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def sample_batch(data, rng, batch, block):
    n = len(data) - block - 1
    ix = rng.randint(0, n, size=batch)
    x = np.stack([data[i : i + block] for i in ix]).astype(np.float32)
    y = np.stack([data[i + 1 : i + block + 1] for i in ix]).astype(np.float32)
    # shift targets left: position t predicts y[t+1]; last column is masked.
    y = np.roll(y, -1, axis=1)
    mask = np.ones_like(y)
    mask[:, -1] = 0.0
    return x, y, mask


def loss_at(model, x, y, m):
    logits = model(tl.tensor(x))
    per = F.softmax_loss(logits, tl.tensor(y), dim=-1)  # (B, T) per-position nll
    return float((per * tl.tensor(m)).sum().to_numpy().reshape(-1)[0] / np.sum(m))


def generate(model, chars, idx_map, seed=None, prompt="\n", steps=300, temperature=0.8):
    """Greedy-with-temperature sampling of fresh Shakespeare."""
    rng = np.random.RandomState(seed)
    ids = [idx_map[c] for c in prompt] if idx_map else []
    for _ in range(steps):
        cond = ids[-BLOCK:]
        logits = model(tl.tensor(np.asarray(cond, dtype=np.float32).reshape(1, -1)))
        z = logits.to_numpy()[0, -1, :].astype(np.float64) / temperature
        z = np.exp(z - z.max())
        p = z / z.sum()
        ids.append(int(rng.choice(len(chars), p=p)))
    return "".join(chars[i] for i in ids)


def prepare():
    """Load text, build the char vocab and split train/val positions.

    ``post.py`` imports this so the validation slice matches training exactly.
    """
    text = load_text()
    chars = sorted(set(text))
    v2i = {c: i for i, c in enumerate(chars)}
    data = np.asarray([v2i[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return {
        "text": text,
        "chars": chars,
        "v2i": v2i,
        "i2v": {i: c for c, i in v2i.items()},
        "data": data,
        "n_tr": n,
    }


def main():
    d = prepare()
    text, chars, v2i, data = d["text"], d["chars"], d["v2i"], d["data"]
    tr, va = data[: d["n_tr"]], data[d["n_tr"] :]
    print(
        f"[data] {len(data):,} chars | vocab {len(chars)} | "
        f"train {len(tr):,} chars"
    )

    model = GPT(len(chars))
    optimizer = Adam(model.parameters(), lr=LR)
    rng = np.random.RandomState(SEED)
    history = []

    for step in range(1, ITERS + 1):
        model.train()
        x, y, m = sample_batch(tr, rng, BATCH, BLOCK)
        optimizer.zero_grad()
        logits = model(tl.tensor(x))
        per = F.softmax_loss(logits, tl.tensor(y), dim=-1)
        loss = (per * tl.tensor(m)).sum() / tl.tensor(m).sum()
        loss.backward()
        optimizer.step()

        if step == 1 or step % EVAL_EVERY == 0:
            model.eval()
            vx, vy, vm = sample_batch(va, rng, BATCH, BLOCK)
            v = loss_at(model, vx, vy, vm)
            cur = float(loss.to_numpy().reshape(-1)[0])
            history.append({"step": step, "train": cur, "val": v, "ppl": float(np.exp(v))})
            print(
                f"step {step:5d}/{ITERS}  train {cur:.3f}  val {v:.3f}  "
                f"perplexity {np.exp(v):.1f}"
            )

    print("\n--- generated text (temperature 0.8) ---")
    print(generate(model, chars, v2i, seed=42, steps=400))

    # Save the checkpoint for post.py (inference + metrics).
    os.makedirs(OUT_DIR, exist_ok=True)
    save_state(model, os.path.join(OUT_DIR, "gpt_model.npz"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(
            {
                "chars": chars,
                "vocab_size": len(chars),
                "block": BLOCK,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
                "iters": ITERS,
            },
            fh,
        )
    with open(os.path.join(OUT_DIR, "history.json"), "w") as fh:
        json.dump(history, fh)
    print(f"[save] checkpoint -> {OUT_DIR}/gpt_model.npz + meta.json")


if __name__ == "__main__":
    main()