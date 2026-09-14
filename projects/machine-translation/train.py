"""
Machine translation (English -> French) on a real parallel corpus.

Dataset: the classic Tatoeba English-French pairs shipped with the PyTorch
seq2seq tutorial (~136k real sentence pairs, "eng-fra.txt"), downloaded once
into ``./data`` and filtered to short sentences.

Model: a from-scratch GRU encoder-decoder with attention and teacher forcing
-- no RNN module exists in torchlight, so the GRU cell is built out of
``Linear`` + activations and the attention out of the raw matmul / softmax
primitives.

After training a checkpoint is saved to ``./out`` (state dict ``.npz`` +
``meta.json``).  Run ``post.py`` afterwards for metrics, sample translations
and plots.

Environment knobs (optional):
    TORCHLIGHT_MT_PAIRS   cap the number of training pairs (default: all)
    TORCHLIGHT_EPOCHS     number of epochs (default: 3)

In Colab:

    !git clone <your-repo-url> torchlight && cd torchlight && pip install -e .
    !python projects/machine-translation/train.py
    !python projects/machine-translation/post.py

Usage:
    PYTHONPATH=src python projects/machine-translation/train.py
"""

import json
import os
import random
import sys
import unicodedata
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
# Hyper-parameters
# ---------------------------------------------------------------------------
EMB = 128          # embedding / hidden width
HIDDEN = 128
SRC_MAX = 8        # keep pairs where both sides have <= these many words
TGT_MAX = 8
MIN_FREQ = 2
MAX_VOCAB = 20_000
BATCH = 64
EPOCHS = int(os.environ.get("TORCHLIGHT_EPOCHS", 3))
SEED = 0

PAD, SOS, EOS, UNK = 0, 1, 2, 3

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


# ---------------------------------------------------------------------------
# Data (Tatoeba eng-fra.txt via the PyTorch tutorial bundle)
# ---------------------------------------------------------------------------
DATA_URL = "https://download.pytorch.org/tutorial/data.zip"


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().strip()
    return " ".join(s.split())


def download_pairs() -> list[tuple[str, str]]:
    cache = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache, exist_ok=True)
    zip_path = os.path.join(cache, "pytorch_tutorial_data.zip")
    if not os.path.exists(zip_path):
        print(f"[data] downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    text = ""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read("data/eng-fra.txt").decode("utf-8", "replace")
    pairs = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        en, fr = _normalize(parts[0]), _normalize(parts[1])
        if not en or not fr:
            continue
        if len(en.split()) > SRC_MAX or len(fr.split()) > TGT_MAX:
            continue
        pairs.append((en, fr))
    return pairs


def build_vocab(sentences: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in sentences:
        for w in s.split():
            counts[w] = counts.get(w, 0) + 1
    keep = {w for w, c in counts.items() if c >= MIN_FREQ}
    ordered = sorted(keep, key=lambda w: (-counts[w], w))[: MAX_VOCAB - 4]
    return {w: i + 4 for i, w in enumerate(ordered)}  # 0..3 reserved


def encode(words: list[str], vocab: dict[str, int]) -> np.ndarray:
    return np.asarray([vocab.get(w, UNK) for w in words], dtype=np.float32)


def prepare(max_pairs: int | None = None):
    """Fetch/shuffle/split/tokenize once; post.py reuses the same function.

    The shuffle uses a fixed seed, so ``prepare()`` always reproduces the
    exact same train/val split (no data-setup divergence at eval time).
    """
    pairs = download_pairs()
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    if max_pairs is not None and max_pairs < len(pairs):
        pairs = pairs[:max_pairs]
    src_vocab = build_vocab([en for en, _ in pairs])
    tgt_vocab = build_vocab([fr for _, fr in pairs])
    src_ids = [encode(en.split(), src_vocab) for en, _ in pairs]
    tgt_ids = [encode(fr.split(), tgt_vocab) for _, fr in pairs]
    n_tr = int(0.9 * len(pairs))
    return {
        "pairs": pairs,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "n_tr": n_tr,
    }


def make_batch(src_ids, tgt_ids, indices):
    """Pad a batch to its max lengths.  Returns encoder input, decoder
    input (``[SOS, t1..tm]``), targets (``[t1..tm, EOS]``), the encoder pad
    penalty, and the valid-target mask."""
    si = [src_ids[i] for i in indices]
    ti = [tgt_ids[i] for i in indices]
    t_src = max(len(s) + 1 for s in si)          # enc input: words + EOS
    t_out = max(len(t) + 1 for t in ti)          # dec: SOS + m targets
    x = np.zeros((len(indices), t_src), dtype=np.float32)
    dec_in = np.zeros((len(indices), t_out), dtype=np.float32)
    tgt = np.zeros((len(indices), t_out), dtype=np.float32)
    mask = np.zeros((len(indices), t_out), dtype=np.float32)
    for k, (s, t) in enumerate(zip(si, ti)):
        x[k, : len(s) + 1] = np.append(s, EOS)
        dec_in[k, 0] = SOS
        dec_in[k, 1 : len(t) + 1] = t
        tgt[k, : len(t)] = t
        tgt[k, len(t)] = EOS
        mask[k, : len(t) + 1] = 1.0              # target positions incl. EOS
    enc_penalty = np.where(x == 0, -1e9, 0.0).astype(np.float32)
    return (
        tl.tensor(x),
        tl.tensor(dec_in),
        tl.tensor(tgt),
        tl.tensor(enc_penalty.reshape(len(indices), 1, t_src)),
        tl.tensor(mask),
    )


# ---------------------------------------------------------------------------
# Model: GRU encoder + attention decoder (all torchlight primitives)
# ---------------------------------------------------------------------------
class GRUCell(nn.Module):
    """One GRU step: ``h' = n + z * (h - n)`` with reset/update gates.

    Built only from `Linear` (``x @ W^T + b``) and elementwise activations,
    so gradients flow through the whole unrolled sequence via autograd.
    """

    def __init__(self, in_features, hidden):
        super().__init__()
        self.xr = nn.Linear(in_features, hidden)
        self.ur = nn.Linear(hidden, hidden)
        self.xz = nn.Linear(in_features, hidden)
        self.uz = nn.Linear(hidden, hidden)
        self.xn = nn.Linear(in_features, hidden)
        self.un = nn.Linear(hidden, hidden)

    def forward(self, x, h):
        r = F.sigmoid(self.xr(x) + self.ur(h))
        z = F.sigmoid(self.xz(x) + self.uz(h))
        n = F.tanh(self.xn(x) + self.un(r * h))
        return n + z * (h - n)


class Encoder(nn.Module):
    def __init__(self, src_vocab):
        super().__init__()
        self.embed = nn.Embedding(src_vocab, EMB)
        self.cell = GRUCell(EMB, HIDDEN)

    def forward(self, x, batch):
        """Run the GRU over ``x``'s padded time axis; return all states."""
        t_src = x.shape[1]
        h = tl.zeros((batch, HIDDEN))
        states = []
        emb = self.embed(x)  # (B, T, EMB)
        for t in range(t_src):
            h = self.cell(emb[:, t, :], h)
            states.append(h)
        return tl.stack(states, dim=1)  # (B, T, HIDDEN)


class AttentionDecoder(nn.Module):
    """GRU decoder; each step attends over the encoder states (Luong scores)."""

    def __init__(self, tgt_vocab):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab, EMB)
        self.cell = GRUCell(EMB, HIDDEN)
        self.fc_out = nn.Linear(2 * HIDDEN, tgt_vocab)

    def forward(self, dec_in, enc_states, enc_penalty, batch):
        """Teacher-forced decode.  Returns logits ``(B, T_out, V)``."""
        t_out = dec_in.shape[1]
        h = tl.zeros((batch, HIDDEN))
        logits = []
        emb = self.embed(dec_in)  # (B, T_out, EMB)
        enc = enc_states.transpose(1, 2)  # (B, HIDDEN, T_src)
        for t in range(t_out):
            h = self.cell(emb[:, t, :], h)
            # scores (B, 1, T_src); + penalty parks pad positions at -inf.
            scores = (h.unsqueeze(1) @ enc) + enc_penalty
            attn = F.softmax(scores, dim=-1)
            ctx = (attn.transpose(1, 2) * enc_states).sum(dim=1)  # (B, 1, H)
            ctx = ctx.reshape(ctx.shape[0], HIDDEN)
            logits.append(self.fc_out(F.tanh(tl.cat([h, ctx], dim=-1))))
        return tl.stack(logits, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.encoder = Encoder(src_vocab)
        self.decoder = AttentionDecoder(tgt_vocab)

    def forward(self, x, dec_in, tgt, enc_penalty, tgt_mask):
        enc = self.encoder(x, x.shape[0])
        logits = self.decoder(dec_in, enc, enc_penalty, x.shape[0])
        per = F.softmax_loss(logits, tgt, dim=-1)  # per-position nll (B, T_out)
        loss = (per * tgt_mask).sum() / tgt_mask.sum()
        return loss


def greedy_decode(model, src_ids, max_len=TGT_MAX + 1):
    """Translate one sentence greedily; returns a list of target token ids."""
    src = np.append(src_ids, EOS)
    x = tl.tensor(src.reshape(1, -1))
    enc_pen = tl.tensor(
        np.where(src == PAD, -1e9, 0.0).astype(np.float32).reshape(1, 1, -1)
    )
    enc = model.encoder(x, 1)          # (1, T_src, HIDDEN)
    encT = enc.transpose(1, 2)         # (1, HIDDEN, T_src)

    def step(h, emb_prev):
        """One decode step given the current hidden state + word/char."""
        h = model.decoder.cell(emb_prev[:, 0, :], h)
        scores = (h.unsqueeze(1) @ encT) + enc_pen
        attn = F.softmax(scores, dim=-1)
        ctx = (attn.transpose(1, 2) * enc).sum(dim=1).reshape(1, HIDDEN)
        logits = model.decoder.fc_out(F.tanh(tl.cat([h, ctx], dim=-1)))
        cur = int(logits.to_numpy().argmax(-1).reshape(-1)[0])
        return h, cur

    h = tl.zeros((1, HIDDEN))
    h, cur = step(
        h, model.decoder.embed(tl.tensor(np.array([[SOS]], dtype=np.float32)))
    )
    out = []
    for _ in range(max_len):
        if cur == EOS:
            break
        out.append(cur)
        h, cur = step(
            h, model.decoder.embed(tl.tensor(np.array([[cur]], dtype=np.float32)))
        )
    return out


def translate(model, src_ids, rev_tgt):
    """Greedy-decode ``src_ids`` into a French sentence string."""
    return " ".join(rev_tgt.get(i, "?") for i in greedy_decode(model, src_ids))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    max_pairs = int(os.environ["TORCHLIGHT_MT_PAIRS"]) if "TORCHLIGHT_MT_PAIRS" in os.environ else None
    data = prepare(max_pairs=max_pairs)
    pairs, src_v, tgt_v = data["pairs"], data["src_vocab"], data["tgt_vocab"]
    src_ids, tgt_ids, n_tr = data["src_ids"], data["tgt_ids"], data["n_tr"]
    rev_tgt = {v: k for k, v in tgt_v.items()}

    print(
        f"[data] {len(pairs)} pairs | train {n_tr} | val {len(pairs) - n_tr} | "
        f"src vocab {len(src_v)} | tgt vocab {len(tgt_v)}"
    )
    model = Seq2Seq(len(src_v) + 4, len(tgt_v) + 4)
    optimizer = Adam(model.parameters(), lr=1e-3)
    rng = random.Random(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)
    history = []
    val_indices = list(range(n_tr, len(pairs)))
    for epoch in range(EPOCHS):
        model.train()
        order = list(range(n_tr))
        rng.shuffle(order)
        losses = []
        for i in range(0, n_tr, BATCH):
            xb, decb, tgtb, pen, msk = make_batch(src_ids, tgt_ids, order[i : i + BATCH])
            optimizer.zero_grad()
            loss = model(xb, decb, tgtb, pen, msk)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.to_numpy().reshape(-1)[0]))
        avg = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": round(avg, 4)})
        print(f"epoch {epoch + 1}  loss {avg:.3f}", end="  ")

        model.eval()
        for vi in val_indices[:3]:
            en, fr = pairs[vi]
            print(f"\n    en: {en}\n    fr: {fr}\n    ->  {translate(model, src_ids[vi], rev_tgt)}")
        print()

    # Save the checkpoint for post.py (inference + metrics).
    save_state(model, os.path.join(OUT_DIR, "mt_model.npz"))
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(
            {
                "src_vocab": src_v,
                "tgt_vocab": tgt_v,
                "emb": EMB,
                "hidden": HIDDEN,
                "src_max": SRC_MAX,
                "tgt_max": TGT_MAX,
                "epochs": EPOCHS,
                "seed": SEED,
                "max_pairs": max_pairs,
            },
            fh,
        )
    with open(os.path.join(OUT_DIR, "history.json"), "w") as fh:
        json.dump(history, fh)
    print(f"[save] checkpoint -> {OUT_DIR}/mt_model.npz + meta.json")


if __name__ == "__main__":
    main()