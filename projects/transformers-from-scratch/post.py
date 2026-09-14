"""
Post-training evaluation for the tiny GPT.

Loads the checkpoint saved by ``train.py`` from ``./out``, reproduces the
validation slice via ``train.prepare()``, then:

  * reports validation loss, perplexity and top-1 character accuracy,
  * shows the recorded training curve,
  * generates fresh Shakespeare at a couple of temperatures,
  * writes a summary PNG (loss curve) when matplotlib is available.

Usage:
    python projects/transformers-from-scratch/post.py  (after train.py)
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.abspath(os.path.join(_ROOT, "..", "..", "src"))
if os.path.isdir(os.path.join(_SRC, "torchlight")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torchlight as tl
from torchlight.utils.serialization import load_state

from train import GPT, BATCH, BLOCK, N_EMBD, N_HEAD, N_LAYER, generate, prepare, sample_batch

OUT = os.path.join(_ROOT, "out")


def loss_and_acc(model, data, rng, batches=20):
    """Mean masked cross-entropy loss, perplexity and top-1 accuracy."""
    total_loss, total_tok, total_hit = 0.0, 0, 0
    for _ in range(batches):
        x, y, m = sample_batch(data, rng, BATCH, BLOCK)
        logits = model(tl.tensor(x)).to_numpy()  # (B, T, vocab)
        pred = logits.argmax(-1)
        exps = np.exp(logits - logits.max(-1, keepdims=True))
        probs = exps / exps.sum(-1, keepdims=True)
        nll = -np.log(np.clip(probs[np.arange(probs.shape[0])[:, None], np.arange(probs.shape[1])[None, :], y.astype(int)], 1e-9, 1.0))
        total_loss += float((nll * m).sum())
        total_hit += int(((pred == y) * m).sum())
        total_tok += int(m.sum())
    loss = total_loss / total_tok
    return loss, float(np.exp(loss)), total_hit / total_tok


def main():
    with open(os.path.join(OUT, "meta.json")) as fh:
        meta = json.load(fh)
    chars = meta["chars"]
    model = GPT(meta["vocab_size"])
    load_state(model, os.path.join(OUT, "gpt_model.npz"))
    print(f"[load] model from out/gpt_model.npz | vocab {meta['vocab_size']} chars | "
          f"{meta['n_layer']} layers x {meta['n_head']} heads x {meta['n_embd']} embd")

    d = prepare()
    v2i = d["v2i"]
    va = d["data"][d["n_tr"] :]
    print(f"[eval] {len(va):,} held-out characters")

    model.eval()
    loss, ppl, acc = loss_and_acc(model, va, np.random.RandomState(0))
    print(f"val loss      : {loss:.4f}")
    print(f"perplexity    : {ppl:.1f}   (random guess would be ~{len(chars):.0f})")
    print(f"top-1 acc     : {acc:.1%}")

    print("\n--- generated at temperature 0.8 ---")
    print(generate(model, chars, v2i, seed=42, steps=400))
    print("--- generated at temperature 1.2 ---")
    print(generate(model, chars, v2i, seed=7, steps=400))

    # Optional plot: train/val loss history.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[plot] matplotlib not available; skipping PNG")
        return

    hist = json.load(open(os.path.join(OUT, "history.json")))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([h["step"] for h in hist], [h["train"] for h in hist], "-o", label="train")
    ax.plot([h["step"] for h in hist], [h["val"] for h in hist], "-o", label="validation")
    ax.set_xlabel("iteration"); ax.set_ylabel("cross-entropy"); ax.set_title("tiny GPT loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT, "gpt_eval.png")
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved {out_png}")


if __name__ == "__main__":
    main()