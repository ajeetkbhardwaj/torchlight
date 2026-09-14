"""
Post-training evaluation for the (EN->FR) translation model.

Loads the checkpoint saved by ``train.py`` from ``./out``, reproduces the
exact validation split via ``train.prepare()``, then:

  * greedy-decode a sample of validation sentences,
  * report exact-match rate, a sentence-level BLEU (bigrams + brevity),
  * print a handful of en/fr/prediction examples,
  * write a summary PNG (training curve + per-sentence BLEU histogram).

Usage:
    python projects/machine-translation/post.py       (after train.py)
"""

import json
import math
import os
import sys
from collections import Counter

import numpy as np

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.abspath(os.path.join(_ROOT, "..", "..", "src"))
if os.path.isdir(os.path.join(_SRC, "torchlight")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torchlight.utils.serialization import load_state

from train import Seq2Seq, prepare, translate

OUT = os.path.join(_ROOT, "out")
N_EVAL = 100   # validation sentences to decode for metrics


def bleu(ref: str, hyp: str, n=2) -> float:
    """Sentence BLEU up to n-grams with a brevity penalty (approx.)."""
    rt, ht = ref.split(), hyp.split()
    if not rt or not ht:
        return 0.0
    prec, tot = 0.0, 0
    for k in range(1, n + 1):
        if len(rt) < k or len(ht) < k:
            continue
        rng = Counter(zip(*[rt[i:] for i in range(k)]))
        hng = Counter(zip(*[ht[i:] for i in range(k)]))
        prec += sum(min(rng[g], c) for g, c in hng.items())
        tot += sum(hng.values())
    prec /= max(tot, 1)
    bp = math.exp(min(0.0, 1.0 - len(rt) / len(ht)))
    return bp * prec


def main():
    with open(os.path.join(OUT, "meta.json")) as fh:
        meta = json.load(fh)
    src_v, tgt_v = meta["src_vocab"], meta["tgt_vocab"]
    rev_tgt = {v: k for k, v in tgt_v.items()}
    model = Seq2Seq(len(src_v) + 4, len(tgt_v) + 4)
    load_state(model, os.path.join(OUT, "mt_model.npz"))
    print(f"[load] model from out/mt_model.npz | src vocab {len(src_v)} | tgt vocab {len(tgt_v)}")

    d = prepare(max_pairs=meta.get("max_pairs"))
    pairs, src_ids = d["pairs"], d["src_ids"]
    val_idx = list(range(d["n_tr"], len(pairs)))
    print(f"[eval] greedy-decoding {min(N_EVAL, len(val_idx))} of {len(val_idx)} validation pairs")

    model.eval()
    exact, bles, oov = 0, [], 0
    for vi in val_idx[:N_EVAL]:
        en, fr = pairs[vi]
        hyp = translate(model, src_ids[vi], rev_tgt)
        exact += int(hyp == fr)
        bles.append(bleu(fr, hyp))
        if any(w == "?" for w in hyp.split()):
            oov += 1

    print(f"exact-match : {exact}/{min(N_EVAL, len(val_idx))} = {exact / min(N_EVAL, len(val_idx)):.1%}")
    print(f"BLEU (mean): {np.mean(bles):.3f}   median {np.median(bles):.3f}   max {np.max(bles):.3f}")
    print(f"sentences with unmapped tokens: {oov}")

    print("\nexamples:")
    for vi in val_idx[:10]:
        print(f"  en: {pairs[vi][0]}")
        print(f"  fr: {pairs[vi][1]}")
        print(f"  -> {translate(model, src_ids[vi], rev_tgt)}")

    # Optional plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[plot] matplotlib not available; skipping PNG")
        return

    hist = json.load(open(os.path.join(OUT, "history.json"))) if os.path.exists(os.path.join(OUT, "history.json")) else []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if hist:
        axes[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], "-o")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].set_title("train loss")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "no history", ha="center")
    axes[1].hist(bles, bins=20, edgecolor="k", alpha=0.7)
    axes[1].axvline(np.mean(bles), color="r", ls="--", label=f"mean {np.mean(bles):.2f}")
    axes[1].set_xlabel("sentence BLEU"); axes[1].set_ylabel("sentences"); axes[1].set_title("BLEU distribution")
    axes[1].legend()
    fig.tight_layout()
    out_png = os.path.join(OUT, "mt_eval.png")
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved {out_png}")


if __name__ == "__main__":
    main()