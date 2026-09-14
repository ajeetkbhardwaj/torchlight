"""
Post-training evaluation for the sentiment classifier.

Loads the checkpoint saved by ``train.py`` from ``./out``, re-evaluates the
held-out test split (identical to training thanks to ``prepare``), reports
classification metrics, prints sample correct/incorrect predictions and writes
a summary PNG (history curve + confusion matrix) when matplotlib is available.

Usage:
    python projects/sentiment-classifier/post.py       (after train.py)
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

from train import SentimentNet, prepare, tokenize

OUT = os.path.join(_ROOT, "out")


def load_model():
    with open(os.path.join(OUT, "meta.json")) as fh:
        meta = json.load(fh)
    model = SentimentNet(meta["vocab_size"])
    load_state(model, os.path.join(OUT, "sentiment_model.npz"))
    return model, meta


def metrics(y_true, y_pred):
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1, (tp, fp, fn, tn)


def main():
    model, meta = load_model()
    d = prepare()
    te_ids, te_mask, te_y, pairs = d["te_ids"], d["te_mask"], d["te_y"], d["pairs"]
    test_pairs = pairs[d["n_tr"] :]
    print(f"[load] model from out/sentiment_model.npz | vocab {meta['vocab_size']}")
    print(f"[eval] {len(te_y)} held-out reviews")

    model.eval()
    logits = model(tl.tensor(te_ids), tl.tensor(te_mask)).to_numpy().reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (logits > 0).astype(int)
    y_true = te_y.astype(int)

    acc, prec, rec, f1, (tp, fp, fn, tn) = metrics(y_true, y_pred)
    print(f"accuracy : {acc:.2%}")
    print(f"precision: {prec:.2%}   recall: {rec:.2%}   F1: {f1:.2%}")
    print(f"confusion  [[TN {tn:.0f}, FP {fp:.0f}], [FN {fn:.0f}, TP {tp:.0f}]]")
    print(f"positive rate in test set: {y_true.mean():.2%}")

    print("\nsample predictions:")
    order = np.argsort(np.abs(probs - 0.5))  # most confident examples first
    shown = 0
    for i in order:
        label, pred = "pos" if y_true[i] else "neg", "pos" if y_pred[i] else "neg"
        if shown < 5:
            print(f"  {probs[i]:.2f}  true={label} pred={pred}  {test_pairs[i][0]}")
            shown += 1
    print("  ...")
    shown = 0
    for i in order[::-1]:  # borderline / confusing examples
        label, pred = "pos" if y_true[i] else "neg", "pos" if y_pred[i] else "neg"
        if pred != label and shown < 5:
            print(f"  {probs[i]:.2f}  true={label} pred={pred}  {test_pairs[i][0]}")
            shown += 1

    # Optional plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[plot] matplotlib not available; skipping PNG")
        return

    hist = json.load(open(os.path.join(OUT, "history.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], "-o", label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.set_title("training loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("confusion matrix")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center", color="w" if cm[r, c] > cm.max() / 2 else "k", fontsize=14)
    fig.tight_layout()
    out_png = os.path.join(OUT, "sentiment_eval.png")
    fig.savefig(out_png, dpi=130)
    print(f"[plot] saved {out_png}")


if __name__ == "__main__":
    main()