# Project: Transformer from Scratch (tiny GPT)

The translation project's attention was *target-to-source*. The Transformer
uses **self-attention** -- every position attends to every earlier position in
the same sequence -- which is what powers modern language models. This project
trains a miniature GPT on real Shakespeare text, character by character, using
only torchlight primitives (there is no `Attention` module; you build it).

## The task and the data

Tiny Shakespeare is ~1.1M characters of public-domain plays. A character-level
language model learns $P(c_{t+1}\mid c_1,\dots,c_t)$, the probability of the
next character given what came before. Its vocabulary is just the 65 distinct
characters in the file, so every token is one character.

Training window: a `BLOCK` of 64 characters; the model must predict the
character *following* each position. The validation slice is the last 10% of
the text (deterministic positional split reused by `post.py`).

## Model from first principles

### 1. Input embeddings: characters + where they are

Each character id $v$ and each position $t$ get their own learned vector from
two embedding matrices (this is a huge win -- attention itself is
permutation-equivariant, so it needs an explicit position signal):

$$\mathbf{z}_t = E_{tok}[v_t] + E_{pos}[t] \in \mathbb{R}^{64}$$

### 2. Causal self-attention

Every position computes a **query**, a **key** and a **value** from its
embedding via learned projections:

$$\mathbf{q}_t=\mathbf{z}_tW_Q,\quad \mathbf{k}_t=\mathbf{z}_tW_K,\quad
\mathbf{v}_t=\mathbf{z}_tW_V$$

"Self" means keys/values come from *other positions of the same sequence*.
The relevance of token $j$ to token $t$ is the scaled dot product; a causal
**mask** $M$ sends positions $j>t$ (the future) to $-\infty$ so softmax gives
them weight zero -- the model never peeks ahead:

$$\mathrm{attn}_{tj} = \mathrm{softmax}_j\!\Big(\frac{\mathbf{q}_t\cdot
\mathbf{k}_j}{\sqrt{d_h}} + M_{tj}\Big),\qquad M_{tj}=\begin{cases}0&t\ge j\\
-\infty & t<j\end{cases}$$

$$\mathbf{o}_t = \sum_{j\le t}\mathrm{attn}_{tj}\,\mathbf{v}_j$$

Dividing by $\sqrt{d_h}$ keeps the softmax from saturating as `d_h` grows
(the inputs to softmax don't explode). This is done for `n_head=4` heads in
parallel on `d_h=16`-dimensional slices, then the heads are concatenated and
mixed with `W_O` -- a parameter-efficient way to let different heads specialise
(e.g. one head tracks "the next char is a vowel", another tracks "has an o").
In code, heads come from a single `Linear(n_embd, 3*n_embd)` split with
`chunk` and reshaped via `view -> permute`.

### 3. LayerNorm and the residual road

Training many stacked layers is only stable if gradients flow by shortcut.
Every block uses **pre-norm layers with residual connections**:

$$\mathbf{h}' = \mathbf{h} + \mathrm{Attn}\big(\mathrm{LN}(\mathbf{h})\big),\qquad
\mathbf{h}'' = \mathbf{h}' + \mathrm{MLP}\big(\mathrm{LN}(\mathbf{h}')\big)$$

LayerNorm normalises each row to zero mean / unit variance, then re-scales:
with $\mu,\sigma$ per-position-row,

$$\mathrm{LN}(\mathbf{x}) = \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\varepsilon}}\odot
\gamma + \beta$$

(shareable $\gamma,\beta$ per feature). The MLP is a two-layer ReLU network
`ReLU(x W_1) W_2` expanding to 4× width. The residual keeps gradients alive --
that is what makes 100s of layers trainable in real LLMs.

### 4. The head and the loss

The final `Linear` maps each position's representation to logits over the 65
characters. Training objective is next-character cross-entropy:

$$\mathcal{L} = -\frac{1}{\sum_t m_t}\sum_t m_t\ \log P(c_{t+1}\mid c_{1:t})$$

where a shifted target makes position $t$ answer for $c_{t+1}$ and the final
column is masked (`y` was rolled left by one; `mask[:, -1] = 0`) because there
is no "next char" after the window. Reported alongside the loss is the
**perplexity** $\exp(\mathcal{L})$: the model's average number of characters it
is unsure between.

## Training

Adam, lr $3\times10^{-3}$, batch 64 windows, 1200 gradient steps
(`TORCHLIGHT_ITERS` to override). A one-iteration baseline on real data:

```
step     1/1200 ... val 4.362  perplexity 78.4
```

-- near the ~65 you'd get from random guessing, and easing downward as
training climbs to 1200 steps. `train.py` saves `out/gpt_model.npz`,
`meta.json` (the character table) and `history.json`.

## Generation: turning a distribution into text

Inference recycles the training forward pass one window at a time. From the
logits of the last position, sample the next id (temperature softmax):

$$p(v)=\frac{\exp(z_v/T)}{\sum_{v'}\exp(z_{v'}/T)},\qquad T<1 \text{ sharpens},
\;T>1 \text{ flattens}$$

Low temperature repeats the char-rnn's most confident words; high temperature
wanders. Greedy $\arg\max$ is just $T\to0$. `post.py` generates a sample at
T=0.8 and one at T=1.2, so the temperature readout is visible.

## Evaluation (`post.py`)

`projects/transformers-from-scratch/post.py` reloads the checkpoint and reports
on the held-out validation slice:

* val cross-entropy loss,
* **perplexity** $e^{L}$,
* **top-1 character accuracy** -- the fraction of positions where the greedy
  argmax character matches the true next character,
* fresh generated text at two temperatures,
* a loss-curve PNG from `history.json`.

## Run it yourself

```bash
python projects/transformers-from-scratch/train.py  # trains + saves out/
python projects/transformers-from-scratch/post.py   # perplexity + generation + plot
# quick checks: TORCHLIGHT_ITERS=300
```

## What this project teaches

- self-attention as a weighted lookup built from $Q,K,V$ -- the core of every
  modern LM,
- causal masking so nothing reads the future,
- why scaling by $1/\sqrt{d_h}$ matters,
- the shape gymnastics of multi-head attention (`view`/`permute`/`chunk`),
- LayerNorm + residuals as the stability recipe,
- temperature sampling as the "creative knob" of generative models.

The three projects now cover three architectures on real data: **pooling MLP**
(sentiment), **GRU + attention** (translation), and **GPT** (generation) --
the arc from bags-of-words to language models, all running on torchlight's own
autograd and backends (see [Tutorial 5 — Backends](../tutorials/05-backends.md)).