# Project: English → French Translation

The sentiment project read a review and threw its word order away (bag of
words). Translation has to **respect and reorder words**: *"i'm not going to
hurt you"* is not a bag of words in French. This project builds a sequence to
**sequence** model from scratch -- a GRU encoder that reads the sentence, and
a GRU decoder with **attention** that writes the translation word by word.

All of it runs on the real Tatoeba English-French corpus and teaches the
mechanics that newer LLMs still use (token embeddings, attention, teacher
forcing, autoregressive decoding).

## The task and the data

`train.py` pulls the classic PyTorch tutorial corpus
(`data/eng-fra.txt`): ~135k real EN-FR sentence pairs. After light
normalisation and cutting pairs longer than 8 words on either side, we train on
the full remainder -- 102,659 pairs -- split 90/10:

```
[data] 102,659 pairs | train 92,393 | val 10,266
```

(splitting reproduces deterministically via a fixed seed, so `post.py` always
evaluates the identical validation set).

## From sentences to tensors

Tokenise whitespace, then map words to ids exactly like the sentiment project:

$$
w \mapsto \mathrm{id}(w),\qquad \text{UNK}\to3,\ \text{PAD}\to0,\ \text{SOS}\to1,\ \text{EOS}\to2
$$

An English sentence $[w_1,\dots,w_n]$ becomes the encoder input
$[\mathrm{id}(w_1),\dots,\mathrm{id}(w_n),\text{EOS}]$ (EOS tells the encoder
"end of input"). Its French target $[t_1,\dots,t_m]$ is split into **two**
aligned sequences -- the decoder *input* starts with SOS, the *target* ends
with EOS:

$$
x=[w_1,\dots,w_n,\text{EOS}],\qquad
\mathrm{dec\_in}=[\text{SOS},t_1,\dots,t_m],\qquad
\mathrm{tgt}=[t_1,\dots,t_m,\text{EOS}]
$$

Padding to the batch maximum is tracked by a mask $m_t$ over target positions.

## Model from first principles

### 1. The GRU cell: a memory that updates itself

A plain network can't count positions; an RNN can, by threading a hidden state
$h_t$ through time. The **Gated Recurrent Unit** computes the new state by a
weighted blend of "remember" and "replace":

$$
\underbrace{z_t}_{\text{update}}=\sigma(W_z x_t + U_z h_{t-1} + b_z),\qquad
\underbrace{r_t}_{\text{reset}}=\sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
\tilde{h}_t=\tanh\big(W_n x_t + U_n (r_t \odot h_{t-1})+b_n\big),\qquad
h_t = \tilde{h}_t + z_t \odot (h_{t-1}-\tilde{h}_t)
$$

When $z_t\to1$ the state keeps its memory; when $z_t\to0$ it takes the new
candidate. Torchlight has no built-in RNN, so `GRUCell` in `train.py` is built
from three `Linear` pairs plus `sigmoid`/`tanh` -- a direct line-by-line
translation of these equations.

### 2. Encoder: read the whole sentence

Run the GRU over the input positions, keeping **every** hidden state (not just
the last one). Each encodes "the sentence meaning up to word $t$":

$$
\text{Encoder}: (emb(x_1),\dots,emb(x_n)) \longmapsto (h^{enc}_1,\dots,h^{enc}_n),\qquad
h^{enc}_t=\mathrm{GRU}(emb(x_t), h^{enc}_{t-1})
$$

### 3. Attention: don't forget anything

The decoder starts from the last encoder state, but long sentences push early
information out of a fixed-size vector. **Attention** fixes this: at each
output step the decoder computes a weighted summary (context) of *all* encoder
states, in Luong's dot-product style:

$$
e_{t i} = h^{dec}_t\cdot h^{enc}_i,\qquad
a_{t i} = \frac{\exp(e_{t i})}{\sum_{i'} \exp(e_{t i'})},\qquad
c_t = \sum_i a_{t i}\, h^{enc}_i
$$

Padded encoder positions get $e=-\infty$ ($\exp=0$) via the `enc_penalty`
tensor, so **padding cannot be attended to**. The per-step output logits are

$$
\mathrm{logits}_t = W_{out}\,\tanh\big([h^{dec}_t;\; c_t]\big)
$$

-- a pairing of "what the decoder is thinking" with "what the sentence said what it should look at next".

### 4. Autoregressive decoding and teacher forcing

At train time the decoder receives the **true** previous token (teacher
forcing), which makes the early game easy and slopes fast. The toy-model loss
at position $t$ is the cross-entropy of its prediction against the target token
$t+1$ -- note position $m$ must predict **EOS** (this was a subtle training bug
until targets gained a trailing EOS and the mask ran to $m{+}1$):

$$
\mathcal{L}=\frac{ \sum_{b}\sum_{t} m_{bt}\,\mathrm{CE}\big(\mathrm{softmax}(\mathrm{logits}_{bt}),\;y_{(b,t+1)}\big)}{\sum_b\sum_t m_{bt}}
$$

At inference there are no true tokens, so the model **feeds its own previous
prediction** back (greedy: pick $\arg\max$) until it emits EOS. That loop is the
entire GPT decoding loop in miniature.

## Training

Teacher-forced GRU with Adam ($10^{-3}$), batch 64, embeddings/hidden = 128,
3 epochs over the full corpus. Even a *quick* sub-corpus run (2 epochs on 20k
pairs) already shows the model producing real French words, once the target
gets its trailing EOS:

```
epoch 2  loss 3.865
    en: i'm not going to hurt you.
    fr: je ne vais pas vous faire de mal.
    ->  je ne suis pas de vous ?      (greedy, undertrained -- but real French words)
```

Run the script at full scale (`python projects/machine-translation/train.py`)
and the greedy outputs sharpen into genuine translations.

## Evaluation and metrics (`post.py`)

`projects/machine-translation/post.py` loads the checkpoint, re-derives the
identical validation split, and greedy-decodes 100 held-out sentences, then
computes:

* **Exact match**: fraction where the decoded sentence equals the reference.
* **Sentence BLEU** (bigram precision + brevity penalty, a cheap stand-in for
  the full metric). For a reference $r$ and hypothesis $h$,
  $$
  BLEU = \min\!\Big(1,\,e^{1-|r|/|h|}\Big)\times \frac{1}{2}\Big(p_1+p_2\Big),
  \qquad p_k=\frac{\sum_{\text{$k$-grams }g} \min(\mathrm{count}_h(g),
  \mathrm{count}_r(g))}{\sum_g \mathrm{count}_h(g)}
  $$
* Example English/French/prediction triples, plus a PNG histogram of per-sentence BLEU.

## Run it yourself

```bash
python projects/machine-translation/train.py   # trains + saves out/mt_model.npz
python projects/machine-translation/post.py    # BLEU, exact-match, translations, plot
# optional knobs: TORCHLIGHT_MT_PAIRS=20000, TORCHLIGHT_EPOCHS=2   (quick tests)
```

## What this project teaches

- RNN/GRU state updates and why gates fix the vanishing-gradient problem,
- how sequence order is preserved (unlike the sentiment bag-of-words),
- scaled/masked attention as a way to *read the source directly*,
- the off-by-one lurking in sequence-to-sequence targets (decoder input starts
  with SOS, targets end with EOS),
- teacher forcing at train time vs. autoregressive sampling at generate time.

Next: the [transformers-from-scratch project](./transformers-from-scratch.md)
replaces recurrence with self-attention -- the architecture inside modern
language models.
