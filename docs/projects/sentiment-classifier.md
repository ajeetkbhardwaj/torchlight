# Project: Sentiment Classifier

Sentiment analysis says "positive or negative?" about free-form text. This
project builds that classifier from scratch: from raw reviews to numbers, then
through an embedding + pooling + MLP that torchlight trains with its own
autograd. The goal of this page is to derive **every** component from first
principles, following the notebook left in `projects/sentiment-classifier/`.

## The task and the data

The UCI *Sentiment Labelled Sentences* set gives 3000 real reviews -- one
labelled sentence each from **IMDb**, **Amazon** and **Yelp** -- split 1000/1000/1000.
Each line is `"review text<TAB>0|1"`, downloaded once into
`projects/sentiment-classifier/data/`.

The job: learn $f:\text{review}\rightarrow[0,1]$ = probability the review is positive.

## From words to numbers

Neural networks multiply and add numbers; they cannot directly see strings.
The first step is to build a **vocabulary**:

1. Lowercase and split every review into tokens (letters/numbers/apostrophes).
2. Count token frequencies across the training half of the reviews.
3. Keep the 10,000 most frequent tokens, assigning each a unique integer id:
   token $w\mapsto \mathrm{id}(w)$, with `0 = PAD`, `1 = UNK`.

A review becomes a vector of ids of fixed length $T=64$:

$$x^{(i)} \in \{0,1,\dots,V-1\}^{64}$$

Padding (`PAD`=0) fills everything past the true length, and a binary *mask*
remembers which positions are real tokens:

$$m^{(i)}_t = \begin{cases}1 & \text{token } t \text{ is a real word}\\0 & \text{it is padding}\end{cases}$$

Why a mask? Because the pooling step below must **ignore padding**, or every
review of different length would implicitly leak its length into the model.

## Model from first principles

### 1. Embedding: the learnable lookup table

An embedding turns each token id into a vector. This is nothing more than a
matrix $E \in \mathbb{R}^{V \times d}$ whose $v$-th row is the vector for token
$v$. Writing the id as a one-hot vector $\mathbf{1}_v$, the lookup is a product:

$$\mathbf{e} = E^{\top}\mathbf{1}_v  \quad(\text{= the } v\text{-th row of } E),\qquad d=128$$

During training the rows of $E$ are free parameters, so words that co-occur
with the same sentiment (e.g. "amazing", "love") learn similar vectors.

### 2. Masked mean pooling: from 64 vectors to one

Each review is now a sequence of 64 vectors $(\mathbf{e}_1,\dots,\mathbf{e}_{64})$.
We summarise them with a **length-corrected average** (the dot product keeps
only the masked positions):

$$\mathbf{h} = \frac{\sum_{t} m_t\,\mathbf{e}_t}{\sum_{t} m_t}$$

This is `nn.Embedding` output fed to `F.masked_mean(x, mask, dim=1)`. The result
$\mathbf{h}\in\mathbb{R}^{128}$ is a single "bag of learned word vectors" -- a
first-principles approximation to sentence meaning. (Sequence order is thrown
away here; the translation and transformer projects show models that keep it.)

### 3. The MLP head: stacking affine maps

A classifier needs to mix the 128 pooled features into one logit. A linear
(purely affine) map alone can only fit lines, so we insert a nonlinearity:

$$\mathbf{a} = \mathrm{ReLU}\big(\mathbf{h}W_1^{\top} + \mathbf{b}_1\big), \qquad
\mathbf{a}\in\mathbb{R}^{64}$$

$$z = \mathbf{a}W_2^{\top} + b_2, \qquad z \in \mathbb{R}^{1}\ \text{(the raw score / logit)}$$

`ReLU` is $\max(0,\cdot)$; it provably makes the composed map able to compute
any piecewise-linear decision boundary. In code that is exactly the layers
`Linear(128,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)`.

### 4. Turning a raw score into a probability

A logit $z\in(-\infty,\infty)$ is squashed to $[0,1]$ by the **logistic**
(sigmoid) function:

$$\hat{p} = \sigma(z) = \frac{1}{1+e^{-z}},\qquad\text{so } \sigma'(z)=\sigma(z)(1-\sigma(z))$$

Prediction threshold: $\hat{y}=1$ if $\hat{p} > 0.5$, i.e. simply $z>0$.

## The loss: binary cross-entropy from first principles

If the true label is $y\in\{0,1\}$, the *correct* model should assign
probability $\hat{p}\approx y$. The likelihood of the observed labels is

$$\prod_i \hat{p}_i^{\,y_i}\big(1-\hat{p}_i\big)^{1-y_i}$$

Maximising the likelihood = minimising its negative log = **binary
cross-entropy**,

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i\log\hat{p}_i +
(1-y_i)\log(1-\hat{p}_i)\Big]$$

Plugging in $\hat{p}=\sigma(z)$ and simplifying gives the famous stable form
used in `F.binary_cross_entropy_with_logits` -- never apply $\log(0)$ or
cexplode:

$$\mathcal{L} = \frac{1}{N}\sum_i \Big[\max(z_i,0) - z_i y_i +
\log\big(1+e^{-|z_i|}\big)\Big]$$

The gradient the autograd engine computes flows back: $\partial z/\partial W$
is the input itself, so weights move in the direction that lowers $\mathcal{L}$
(see [Tutorial 2 — Autograd](../tutorials/02-autograd.md)).

## Training

Adam (first- and second-moment adaptive steps -- `torchlight.optim.Adam`) is
run for 8 epochs, batch 64, lr $10^{-3}$, on a deterministic 80/20 split of the
shuffled pairs.

```
[data] 3000 reviews | vocab 4675 | train 2400 | test 600
epoch  8  loss 0.5756  test acc 66.83%
[save] checkpoint -> out/sentiment_model.npz + meta.json
```

Every training detail lives in `projects/sentiment-classifier/train.py`, which
afterwards serialises the state dict to `out/sentiment_model.npz` plus the
vocabulary in `out/meta.json`.

## Evaluation and metrics (`post.py`)

`projects/sentiment-classifier/post.py` reloads the checkpoint, reproduces the
**same** test split via `train.prepare()` (fixed seed), and reports the
four classic metrics defined from the confusion counts (TP/FP/FN/TN):

$$\text{acc}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
\text{prec}=\frac{TP}{TP+FP},\qquad
\text{rec}=\frac{TP}{TP+FN},\qquad
F_1=\frac{2\cdot prec\cdot rec}{prec+rec}$$

A real run prints:

```
[eval] 600 held-out reviews
accuracy : 66.83%    precision: 68.42%    recall: 58.28%    F1: 62.94%
```

plus the confusion matrix, five confident and five borderline predictions, and
a PNG (`out/sentiment_eval.png`) with the loss curve and confusion heatmap.

## Run it yourself

```bash
python projects/sentiment-classifier/train.py   # trains + saves out/
python projects/sentiment-classifier/post.py    # metrics + plots
# optional knobs: TORCHLIGHT_EPOCHS=15, TORCHLIGHT_DEVICE=cuda on a GPU box
```

## What this project teaches

- how text becomes discrete ids then dense vectors (one-hot $\to$ embedding row),
- masked pooling to handle variable length,
- why logits + cross-entropy is numerically friendly while raw squared error on
  probabilities is not,
- the full train/evaluate/save/load lifecycle of a torchlight model.

Next: the [machine-translation project](./machine-translation.md) replaces the
bag-of-words with a recurrent sequence model that sees order.