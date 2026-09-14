# Example: Class-Conditioned Autoencoder

`examples/sklearn_autoencoder.py` learns to *compress* an 8×8 digit image into
an 8-dimensional latent code and *rebuild* it, conditioned on the digit's
class. Understandable from one equation -- the autoencoder tries to make
reconstruction identical to the input.

## Anatomy of an autoencoder

```python
self.encoder = nn.Sequential(nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
                             nn.Linear(32, 8))
self.labels  = nn.Embedding(10, 8)                 # 8-d vector per digit class
self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                             nn.Linear(32, 64), nn.Sigmoid())
```

Forward: flatten the image to $\mathbf{x}\in\mathbb{R}^{64}$, squeeze it with
the encoder to a latent $\mathbf{z}\in\mathbb{R}^8$, look up the class vector
$\mathbf{c}$ from the embedding, concatenate, and decode back to an image:

$$\mathbf{z}=f_{enc}(\mathbf{x}),\qquad
\hat{\mathbf{x}}=\sigma\big(f_{dec}([\mathbf{z};\ \mathbf{c}])\big)$$

with $\sigma$ the sigmoid, chosen because pixel intensities $\in[0,1]$.

## Why condition on the class?

A plain autoencoder has to bury "which digit is this" into the 8 latent
numbers. Conditioning on the label makes the bundle `[z; c]` of size 16: the
*class-specific* structure ($c$) and the *instance-specific* stroke detail
($z$) split cleanly, which produces sharper reconstructions and a latent space
where style information is easy to read.

The targets come out of the DataLoader as `(B, 1)`, so the example flattens
them before the embedding lookup:

```python
c = self.labels(y.view(y.shape[0]))   # (B, 1) -> (B,) -> (B, 8)
```

## The loss: mean squared error

The reconstruction error, averaged over all pixels and examples:

$$\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\|\mathbf{x}_i-\hat{\mathbf{x}}_i\|_2^2
=\frac{1}{N}\sum_i\sum_k\big(x_{ik}-\hat{x}_{ik}\big)^2$$

Backprop pushes the decoder to rebuild and the encoder to keep whatever the
decoder needs -- the two do a tug-of-war that converges to a working latent
code. Because the objective compares pixel-averaged differences (not per-class
log-probabilities), MSE is the natural loss here;

### Reconstructed images converge to a mean absolute error of a few percent.

## Exercises

- Drop the class `Embedding` and watch reconstructions blur into generic
  digits.
- Interpolate between two test latents `z` and see the decoded images morph.