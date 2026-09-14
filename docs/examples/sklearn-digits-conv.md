# Example: CNN on Handwritten Digits

`examples/sklearn_digits_conv.py` treats the 8×8 handwritten digits as images
and classifies them with a genuine convolutional network -- `Conv2d` and
`MaxPool2d` are torchlight's own autograd convolutional layers (see
`src/torchlight/autograd/convolutions.py`).

## Images as tensors

Each `load_digits()` sample `(8, 8)` is reshaped to `(1, 8, 8)` --
(channel, height, width). The full batch is `(B, 1, 8, 8)`.

## Convolution from first principles

A convolution slides a small kernel over the image and writes, at each spot,
the weighted sum of the pixels in its receptive field. For a single kernel
$w\in\mathbb{R}^{k_h\times k_w}$ with stride 1 and padding $p$:

$$\text{out}[i,j]=\sum_{a=0}^{k_h-1}\sum_{b=0}^{k_w-1}
x\big[i+a-p,\ j+b-p\big]\cdot w[a,b]$$

<br>

Key differences from a dense `Linear`:

1. **Weight sharing** -- the same $w$ is applied at every location, so the
   layer has $k_h k_w \times C_{in} \times C_{out}$ parameters instead of one
   per indefinite pixel. Translation-invariance is baked in.
2. **Locality** -- an output pixel only ever sees its $k\times k$ neighbourhood.
3. **Channels** -- a `Conv2d(1, 8, 3, padding=1)` produces 8 output channels,
   i.e. 8 feature maps, each a different learned filter
   (edges, strokes, loops...).

This network is:

```python
self.conv1 = nn.Conv2d(1, 8, 3, padding=1)   # (B, 8, 8, 8)
self.pool1 = nn.MaxPool2d(2)                  # (B, 8, 4, 4)
self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # (B, 16, 4, 4)
self.pool2 = nn.MaxPool2d(2)                  # (B, 16, 2, 2)
self.fc    = nn.Linear(16 * 4, 10)
```

## Max pooling

Max pooling downsamples each 2×2 patch by keeping its maximum -- a crude
"is there any strong response here?" that shrinks the map size by 4× while
making the features slightly translation-robust:

$$\text{pool}[i,j]=\max_{a,b\in\{0,1\}} \text{out}[2i+a,\ 2j+b]$$

## Classification head

The final feature maps are flattened and fed to a 10-way classifier trained
with softmax cross-entropy over the ten digits:

$$p_k=\frac{e^{z_k}}{\sum_{j}e^{z_j}},\qquad
\mathcal{L}=-\log p_{y} \quad(\text{the NLL of the true digit})$$

## Results

Tiny images already respond to tiny kernels: the conv net converges quickly on
`load_digits()` to high accuracy (well into the high 90s % on the test split),
still in just a second or two of laptop training.

## Exercises

- Remove a conv/pool stage and watch accuracy (and convergence speed) drop.
- `visualize the filters` -- print `model.conv1.weight` after training and see
  the learned edge detectors.