# Example: MLP Classifier on Breast Cancer

`examples/sklearn_classifier.py` is a complete, from-scratch deep-learning
solution to a classic scikit-learn task (the Wisconsin Breast Cancer set), and
it exercises nearly every training technique torchlight ships. This page
derives the pieces; the flow is also summarised in
[Tutorial 7](../tutorials/07-real-world-datasets.md).

## Data

30 numeric features per tumour (radius, texture, perimeter, ...), binary label
malignant/benign. Sklearn-style preprocessing standardises every feature:

$$z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}$$

so no feature dominates and Adam's gradients are well scaled.

## The model: a deeper MLP

```python
nn.Sequential(
    nn.Linear(30, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 1),
)
```

**BatchNorm** re-centres each mini-batch during training so the layer above
always sees data with stable mean/variance (with running estimates at
inference):

$$\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},\qquad y=\gamma\hat{x}+\beta$$

**Dropout** randomly zeroes `p=0.3` of activations per step, forcing the
network not to rely on any single feature (a cheap ensemble):

$$\tilde{a}_i=\frac{1}{1-p}\,m_i\,a_i,\qquad m_i\sim\mathrm{Bernoulli}(1-p)$$

## Objective and tricks

The loss is binary cross-entropy on the logit (derived in the
[sentiment project](../projects/sentiment-classifier.md)), Adam with
$10^{-3}$, and:

* **Cosine annealing** schedules the learning rate from $\eta_{max}$ down to
  $\eta_{min}$:
  $$\eta_t=\eta_{min}+\tfrac12(\eta_{max}-\eta_{min})\big(1+\cos(t\pi/T_{max})\big)$$
* **Gradient clipping** caps the update size when the total gradient norm is
  huge (as can happen near saddle points):
  $$g\leftarrow g\cdot\min\Big(1,\ \frac{\text{max\_norm}}{\|g\|_2}\Big)$$

## Results

Runs in a few seconds on a laptop and reaches ~97% test accuracy:

```
Reaches ~97% test accuracy in under 10 seconds.
```

## Exercises

- Remove BatchNorm -- watch the loss struggle to converge.
- Raise dropout to 0.6 -- the model underfits (accuracy drops).
- Replace the cosine schedule with a constant LR -- later epochs wobble more.