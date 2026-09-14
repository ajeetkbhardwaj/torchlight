# API Reference: optim

`torchlight.optim` implements SGD, Adam, and AdamW, plus LR schedulers.

```python
from torchlight.optim import SGD, Adam, AdamW, StepLR, CosineAnnealingLR
```

## Optimizer (base)

```python
class Optimizer(parameters: Iterable[Parameter], defaults=None)
```

| Method | Description |
|--------|-------------|
| `zero_grad()` | zero the `.grad` of every parameter |
| `step()` | apply one update per parameter |
| `parameters` | the parameter list |

All optimizers require a list of `Parameter` objects (use
`model.parameters()`).

## SGD

```python
class SGD(parameters, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)
```

| Argument | Meaning |
|----------|---------|
| `lr` | learning rate |
| `momentum` | momentum factor (velocity) |
| `dampening` | momentum dampening |
| `weight_decay` | L2 penalty added to the gradient |
| `nesterov` | use Nesterov momentum |

```python
opt = SGD(model.parameters(), lr=0.1, momentum=0.9)
```

## Adam

```python
class Adam(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

| Argument | Meaning |
|----------|---------|
| `lr` | learning rate |
| `betas` | `(beta1, beta2)` moment decay factors |
| `eps` | numerical stabilizer |
| `weight_decay` | L2 (coupled) weight decay |

Uses bias-corrected first/second moments.

## AdamW

```python
class AdamW(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
```

Adam with **decoupled** weight decay (`w *= (1 - lr * lambda)`); default
`weight_decay=0.01`, recommended for modern transformers.

## Learning-rate schedulers

`torchlight.optim` ships with standard LR schedulers.  Each wraps an
optimizer, mutates its single global `lr`, and follows PyTorch's convention:
the first `.step()` advances to epoch `0` (the base lr, no decay yet).

```python
from torchlight.optim import SGD, StepLR, CosineAnnealingLR

optimizer = SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.001)

for epoch in range(50):
    ...
    optimizer.step()
    scheduler.step()
```

### LRScheduler (base)

```python
class LRScheduler(optimizer, last_epoch=-1)
```

| Method | Description |
|--------|-------------|
| `step()` | advance one epoch and call `get_lr()` |
| `last_epoch` | current epoch index |
| `state_dict()` / `load_state_dict(state)` | save / restore scheduler state |

Override `get_lr()` in subclasses to define a schedule.

### StepLR

```python
class StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

Multiply `lr` by `gamma` every `step_size` epochs.

### MultiStepLR

```python
class MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

Multiply `lr` by `gamma` at each epoch in `milestones` (a list of ints).

### ExponentialLR

```python
class ExponentialLR(optimizer, gamma, last_epoch=-1)
```

Multiply `lr` by `gamma` every epoch: `lr = lr_0 * gamma ** epoch`.

### CosineAnnealingLR

```python
class CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)
```

Cosine decay from the base lr to `eta_min` over `T_max` epochs.

| Scheduler            | lr at epoch `e` (`e` ≥ 0, `η = eta_min`, `γ = gamma`) |
| -------------------- | ------------------------------------------------------ |
| `StepLR`             | `lr0 · γ^(⌊e/step_size⌋)`                            |
| `MultiStepLR`        | `lr0 · γ^(#{m ∈ milestones : m ≤ e})`                |
| `ExponentialLR`      | `lr0 · γ^e`                                          |
| `CosineAnnealingLR`  | `η + (lr0 − η) · (1 + cos(π·e/T_max)) / 2`            |

## Example

```python
model = Sequential(Linear(4, 16), ReLU(), Linear(16, 1))
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
sched = CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)

for epoch in range(100):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    sched.step()
```