# API Reference: utils

`torchlight.utils` stores and restores models.

```python
from torchlight.utils import save_model, load_model, save_state, load_state
from torchlight.utils import state_dict, load_state_dict
```

## Whole-model save/load

```python
def save_model(model: Any, path: str) -> None
def load_model(path: str) -> Any
```

Pickle / unpickle an entire `Module`.

```python
save_model(model, "model.pkl")
loaded = load_model("model.pkl")
```

## State dicts (portable, numpy)

```python
def save_state(module: Module, path: str) -> None
def load_state(module: Module, path: str) -> None
```

Write/read a model's named state as a `.npz` file (no pickling).

```python
save_state(model, "state.npz")
load_state(new_model, "state.npz")   # same weights, no pickle
```

## Dict helpers

```python
def state_dict(module: Module) -> Dict[str, np.ndarray]
def load_state_dict(module: Module, state, strict=True) -> None
```

`state_dict()` returns the numpy arrays behind every named parameter. With
`strict=True` (default) `load_state_dict` raises if the names don't match
exactly.

```python
state = state_dict(model)              # {'linear.weight': np.ndarray, ...}
load_state_dict(model, state)          # restore in place
```