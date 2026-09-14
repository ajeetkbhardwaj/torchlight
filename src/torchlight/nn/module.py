"""
``Module`` / ``Parameter``: the tree structure behind every neural network.

A ``Module`` owns two registries:

* ``_modules`` -- child modules (set via attribute assignment),
* ``_parameters`` -- trainable :class:`Parameter` objects.

Assignment is intercepted by ``__setattr__`` / ``__getattr__`` so that::

    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.fc = Linear(2, 4)     # becomes a child module
            self.bias = Parameter(...) # becomes a registered parameter

``named_parameters``/``parameters`` recursively gather everything below the
module -- exactly what optimizers iterate over.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..tensor import Tensor


class Parameter:
    """
    A trainable container: a tensor plus a name.

    Parameters live inside :class:`Module`; optimizers read ``param.value``
    and call :meth:`update` to write back the new (detached) value.
    """

    def __init__(self, data: Any, name: Optional[str] = None, requires_grad: bool = True):
        #: The current value tensor.  May be any object for flexibility, but
        #: trainable parameters are always Torchlight tensors.
        self.value = data
        if isinstance(data, Tensor):
            self.value.requires_grad_(requires_grad)
        if name is not None:
            self.name = name

    def update(self, value: Any) -> None:
        """
        Replace this parameter's value with ``value``.

        The incoming tensor (typically ``value - lr * grad``) is a graph
        intermediate; we rebind to a *detached leaf* so the next forward
        starts from a clean parameter node.
        """
        if isinstance(value, Tensor):
            value = value.detach().requires_grad_(True)
        self.value = value

    @property
    def data(self) -> Any:
        """Alias for ``value`` (matches torch's ``param.data``)."""
        return self.value

    def __getattr__(self, key: str) -> Any:
        # Forward any other attribute straight to the underlying tensor so a
        # Parameter behaves like the tensor it wraps (e.g. param.shape).
        value = object.__getattribute__(self, "value")
        return getattr(value, key)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Parameter({self.value!r})"


class Module:

    #: Registered child modules (by attribute name).
    _modules: Dict[str, "Module"]
    #: Registered parameters (by attribute name).
    _parameters: Dict[str, Parameter]
    #: ``True`` while in training mode (affects dropout / batchnorm).
    training: bool

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            object.__setattr__(self, key, val)

    def __getattr__(self, key: str) -> Any:
        # Only called when normal lookup fails (i.e. attribute not on instance).
        params = self.__dict__.get("_parameters", {})
        if key in params:
            return params[key]
        modules = self.__dict__.get("_modules", {})
        if key in modules:
            return modules[key]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {key!r}")

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        """Register a named child module."""
        if not isinstance(module, Module):
            raise TypeError(f"add_module expects a Module, got {type(module)!r}")
        self._modules[name] = module

    def add_parameter(self, name: str, parameter: Any) -> Parameter:
        """Register (and return) a named parameter."""
        if isinstance(parameter, Parameter):
            param = parameter
        else:
            param = Parameter(parameter, name=name)
        self._parameters[name] = param
        return param

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def modules(self) -> List["Module"]:
        """Direct child modules."""
        return list(self._modules.values())

    def named_modules(self) -> Iterator[Tuple[str, "Module"]]:
        """Yield ``(name, module)`` pairs for this module and descendents."""
        yield "", self
        for name, mod in self._modules.items():
            for sub_name, sub_mod in mod.named_modules():
                yield f"{name}.{sub_name}" if sub_name else name, sub_mod

    def parameters(self) -> List[Parameter]:
        """All parameters of this module and its descendents."""
        return [p for _, p in self.named_parameters()]

    def named_parameters(self) -> List[Tuple[str, Parameter]]:
        """
        All ``(name, parameter)`` pairs of this module and its descendents,
        with dotted names like ``"fc.weight"``.
        """
        params: Dict[str, Parameter] = {}
        for k, v in self._parameters.items():
            params[k] = v
        for mod_name, m in self._modules.items():
            for k, v in m.named_parameters():
                params[f"{mod_name}.{k}"] = v
        return list(params.items())

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Switch this module and all descendents into training mode."""
        for m in self.modules():
            m.train()
        self.training = True

    def eval(self) -> None:
        """Switch this module and all descendents into evaluation mode."""
        for m in self.modules():
            m.eval()
        self.training = False

    # ------------------------------------------------------------------
    # Calling / printing
    # ------------------------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        """Subclasses implement the actual forward pass."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """PyTorch-style summary: module name, child modules, and params."""
        parts = []
        for key, mod in self._modules.items():
            parts.append(f"({key}): {mod!r}")
        for name, p in self._parameters.items():
            parts.append(f"({name}): Parameter(shape={tuple(p.value.shape)})")
        if not parts:
            return f"{type(self).__name__}()"
        body = ",\n".join(self._add_indent(part, 2) for part in parts)
        return f"{type(self).__name__}(\n{body}\n)"

    @staticmethod
    def _add_indent(s: str, num_spaces: int) -> str:
        """Indent *every* line of ``s`` by ``num_spaces``."""
        pad = num_spaces * " "
        return pad + s.replace("\n", "\n" + pad)