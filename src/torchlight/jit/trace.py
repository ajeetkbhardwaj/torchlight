"""
Lightweight graph tracing ("mini JIT").

``trace(model, *example_inputs)`` runs one forward pass while recording every
:class:`Function` application into an operation tape.  The resulting
:class:`GraphTape` is a *detached* executable copy of the model: it can be
replayed on new inputs without rebuilding the autograd graph, exports an
op-level graph (useful for LLM debugging / graph dumps), and counts ops.

This is deliberately a small, readable precursor to a real JIT -- no fusion,
no codegen -- but the tape is exactly the intermediate representation a
compiler-style backend would consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..autograd.autodiff import Context, set_op_callback
from ..tensor import Tensor


@dataclass
class OpRecord:
    """
    One recorded forward operation.

    Attributes:
        fn: the :class:`~torchlight.autograd.functions.Function` applied.
        inputs: the input tensors (as recorded at trace time).
        output: the output tensor.
    """

    fn: Any
    inputs: Tuple[Tensor, ...]
    output: Tensor

    @property
    def name(self) -> str:
        """Short op name, e.g. ``"Add"``."""
        return self.fn.__name__

    def __repr__(self) -> str:
        shapes = ", ".join(str(getattr(i, "shape", "?")) for i in self.inputs)
        return f"{self.name}({shapes}) -> {self.output.shape}"


class Tracer:
    """
    Context manager that records every op executed while inside it.

        >>> with Tracer() as t:
        ...     y = model(x)
        >>> records = t.records
    """

    def __init__(self):
        self.records: List[OpRecord] = []
        self._previous_callback = None

    def __enter__(self) -> "Tracer":
        self._previous_callback = set_op_callback(self._record)
        return self

    def __exit__(self, *exc: Any) -> None:
        set_op_callback(self._previous_callback)
        return False

    def _record(self, fn: Any, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        self.records.append(OpRecord(fn=fn, inputs=inputs, output=output))


class GraphTape:
    """
    An executable op tape captured by :func:`trace`.

    ``run`` replays the recorded operations on fresh inputs *without* wiring
    the autograd graph (each op is executed with ``no_grad`` semantics).
    """

    def __init__(self, records: List[OpRecord], sample_inputs: Tuple[Tensor, ...], outputs: Tuple[Tensor, ...]):
        self.records = records
        self.sample_inputs = sample_inputs
        self.output_ids = tuple(o.unique_id for o in outputs)
        #: Snapshot constant-metadata tensors (per op inputs) for replay.
        self._snapshots: Dict[int, Tuple] = {}
        for rec in self.records:
            for inp in rec.inputs:
                if inp.unique_id not in self._snapshots:
                    self._snapshots[inp.unique_id] = (inp.to_numpy(), inp.shape)

    # -- Execution ------------------------------------------------------------
    def run(self, *run_inputs: Tensor) -> Tuple[Tensor, ...]:
        """
        Replay the tape with ``run_inputs`` standing in for the original
        sample inputs.  Returns the tuple of resulting tensors.
        """
        if len(run_inputs) != len(self.sample_inputs):
            raise ValueError(
                f"GraphTape expects {len(self.sample_inputs)} inputs (like the "
                f"original trace), got {len(run_inputs)}"
            )

        # Resolution maps: unique_id -> replay tensor.
        resolved: Dict[int, Tensor] = {
            s.unique_id: r for s, r in zip(self.sample_inputs, run_inputs)
        }

        for rec in self.records:
            args = []
            for inp in rec.inputs:
                if inp.unique_id in resolved:
                    args.append(resolved[inp.unique_id])
                else:
                    args.append(self._restore(inp))
            # Execute the plain forward with a no-grad context.
            ctx = Context(no_grad=True)
            out = rec.fn._forward(ctx, *args)
            resolved[rec.output.unique_id] = out

        return tuple(resolved[i] for i in self.output_ids)

    def _restore(self, t: Tensor) -> Tensor:
        """Recreate a constant tensor from its traced snapshot."""
        data, shape = self._snapshots[t.unique_id]
        return Tensor.make(data.reshape(-1), shape)

    # -- Introspection --------------------------------------------------------
    def count_ops(self) -> Dict[str, int]:
        """Tally occurrences of each op name."""
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[rec.name] = counts.get(rec.name, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        """Human-readable one-line-per-op dump of the tape."""
        lines = [f"{rec}" for rec in self.records]
        return "\n".join(lines)

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}x{v}" for k, v in sorted(self.count_ops().items()))
        return f"GraphTape({len(self)} ops: {counts})"


def trace(fn: Callable[..., Any], *sample_inputs: Tensor) -> Tuple[GraphTape, Tuple[Tensor, ...]]:
    """
    Capture one forward pass of ``fn`` as a :class:`GraphTape`.

    Args:
        fn: a callable (model, function, composed ops) taking tensors.
        *sample_inputs: example tensor arguments used to run the forward.

    Returns:
        ``(tape, outputs)`` where outputs are the tensors ``fn`` returned.

    Example:
        >>> tape, _ = trace(model, x_sample)
        >>> y = tape.run(x_new)              # same math, no autograd graph
        >>> tape.count_ops()                 # {'MatMul': 1, 'Add': 1, ...}
    """
    with Tracer() as tracer:
        out = fn(*sample_inputs)
    outs = out if isinstance(out, tuple) else (out,)
    tape = GraphTape(tracer.records, tuple(sample_inputs), outs)
    return tape, outs


def exec_tape(tape: GraphTape, *run_inputs: Tensor) -> Tuple[Tensor, ...]:
    """Alias for :meth:`GraphTape.run`."""
    return tape.run(*run_inputs)


def count_ops(tape: GraphTape) -> Dict[str, int]:
    """Alias for :meth:`GraphTape.count_ops`."""
    return tape.count_ops()


__all__ = ["OpRecord", "Tracer", "GraphTape", "trace", "exec_tape", "count_ops"]