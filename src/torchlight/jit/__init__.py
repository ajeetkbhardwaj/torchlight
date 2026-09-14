"""
torchlight.jit -- lightweight graph tracing.

* :class:`Tracer` / :func:`trace` -- record a forward pass as an op tape.
* :class:`GraphTape` -- replay the tape on new inputs (mini "compiled" graph).
"""

from .trace import GraphTape, OpRecord, Tracer, count_ops, exec_tape, trace

__all__ = ["Tracer", "trace", "OpRecord", "GraphTape", "count_ops", "exec_tape"]