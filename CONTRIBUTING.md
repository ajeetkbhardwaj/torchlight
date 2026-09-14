# Contributing to Torchlight

Thanks for your interest! Torchlight is a small, from-scratch deep learning
framework, and every line exists to be read. Contributions that keep it small,
correct, and well-documented are very welcome.

## What this project values

- **Tiny surface area.** The whole engine is a tensor + a four-primitive
  backend (`map / zip / reduce / matmul`) + fused softmax/LayerNorm kernels.
  New features should compose out of these, not grow a parallel layer.
- **Readability over cleverness.** Docstrings explain *why*; formulas live in
  the docs (rendered with `$$...$$` + MathJax).
- **Numerical correctness.** Every new op gets a finite-difference gradcheck —
  math is cheap to assert precisely.
- **Numpy-only runtime.** `src/torchlight` depends on `numpy>=1.26` and
  nothing else. Anything optional (numba, sklearn, mkdocs) is an extra.

## Repository layout

```
src/torchlight/
├── core/        # scalar op specs, device, tensor_data (strides/broadcast)
├── backends/    # CPU (numpy), numba / cuda kernels over the primitive contract
├── autograd/    # Function / Context, reverse-mode autodiff, convs, indexing
├── tensor/      # Tensor object + factories
├── nn/          # Module / Parameter tree, layers, losses
├── optim/       # SGD / Adam / AdamW + LR schedulers
├── data/        # Dataset / DataLoader + synthetic problems
├── utils/       # serialization (state dict .npz, pickling)
└── jit/         # trace -> GraphTape -> count / replay
tests/           # pytest suite
examples/        # short standalone demos
projects/        # end-to-end experiments on real data
docs/            # mkdocs site (tutorials, projects, examples, API)
```

## Development setup

```bash
git clone https://github.com/<you>/torchlight
cd torchlight

# any Python 3.10+ environment works
conda create -n torchlight python=3.12
conda activate torchlight

pip install -e ".[test]"   # package + pytest
pytest                     # run the suite (default CPU backend)
```

The suite is deliberately fast (a few seconds) and CPU-only. Run it before and
after your changes:

```bash
pytest -q
```

## Conventions

### Style

- Type hints everywhere (`from __future__ import annotations` is already used
  in the codebase).
- Public API gets a docstring; private helpers get a one-liner. Avoid
  inline comments that restate the code — put the *why* in a docstring.
- Follow the style of the file you're editing; keep changed lines minimal so
  diffs stay reviewable.
- We track lint with `ruff`. The repo currently carries pre-existing
  upgrade-style findings from newer rule defaults — don't *add* new ones, and
  only `ruff check --fix` the lines you touched.

### Adding a new operation

1. **Spec** the math in `core/ops` (e.g. `scalar.foobar`).
2. **Autograd**: add a `Function` subclass in `autograd/functions.py`
   (`forward` builds the output; `backward(ctx, d_out)` returns per-input
   grads, reusing `ctx`-saved constants where possible).
3. **Backend**: implement via the primitives in `backends/cpu.py`
   (`map`/`zip`/`reduce`/`matmul`); add the fused kernel to the numba backend
   only if it materially speeds a hot path.
4. **Wire up**: export it on `Tensor` (if it's a typical array op) and/or as a
   functional in `nn/functional.py`.
5. **Test**: forward correctness + a `gradcheck` against the scalar
   derivative; add to the matching `tests/` file.

### Numeric stability

This engine builds softmax, log-softmax and cross-entropy with hostile inputs
in mind (`exp(x - max)` etc.). New ops must be NaN-safe on extreme values, and
fused kernels must match the CPU backend to float tolerance — the test suite
locks op parity between backends.

## Documentation

Docs are mkdocs-material in `docs/`, served on GitHub Pages from `main`.

- Tutorials / API pages live in `docs/tutorials/` and `docs/api/`.
- Project pages (`docs/projects/*.md`) explain one end-to-end experiment with
  first-principles maths.
- Example pages (`docs/examples/*.md`) do the same for `examples/`.
- New pages must be added to the `nav:` in `mkdocs.yml`.

Math is plain LaTeX inside mkdocs: `$inline$` and `$$display$$`. Diagrams are
Mermaid code fences. After editing docs, verify:

```bash
pip install mkdocs mkdocs-material
mkdocs build --strict   # fails on any broken link / bad nav
mkdocs serve            # http://127.0.0.1:8000 for a live preview
```

## Testing your changes

- `pytest -q` — entire suite, must stay green.
- `mkdocs build --strict` — any touched docs must build.
- For new projects keep everything runnable in one step
  (`python train.py` → saves `out/`, `python post.py` → evaluates), cached
  data inside `projects/<name>/data/`, no secrets, no absolute paths.

## GitHub workflow

- Work on a feature branch: `git checkout -b feat/my-change`.
- Keep pull requests small and focused; one logical change per PR.
- Commit messages: short imperative summary line, optional detail body.
- **CI** runs on every PR (`.github/workflows/ci.yml`): the pytest suite on
  Python 3.10/3.11/3.12 (Ubuntu) and a `mkdocs build --strict`.
- **CD** (`.github/workflows/pages.yml`) builds `docs/` and deploys to
  GitHub Pages whenever `main` is pushed.

To get Pages CD working in your fork:

1. Push the repo: `git init && git add . && git commit -m "init" `
   `git branch -M main && git remote add origin ... && git push -u origin main`.
2. GitHub → **Settings → Pages → Source: "GitHub Actions"**.
3. Push again (or re-run the workflow) — the site appears at
   `https://<user>.github.io/<repo>/`.

## Reporting issues

When opening a bug report, include:

- the exact reproduction (script or snippet),
- the `torchlight.__version__`, Python version, OS, and backend
  (`TORCHLIGHT_DEVICE` / whether `.[cuda]` is installed),
- the full traceback, and what you expected.

## License

By contributing you agree your changes are licensed under the same MIT license
as the project.
