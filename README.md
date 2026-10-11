# gpt-lab

A GPT written from scratch in pure Python. No PyTorch, no NumPy. Each number
in the model is a scalar node with its own gradient.

The purpose is to learn. The code is small enough to read every line. When
the base model works, the lab part starts: swap one transformer component at
a time (for example, RoPE for learned positions, or Muon for AdamW) and
compare the results.

The inspiration is Karpathy's `microgpt.py`. The code here is written by
hand, and it will diverge as components are implemented and swapped.

## Status

| Module                     | State | Document                              |
|----------------------------|-------|---------------------------------------|
| `src/gptlab/autograd.py`   | done  | [docs/AUTOGRAD.md](docs/AUTOGRAD.md)  |

## Layout

```
src/gptlab/     the package, one module for each component
tests/gptlab/   one test file for each module, same name, same structure as src
docs/           one document for each module, same name in upper case
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.14.

```
make install    create the virtualenv, install all dependencies, install git hooks
make validate   format, lint, type-check, and run the tests
make help       list all commands
```
