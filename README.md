# gpt-lab

![Python 3.14](https://img.shields.io/badge/python-3.14-blue?logo=python&logoColor=white)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--checked-ty-261230)](https://github.com/astral-sh/ty)

[![pytest](https://img.shields.io/badge/tested_with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![lefthook](https://img.shields.io/badge/hooks-lefthook-FF1E1E)](https://github.com/evilmartians/lefthook)

![Pure Python](https://img.shields.io/badge/pure-python-3776AB)
![No dependencies](https://img.shields.io/badge/dependencies-none-success)

[![Claude Code](https://img.shields.io/badge/built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)

A GPT written from scratch in pure Python. No PyTorch, no NumPy. Each number
in the model is a scalar node with its own gradient.

The purpose is to learn. The code is small enough to read every line. When
the base model works, the lab part starts: swap one transformer component at
a time (for example, RoPE for learned positions, or Muon for AdamW) and
compare the results.

The inspiration is Karpathy's `microgpt.py`. The code here is written by
hand, and it will diverge as components are implemented and swapped.

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
