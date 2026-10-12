# CLAUDE.md

Instructions for Claude Code in this repository. Read this file before any
work. The user edits it to say exactly how they want the work done.

## 1. What this repository is

A GPT written from scratch in pure Python. No PyTorch, no NumPy, no
dependencies. Every number in the model is a scalar `Node` with its own
gradient. The inspiration is Karpathy's `microgpt.py`. The code is written
by hand and diverges from it.

The purpose is learning. The user wants a deep grasp of every line. The code
is the record of that learning, not the product. When the base model works,
the lab part starts: swap one transformer component at a time (RoPE for
learned positions, Muon for Adam, GELU for ReLU) and compare. The goal is for 
any user to be able to come in and learn about GPT components with minimal 
background knowledge.

The package is `gptlab`, the distribution is `gpt-lab`. Python 3.14, `uv`
for everything.

## 2. How to work with the user

The user writes the code. Claude does not.

- Do not write an implementation unless the user asks for one by name.
  Writing a throwaway reference to verify a test file is fine, as long as
  it never lands in `src/`.
- Prefer questions over answers. Point out a gap, ask a question, confirm
  or correct the user's reasoning. Offer a hint before a solution.
- When the user says they are lost, drop the questions for a moment. Give a
  numbered, step-by-step explanation in Simplified Technical English. Then
  return to questions.
- If the user prefers a way of explaining something or specifies some 
  information that was helpful, include the information in the documentation.
- Correct a wrong statement plainly. Do not soften a wrong answer into a
  partly right one.
- Audit, review, and write. Those are Claude's jobs: outlines, tests,
  audits, docstrings, and documents.

## 3. The process for one module

Each module has a GitHub issue. The issues are #1 to #9, one per module,
in build order: tokenizer, layers, loss, attention, mlp, model, optim,
train, sample. Each issue has Scope, Decisions, Tests, Doc sections, and
Depends on. Read the issue first. The plan lives in the issues, not in the
repo, so nothing in the repo can go stale.

The steps, in order:

| Step | Who    | Work                                                                                                                                                                                                                                       |
|------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Claude | Write `docs/<MODULE>.md` as a work outline: what the module does, the work steps, the function with a shape table, implementation steps per part, the test groups, the planned final sections, a decisions log, and open questions.        |
| 2    | Claude | Write `tests/gptlab/test_<module>.py` and a one-line docstring stub for `src/gptlab/<module>.py` so pytest collects. Verify the tests against a throwaway reference that never lands in `src/`. Then confirm every test fails on the stub. |
| 3    | User   | Implement against the tests until they pass (TDD).                                                                                                                                                                                         |
| 4    | Both   | Discuss. Claude adds each decision to the decisions log in the outline as it is made, with the reason.                                                                                                                                     |
| 5    | Claude | Audit the code against the issue and the conventions of the existing modules. Findings most important first, with a pointer to the sibling code to compare. Record the audit in the outline.                                               |
| 6    | User   | Refactor from the audit.                                                                                                                                                                                                                   |
| 7    | Claude | Write all docstrings, Google style. Leave the body unchanged.                                                                                                                                                                              |
| 8    | Claude | Rewrite `docs/<MODULE>.md` in the format of the existing documents, in Simplified Technical English. Fold the decisions log into the prose as the reason for each choice.                                                                  |
| 9    | Claude | Add the row to `docs/README.md`. Run the full checks. Report that the module is ready to commit.                                                                                                                                           |
| 10   | User   | Commit the module, tests, and docs and push. See the format below.                                                                                                                                                                         |
| 11   | Claude | After the user says the commit is in, close the issue with a comment that names the commits. See the format below.                                                                                                                         |

The implementation steps in the outline (step 1) break the function into
parts. Each part has numbered steps, a column that says which tests the
part makes pass, and questions to settle before writing it. When the user
asks for more detail on a part, add what they hold at that point, what the
part must do to it, a comparison with a part they already wrote, and a
table of likely failures with their causes.

Keep the work-steps table in the outline current: mark a step done when
it is done.

### Committing (step 10)

The user commits and pushes. Claude never runs `git commit` or `git push`,
not even when a step table says the work is done. Claude reports that the
work is ready and stops.

```
<module> implementation, tests, and docs
```

A change that is not a module landing (a convention change across files,
a CLAUDE.md edit) gets its own commit with a subject that says what
changed, as `import functions by name instead of module` does.

### Closing the issue (step 11)

Close the issue after the user says the commit of step 10 is in. Check
`git status` is clean and `git log --oneline` shows the commit first. Close with
`gh issue close <N> --comment "..."`. The comment is one line in this form:

```
Done in <hash>: <module>.py, test_<module>.py (<count> tests), docs/<MODULE>.md.
```

- `<hash>` is the short hash of the commit that landed the module, tests,
  and doc. Take it from `git log --oneline`, never from memory.
- `<count>` is the number of tests in the file. Take it from
  `uv run pytest tests/gptlab/test_<module>.py -q`.
- If later commits changed the module before the close, add one sentence:
  "Follow-ups in <hash> (<what changed>), <hash> (<what changed>)."
- No other text. The doc and the commit message hold the detail.

Example, from issue #5:

```
Done in 9c28642: mlp.py, test_mlp.py (11 tests), docs/MLP.md. Follow-ups in 9313b62 (tests import functions by name) and 9062153 (CLAUDE.md with the module process).
```

## 4. Code conventions

Conventions to follow through out the code. 

- Vectors and matrices are plain lists: `Vector = list[Node]`,
  `Matrix = list[Vector]`, both aliases in `layers.py`. No helper class.
- No new autograd primitive. There are six: add, mul, pow, exp, log, relu.
  Square root is `** 0.5`. Division is `* ... ** -1`. Embedding lookup is
  list indexing.
- A layer function takes lists of nodes and returns nodes. The scalar
  operations make every output node, so the backward pass needs nothing
  new. List operations (slice, transpose, extend) make no node.
- Elementwise maps are list comprehensions. A sum is a loop seeded from the
  first element, not `sum()`, so that no float zero enters the graph.
- Local names are one short noun for what the vector is: `scores`,
  `weights`, `squares`, `pre_activations`, `outputs`. The loop variable is
  the singular of the list: `exponential` in `exponentials`. Not the type. 
- Every local variable has a type hint.
- Every parameter is keyword-only. A bare `*` opens each parameter list:
  `def mlp(*, x: Vector, fc1: Matrix, fc2: Matrix)`. So every call names
  each argument: `mlp(x=x, fc1=fc1, fc2=fc2)`, `Node(forward_val=3.0)`,
  `_nodes(values=X_VALS)` in a test. No `*` unpacking into a call. This
  holds for the `gptlab` functions, `Node.__init__`, the test helpers, and
  the test functions. Ruff rule `PLR0917` with `max-positional-args = 0`
  in `pyproject.toml` fails any definition that forgets the `*`. The
  exceptions are the calls Python makes positionally: builtins and library
  functions (`zip`, `len`, `range`, `math.exp`, `pytest.approx`), the
  operator methods (`__add__`), a closure that Python calls through an
  attribute (`propagate(g)`), and a `*args` parameter (`_seeded(a, b)`).
  `autograd.py` is excluded from the rule for the operator methods and
  the closures, but `Node.__init__` still has the `*`. A closure in a test
  that stands in for `propagate` carries `# noqa: PLR0917`.
- No shape check where a strict `zip` already raises `ValueError`. A guard
  is added only where the error message can say more than the free error
  can, as the `n_head` guard in `attention` does, and as the `is empty`
  guard does in every function that reads element zero of an input.
- A plain float that does not need a gradient stays a float (the softmax
  shift, the attention scale).
- The residual connection and normalization live in `model`, not in the
  block functions. Each block is a pure function of its arguments so the
  model can swap easily.

### Docstrings

Google style. Written by Claude in step 7.

- Module docstring: what the function does in two or three sentences, that
  the backward pass needs nothing new if that is true, and the line "The
  design notes and the reason for each choice are in docs/<MODULE>.md."
- Function docstring: a one-line summary, often with the formula. A short
  paragraph on what it does and does not do. `Args` with shapes, `Returns`,
  and `Raises`.
- Shapes in docstrings are general (`n_in`, `n_hidden`, `n_out`), not the
  values the model happens to use. The doc says what the model uses.
- Code names in prose use double backticks.

## 5. Tests

One test file per module, same name, same structure as `src/`.

- Module docstring says that each test group maps to a section of the doc,
  and that the functions are imported by name, so the import fails with
  one collection error until they exist.
- Import the functions by name: `from gptlab.mlp import mlp`, then
  `mlp(...)`. Not the module. `mlp.mlp` and `attention.attention` read
  badly because module and function share a name.
- Helpers `_nodes`, `_matrix`, `_values`, `_random_floats`,
  `_random_matrix`.
- A float reference, written with floats and `math` only, that shares no
  code with the module. The forward tests compare against it where a hand
  computation is too long. The gradient check perturbs it.
- A shared small case with fixed seeds and module-level constants.
- Test groups with a header comment:
  `# --- Group N: name (Section M) ----`, padded to the line width. The
  comment can explain the case in a few lines below the header.
- Each module has a hand case that a reader can verify on paper, chosen
  so that a plausible wrong implementation gives a different answer.
- Each module ends with a numeric gradient check: central finite
  difference, `H = 1e-6`, `TOL = 1e-6`, a weighted loss with fixed
  distinct coefficients so no gradient is zero by construction, every
  input float perturbed. Guard against known kinks (ReLU at zero) with an
  explicit assertion before the comparison.
- Test names are sentences: `test_a_later_position_does_not_change_earlier_outputs`.

## 6. Documents

One document per module in a flat `docs/`, named after the module in upper
case. `docs/README.md` is the index in reading order and holds the shared
"How to read a document" section.

- Nothing in the repo describes code that does not exist. No stubs, no
  "coming soon", no plan sections. The plan is in the issues.
- Sections follow the order of the code. Each numbered section opens with
  a `> Code:` line (names, never line numbers) and a `> Tests:` line (test
  group). Section 1 is "The problem" and has no `Code:` line. The last
  section is "Check the gradients with numbers", with `Code: none`.
- Cross-references name the document and the section:
  "section 4 of `docs/LAYERS.md`". A section number alone points inside
  the current document. No forward references.
- Verify every cross-reference against the headings before writing it.
- Each design choice gets its reason in the prose. The alternative that
  was not chosen is named and the reason it lost is given.
- Tables for shapes, for comparisons of two things, and for cases. Follow
  existing table formatting. 
- When a new document lands, add its row to the table in `docs/README.md`.

### Writing style

Simplified Technical English, ASD-STE100 style, for all documents and for
explanations when the user is lost.

- Short sentences. One idea per sentence. About 20 words at most.
- Active voice. "The function returns" not "is returned by".
- One word for one thing. Pick a term and keep it: "position", "element",
  "node", "vector", "matrix", "pre-activation", "hidden unit".
- Define a word in bold the first time, then use it without decoration.
- Numbered steps for a procedure. A table for a comparison.
- No idioms, no metaphors, no hedging. State the fact.
- Present tense. "The code calls it once per pre-activation."
- Code names in backticks. Formulas in fenced blocks.

## 7. Tooling

```
make install     create the venv, install deps, install git hooks
make validate    format, lint, type-check, test
uv run pytest tests/gptlab/test_<module>.py -q
uv run ruff check src tests && uv run ruff format --check src tests
uv run ty check src
```

Lefthook runs format, lint, and typecheck on commit, and tests on push.
Run the full checks before reporting a step done.
