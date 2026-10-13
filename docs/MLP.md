# MLP: Design Notes

> Code: [`src/gptlab/mlp.py`](../src/gptlab/mlp.py)
> Tests: [`tests/gptlab/test_mlp.py`](../tests/gptlab/test_mlp.py)

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Expand, cut, contract](#2-expand-cut-contract)


## 1. The problem

Attention, in `docs/ATTENTION.md`, is the one operation where positions
read each other. After attention, the vector at position `i` holds a mix of
data from positions `0` to `i`. But attention only moves data between
positions. Inside one position, the output is a weighted sum of value
vectors, and each value is a linear map of an input vector. A sum of linear
maps is a linear map. Attention on its own cannot compute a function that
is not linear in its inputs.

The MLP gives the model that function. It acts on one position at a time.
It has no data about the other positions. It takes the `n_embd` elements of
one vector, mixes them, and returns `n_embd` new elements.

The two blocks divide the work:

| Block     | Mixes across | Reads other positions |
|-----------|--------------|-----------------------|
| Attention | positions    | yes                   |
| MLP       | features     | no                    |

The name is short for **multi-layer perceptron**, the oldest name for a
stack of linear maps with a nonlinear function between them. The module
has one function, `mlp`. It takes one vector and two weight matrices and
returns one vector. The model calls it once per position.

The function has no residual connection and no normalization. Both belong
to the model, which wraps each call, as it does for attention. The model
can then swap this function for a different block without a change to the
code around it.

## 2. Expand, cut, contract

> Code: `mlp`
> Tests: group 1 (shape), group 2 (hand case), group 3 (float reference),
> group 4 (numeric gradient check)

The function computes:

$$
y = \texttt{fc2} \cdot \mathrm{relu}(\texttt{fc1} \cdot x)
$$

The code has three lines, one per stage. Each line makes a vector, and the
name of the vector says what it is.

| Stage    | Line                                           | Length     |
|----------|------------------------------------------------|------------|
| Expand   | `pre_activations = linear(x=x, w=fc1)`         | `n_hidden` |
| Cut      | `hidden = [p.relu() for p in pre_activations]` | `n_hidden` |
| Contract | `outputs = linear(x=hidden, w=fc2)`            | `n_embd`   |

`n_hidden` is the width of the hidden vector. It is not an argument of
`mlp` and the function never reads it. The caller sets it by the number of
rows it gives `fc1`.

### The weights

`fc1` and `fc2` are the two parameters of the block. The name is short for
**fully connected**, the older name for a linear map where every input
element touches every output element. The number says which map comes
first.

`linear(x=x, w=w)`, from section 4 of `docs/LAYERS.md`, returns one dot product
per row of `w`. So the row count of a matrix sets the output length, and
the row length must match the input:

| Matrix | Rows       | Row length | One row is                                      |
|--------|------------|------------|-------------------------------------------------|
| `fc1`  | `n_hidden` | `n_embd`   | one hidden unit, what it reads from `x`         |
| `fc2`  | `n_embd`   | `n_hidden` | one output element, what it reads from `hidden` |

The shapes are not square unless `n_hidden` equals `n_embd`. Attention
maps `n_embd` to `n_embd`, so its four matrices are square. The MLP goes
wide and comes back, so its two matrices are rectangles, and one has the
transpose shape of the other.

### Expand

`linear(x, fc1)` gives `n_hidden` nodes. Each node is the dot product of
`x` with one row of `fc1`. Call it a **pre-activation**: the value of one
hidden unit before the cut.

The model in this lab sets `n_hidden` to $4 \cdot n_{\text{embd}}$. This is the ratio
of the first Transformer and the first GPT models, and later models copied
it. It is a convention, not an outcome of math. The ratio sets the balance
of parameters between the two blocks: attention has $4 n_{\text{embd}}^2$,
and the MLP has $2 n_{\text{hidden}} n_{\text{embd}}$, so at ratio $4$
the MLP holds twice as
many. Measured loss is flat across a wide range of ratios, so 4 is a safe
default and an easy lab swap. A wider hidden vector gives the block more
units to cut, and so more functions it can compute, at the cost of more
parameters to fit on a small dataset.

### Cut

`relu` is $\max(0, v)$. It is a primitive of the autograd engine, section 6
of `docs/AUTOGRAD.md`. The code calls it once per pre-activation, so the
hidden vector has one ReLU node per hidden unit.

The cut is the one nonlinear step. Without it, $\texttt{fc2}\,(\texttt{fc1}\, x)$
is one linear map, $(\texttt{fc2}\,\texttt{fc1})\, x$, and the wide hidden
vector gains nothing.
With it, each hidden unit is either on, with its pre-activation, or off,
at zero. Which units are on depends on `x`. So different inputs reach
`fc2` through different sets of units, and the block computes a different
linear map for each set.

The backward pass follows the same rule. A unit that is on passes the
gradient through unchanged. A unit that is off passes nothing. The
gradient of every weight in the row of `fc1` that feeds an off unit is
zero on that input. The hand case tests check both: the row that feeds the
cut unit receives zero, and the row that feeds an on unit receives `x`.

At exactly zero, `relu` passes no gradient. The forward value is zero
either way, so the choice only matters for the gradient. The point is a
**kink**: the slope is $0$ on one side and $1$ on the other, and the
derivative is not defined at the point itself. A central finite difference
averages the slope on both sides of a point. If a pre-activation is within
$h$ of zero, one side is on and the other is off, and the average matches
neither. The autograd gradient picks one side. So before the gradient check
of group 4 compares anything, it checks that every pre-activation of its
case is more than $10^{-3}$ from zero, a thousand times $h$. The check makes
the seed choice explicit. A future change of the seed then fails on the
check and not on a gradient that looks wrong for no reason.

The block uses ReLU and not GELU. GELU is smooth and is the usual choice in
later models. It would need a new primitive, or a long expression of the
six that exist. ReLU is already a primitive. A different activation is a
later lab swap, and the seam is this one line.

The cut sits between the maps and not after `fc2`. The output of the block
goes into the residual stream of the model, where it is added to `x`. A cut
on the output could only add zero or a positive number to each element,
and the block could never move an element down.

### Contract

`linear(hidden, fc2)` gives `n_embd` nodes. It is the same operation as the
expand with the other matrix. Each output element is a weighted sum of all
`n_hidden` hidden units, with the weights in one row of `fc2`. The model
learns in `fc2` how to combine what the hidden units found.

The output has `n_embd` nodes because the model adds it to `x`. The
function does not require it. A caller with `fc2` of a different row count
gets a different output length, and nothing in `mlp` objects.

### The hand case

The tests use `n_embd = 2` and `n_hidden = 4`, the ratio of the model:

$$
x = \begin{bmatrix} 1 \\ -1 \end{bmatrix},
\qquad
\texttt{fc1} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 1 \\ 1 & -1 \end{bmatrix},
\qquad
\texttt{fc2} = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

Each row of `fc1` is one hidden unit:

| Unit | Reads        | Pre-activation | After the cut |
|------|--------------|----------------|---------------|
| 0    | $x_0$        | $1$            | $1$           |
| 1    | $x_1$        | $-1$           | $0$           |
| 2    | $2x_0 + x_1$ | $1$            | $1$           |
| 3    | $x_0 - x_1$  | $2$            | $2$           |

Output 0 sums every hidden unit: $1 + 0 + 1 + 2 = 4$. Output 1 reads only
unit 1: $0$. The output is $[4, 0]$. Without the cut it is $[3, -1]$. The
second row of `fc2` reads only the cut unit, so a missing cut changes both
elements and the test cannot pass by chance.

### What `mlp` does not do

- No bias term. `linear` has none, section 4 of `docs/LAYERS.md`.
- No shape check. `dot` uses a strict `zip`, so a row of `fc1` with the
  wrong length, or a row of `fc2` that does not match the hidden vector,
  raises `ValueError` inside `linear`. 
- No residual, no normalization, no loop over positions. Section 1 says
  where they are.
