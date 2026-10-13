# Attention: Design Notes

> Code: [`src/gptlab/attention.py`](../src/gptlab/attention.py)
> Tests: [`tests/gptlab/test_attention.py`](../tests/gptlab/test_attention.py)

---

## Contents

- [1. The problem](#1-the-problem)
- [2. One query reads many positions](#2-one-query-reads-many-positions)
- [3. Heads](#3-heads)
- [4. Queries, keys, values](#4-queries-keys-values)
- [5. The causal mask is a slice bound](#5-the-causal-mask-is-a-slice-bound)
- [6. Join and output projection](#6-join-and-output-projection)


## 1. The problem

The model reads a sequence of tokens and predicts the next one. Before
attention, each position holds one vector. That vector has no data about
the other positions. The embedding of position 3 is the same when position
2 holds "a" and when it holds "z".

Attention is the one operation in the model where positions read each other.
Every other operation acts on one position at a time. After attention, the
vector at position `i` holds a mix of data from positions `0` to `i`.

Two rules apply:

- Position `i` can read positions `0` to `i`. It cannot read positions after
  `i`. The model predicts the next token, so the next token must stay
  hidden. This is the **causal** rule.
- Position `i` decides for itself how much to read from each earlier
  position. The weights are not fixed. They come from the vectors, so they
  change with the content. This is what the word "attention" names.

The module has three functions, in this order:

- `attend` is the core. One query reads a list of keys and values and
  returns one vector. It has no data about positions, heads, or the causal
  rule.
- `split_heads` cuts a matrix into one matrix per head.
- `attention` takes the whole sequence and returns the whole sequence. It
  projects, splits, calls `attend` once per head per position, joins, and
  projects again. The causal rule is here, as a slice bound on the keys
  and values that `attend` receives.

The full name of `attention` is **causal multi-head self-attention**:

- **Self**: the queries, keys, and values all come from the same sequence.
- **Multi-head**: the vector is split into `n_head` groups, and each group
  reads on its own. Section 3.
- **Causal**: position `i` reads positions `0` to `i` only. Section 5.

This is the original form, as in the first GPT models. Later labs can swap
the projections of section 4 for a variant.

The function has no residual connection and no normalization. Both belong
to the model, which wraps each call. The model also adds the position
embeddings before it calls this function. The model can then swap this
function for a different attention without a change to the code around it.

## 2. One query reads many positions

> Code: `attend`
> Tests: group 1 (attend)

`attend` takes one **query** vector, a list of **key** vectors, a list of
**value** vectors, and a float `scale`. Key `j` and value `j` belong to the
same position. The function returns one vector with the length of a value.

The three words name three roles:

- The query says what the reader looks for.
- A key says what one position offers.
- A value is what the reader receives if it reads that position.

The function scores the query against each key, turns the scores into
weights, and returns the weighted sum of the values. The code has three
lines, one per stage.

### The score

The score of key $j$ is:

$$
\mathrm{score}_j = (q \cdot \mathrm{keys}_j) \cdot \mathrm{scale}
$$

The dot product $q \cdot \mathrm{keys}_j$ is `dot`, the function of section 3
of `docs/LAYERS.md`.

### Why the scale

A dot product of two long vectors is a large number. Its usual size grows
with the square root of the vector length, `head_dim`. `softmax` of large
scores gives one weight near one and all other weights near zero, and its
derivative is then near zero. Training would stop before the model learns
what to attend to.

Multiply each score by $\texttt{head\_dim}^{-1/2}$, which is
$1 / \sqrt{\texttt{head\_dim}}$.
The scores return to the size where `softmax` has a useful gradient.

`attend` does not compute the scale. It does not know `head_dim`, and the
caller does. Section 4 says where the float comes from.

### The weights

`softmax(logits=scores)` turns the $n$ scores into $n$ weights. Each weight is
between zero and one, and the weights sum to one. Weight `j` is how much the
query reads from position `j`.

With one key, `softmax` of one score is exactly one, for any value of the
score. The output is then the one value.

### The weighted sum

The output has `head_dim` elements. Element $t$ is:

$$
\mathrm{out}_t = \mathrm{weights}_0 \cdot \mathrm{values}_{0,t} + \mathrm{weights}_1 \cdot \mathrm{values}_{1,t} + \cdots
$$

This is a weighted sum across **positions**, one for each element.
`linear` sums across the elements of one row. The rows of `values` are
positions, so they point in the wrong direction. `transpose(m=values)` turns
them, and `linear(x=weights, w=transpose(m=values))` is then the sum above.
Section 7 of `docs/LAYERS.md` gives the same argument from the other side.

### What `attend` does not know

`attend` has no position index, no head index, and no rule about "earlier".
It reads every key that it receives. The caller decides which keys and
values to pass. That decision is the causal rule of section 5. The rule is
outside `attend` so that the tests can check the core on its own, with any
keys.

## 3. Heads

> Code: `split_heads`
> Tests: group 2 (split_heads)

One score per pair of positions is one number. One number can say how much
position `i` reads from position `j`, but not for what reason. A head
splits the `n_embd` elements into `n_head` groups of `head_dim`. Each group
scores and reads on its own, so each group can learn a different reason.

`split_heads` takes a matrix with one row per position. It returns a list
of `n_head` matrices. Matrix $h$ holds elements $h \cdot \texttt{head\_dim}$ to
$(h + 1) \cdot \texttt{head\_dim} - 1$ of every row. With `n_embd = 4` and
`n_head = 2`:

| head $h$ | elements |
|----------|----------|
| 0        | 0, 1     |
| 1        | 2, 3     |

The index order of the result is head first, then position. `q[h][i]` is
the query of position `i` in head `h`. `k[h][: i + 1]` is the keys of
positions `0` to `i` in head `h`.

The list operations in this module make no node. A slice, a transpose, and
an `extend` each move node references. A gradient that reaches a node in a
slice reaches the same node in the row that the slice came from.

Head 0 never scores against an element of head 1, and never reads one. The
heads are independent until section 6 joins them.

`split_heads` does not check that `n_head` divides the row length. That
guard is in `attention`, where the error message can give both numbers.
`split_heads` does check that `m` has a row, because `head_dim` reads the
length of `m[0]`. An empty `m` raises `ValueError` with the message
`m is empty`, as `dot` does for an empty vector in section 3 of
`docs/LAYERS.md`.

## 4. Queries, keys, values

> Code: `n_embd`, the two `ValueError` guards, `head_dim`, `scale`, and the
> three lines that project and split
> Tests: group 3 (shape)

`attention` reads `n_embd` from the first input vector. It is not an
argument. The weight matrices are `n_embd x n_embd`, so the model sets the
size when it makes them.

`linear` projects each position three times:

- `linear(x=x[i], w=wq)` is the query of position `i`.
- `linear(x=x[j], w=wk)` is the key of position `j`.
- `linear(x=x[j], w=wv)` is the value of position `j`.

Three weight matrices let the model learn a query that is not the same as
the key of the same position. One matrix would give the same vector three
roles.

Each projection is a comprehension over every position. The result goes
directly into `split_heads`. Thus, `q`, `k`, and `v` are lists of matrices,
head first, then position, as section 3 describes.

### The guards

`x` must have a position, because `n_embd` reads the length of `x[0]`. An
empty `x` raises `ValueError` with the message `x is empty`. Without the
guard, the error would be `IndexError` with no name in it.

`n_head` must divide `n_embd`. If it does not, `n_embd // n_head` rounds
down. The last `n_embd % n_head` elements of every vector would then belong
to no head. The forward pass would run, and the output would have the wrong
size. The function raises `ValueError` instead. A wrong shape is an error,
not a wrong number.

### The scale

`scale` is $\texttt{head\_dim}^{-1/2}$. It is a plain float. The code computes it
one time, before the loops, and passes it to every call of `attend`.
Section 2 says why.

## 5. The causal mask is a slice bound

> Code: the loop over `i` and `h`, `k[h][: i + 1]`, `v[h][: i + 1]`, and
> the call of `attend`
> Tests: group 5 (causality)

For position `i` and head `h`, the code calls:

```
attend(q=q[h][i], keys=k[h][: i + 1], values=v[h][: i + 1], scale=scale)
```

The slice `[: i + 1]` holds positions `0` to `i`. At `i = 0` it holds one
row. Position `i` reads itself and every position before it. It never
reads a later position, because `attend` never sees one.

A common alternative computes every score for every pair. It then sets the
scores for $j > i$ to minus infinity before `softmax`. `exp` of minus
infinity is zero, so those positions receive a weight of zero. The result
is the same.

The graph is not the same. That method makes $T^2$ score nodes per head.
Half of them carry a weight of zero into every backward pass. The slice
bound makes $T(T + 1)/2$ score nodes and no masked node at all.

The causality tests check this by change. Change the input at position 2.
The outputs at positions 0 and 1 are the same floats, not only near.
Nothing in their graphs reads position 2, so nothing can move. Change
position 0, and every output changes, because every position reads
position 0.

## 6. Join and output projection

> Code: `joined.extend(...)`, `linear(x=joined, w=wo)`
> Tests: group 3 (shape) and group 4 (one position)

### The join

`joined.extend(...)` joins the outputs of the heads. After the loop over
`h`, `joined` holds `n_embd` nodes: the `head_dim` outputs of head 0, then
of head 1, and so on.

The code uses `extend` and not `+=`. On lists, `+=` is also a join. On
nodes, `+=` makes a sum node. The two look the same in code and do
different things. `extend` can only mean the list operation.

### The output projection

The heads are independent. No node in `joined` mixes head 0 with head 1.
`linear(x=joined, w=wo)` mixes them. Each output element is a weighted sum of
all `n_embd` elements of `joined`, across every head. The model learns in
`wo` how to combine what the heads found. In the backward pass, `wo` is
also where the gradient from the next layer splits and flows back into
every head.

There are two mixes in this module, in two directions:

| Mix                           | Across                                | Weights come from                                                     |
|-------------------------------|---------------------------------------|-----------------------------------------------------------------------|
| The weighted sum of section 2 | positions `0` to `i`, inside one head | `softmax` of the scores. Not a parameter. They change with the input. |
| The output projection         | heads, inside one position            | `wo`. A parameter. The same for every position.                       |

Only the second mix is a weight matrix that the model learns. The code
computes the first mix from the content on every call.

### One position

With one position, every call of `attend` receives one key, and section 2
says that the output is then the one value. The output of `attention` is
$\texttt{wo}\,(\texttt{wv}\, x_0)$, and `wq` and `wk` have no effect. A position with
nothing to choose between has no use for a query or a key.
