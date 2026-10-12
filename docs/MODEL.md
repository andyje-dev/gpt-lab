# Model: Design Notes

> Code: [`src/gptlab/model.py`](../src/gptlab/model.py)
> Tests: [`tests/gptlab/test_model.py`](../tests/gptlab/test_model.py)

---

## Contents

- [1. The problem](#1-the-problem)
- [2. The shapes](#2-the-shapes)
- [3. Which nodes are parameters](#3-which-nodes-are-parameters)
- [4. Embeddings and the lookup question](#4-embeddings-and-the-lookup-question)
- [5. The residual stream](#5-the-residual-stream)
- [6. The forward pass, step by step](#6-the-forward-pass-step-by-step)


## 1. The problem

The four modules before this one are pure functions. `linear`, `softmax`,
and `rmsnorm` in `docs/LAYERS.md`, `attention` in `docs/ATTENTION.md`,
and `mlp` in `docs/MLP.md` each take nodes and weights and return nodes.
None of them owns a weight. None of them knows how many positions the
model reads, how many times a block repeats, or where one output goes
next.

The model is the module that owns the weights and the order. It has two
jobs:

1. Make every **parameter**, a weight node that training moves, one time
   at construction. Put all of them in one flat list for the optimizer.
2. Run the **forward pass**: take a list of token ids and return one
   logits vector per position, as `docs/LOSS.md` defines logits.

The forward pass is the chain of what exists:

```
x[i]  = wte[token[i]] + wpe[i]          one vector per position
x     = rmsnorm(x)                      per position
for each block:
    x = x + attention(rmsnorm(x))       whole sequence, one call
    x = x + mlp(rmsnorm(x))             per position
x     = rmsnorm(x)                      per position
logit = linear(x, lm_head)              per position
```

Two words that this document keeps apart:

| Word  | Means                                                        | Where                              |
|-------|--------------------------------------------------------------|------------------------------------|
| layer | one function of `layers.py`: `linear`, `softmax`, `rmsnorm`  | `docs/LAYERS.md`                   |
| block | one attention call plus one MLP call, with their six tables  | `Block`, `blocks`, `n_blocks` here |

A **block** has two halves. The attention half mixes across positions.
The MLP half mixes across features of one position. Section 1 of
`docs/MLP.md` gives the division of work. The model stacks `n_blocks`
of them.

The module has three classes:

| Class    | Holds                                          | Computes         |
|----------|------------------------------------------------|------------------|
| `Config` | the five sizes                                 | nothing          |
| `Block`  | the six tables of one block                    | nothing          |
| `GPT`    | the tables, the flat list, and the config      | `forward`        |

## 2. The shapes

> Code: `Config`, `Block`, `INIT_STD`, `MLP_RATIO`, `GPT.__init__`
> Tests: group 1 (config), group 2 (init)

### The config

`Config` is a frozen, keyword-only dataclass with five integers:

| Field            | Sets                                                                  |
|------------------|-----------------------------------------------------------------------|
| `vocab_size`     | the number of tokens the model knows. Rows of `wte` and `lm_head`.    |
| `n_embd`         | the width of every vector in the residual stream                      |
| `n_blocks`       | how many blocks the forward pass runs                                 |
| `n_head`         | the number of attention heads. It must divide `n_embd`.               |
| `context_length` | the most tokens one call accepts. Rows of `wpe`.                      |

`frozen=True` makes the fields read-only. `kw_only=True` makes every
call name each field, the convention of every function in this project.
A dataclass fits because five values go in and the same five come out,
with nothing computed.

The one check is in `__post_init__`, the hook a dataclass calls after
its own `__init__`. It raises if `n_head` does not divide `n_embd`,
with both numbers in the message. `attention` has the same guard, but
it fires on the first forward call. This one fires at construction,
before any parameter is made, which is where a wrong number is cheapest
to find. The guard in `attention` stays for callers that use that
function alone.

The config says what the model is. Two models with the same config have
the same tables, the same parameter count, and the same forward pass.
The seed is not a field. It says which draw of the model you got, and
"same config, five seeds" is one experiment with one config.

**Context length** is the window of tokens the model sees at one time.
The common name in GPT code is `block_size`. This project uses
`context_length` so that "block" has one meaning in the file.

### The tables

Every parameter lives in one of nine kinds of table. A table is a
`Matrix` from section 2 of `docs/LAYERS.md`: a list of rows, each row a
list of nodes. `n_hidden` is `MLP_RATIO * n_embd`, with `MLP_RATIO = 4`,
the ratio of section 2 of `docs/MLP.md`. The model sets it by the row
count of `fc1`. It is not a config field, because no caller passes a
different value and a later lab swap changes the constant in one place.

| Table     | Name                     | Rows             | Row length | One row is                                | Used by                   |
|-----------|--------------------------|------------------|------------|-------------------------------------------|---------------------------|
| `wte`     | token embeddings         | `vocab_size`     | `n_embd`   | the embedding of one token                | the lookup, by token id   |
| `wpe`     | position embeddings      | `context_length` | `n_embd`   | the embedding of one position             | the lookup, by position   |
| `wq`      | query weights            | `n_embd`         | `n_embd`   | one query element                         | `attention`, per block    |
| `wk`      | key weights              | `n_embd`         | `n_embd`   | one key element                           | `attention`, per block    |
| `wv`      | value weights            | `n_embd`         | `n_embd`   | one value element                         | `attention`, per block    |
| `wo`      | output weights           | `n_embd`         | `n_embd`   | one output element                        | `attention`, per block    |
| `fc1`     | first fully connected    | `n_hidden`       | `n_embd`   | one hidden unit                           | `mlp`, per block          |
| `fc2`     | second fully connected   | `n_embd`         | `n_hidden` | one output element                        | `mlp`, per block          |
| `lm_head` | language model head      | `vocab_size`     | `n_embd`   | one logit, what it reads from the vector  | `linear`, per position    |

The names behind the short forms: `w` is weight, `te` and `pe` are
token and position embedding, `q k v o` are query, key, value, and
output as section 4 of `docs/ATTENTION.md` defines them, `fc` is fully
connected as section 2 of `docs/MLP.md` defines it, and `lm` is
language model. The **head** is the last map, from the stream to the
output.

The rule for every shape: `linear` returns one dot product per row, so
the row count is the output length and the row length is the input
length, section 4 of `docs/LAYERS.md`. `lm_head` has `vocab_size` rows
because it makes `vocab_size` logits, and each row has `n_embd` elements
because it reads one vector of the stream. The test case uses
`vocab_size = 4` and `n_embd = 6`, two different numbers, so that a table
made with its two sizes swapped has the wrong shape. A square would hide
the swap.

`wte` and `lm_head` have the same shape and are not the same table. The
first maps a token to a vector. The second maps a vector to a score per
token. Some models share one table for both. That is a later lab swap.

The six tables of one block sit in a `Block`, a frozen, keyword-only
dataclass with six `Matrix` fields. `attention` and `mlp` take six named
matrices, so `forward` passes `block.wq` and the others straight
through. A dict by name was the other choice. Its keys are strings that
the type checker cannot see, and `block["wq"]` reads worse than
`block.wq`.

### Challenge: count the parameters

Write the parameter count as a formula in the five config sizes, from
the rows and row lengths in the table. Then check it on the shared test
case, `vocab_size = 4`, `n_embd = 6`, `n_blocks = 2`, `n_head = 2`,
`context_length = 4`. The test `test_params_count_matches_the_formula`
holds the number.

Two hints:

- Two tables have the same shape and the same count.
- Each block has six tables. Four are squares. The other two are
  rectangles with the same count as each other.

<details>
<summary>Answer</summary>

```
count = 2 * vocab_size * n_embd
      + context_length * n_embd
      + n_blocks * 12 * n_embd ** 2
```

The `2` is `wte` and `lm_head`. The `12` is four squares of `n_embd ** 2`
for attention and `4 + 4` for the two rectangles of the MLP, each
`n_hidden * n_embd` with `n_hidden = 4 * n_embd`. `n_head` does not
appear: the heads are slices of the same four tables, section 3 of
`docs/ATTENTION.md`. The test case has `48 + 24 + 864 = 936`.

</details>

### The init

Every node is a leaf: `Node(forward_val=...)` with no children, section
2 of `docs/AUTOGRAD.md`. Its value is one draw from a Gaussian with mean
`0` and standard deviation `INIT_STD = 0.08`. One `random.Random(seed)`
makes every draw, in the order of the table above, row by row. So the
same seed gives the same parameters, and a different seed gives
different ones. The generator is a local of `__init__`, not an
attribute, because nothing reads it after construction.

Two nested functions make the tables. `random_vector` draws one row of
`cols` nodes. `random_matrix` calls it `rows` times. Both close over the
generator, so no caller threads it through. The draw is inside the
comprehension, once per node. A draw outside it would give every node of
a row the same value.

Why `0.08`. The norms remove the scale of `wte` and `wpe` before any
block reads them, so their spread only sets a direction. Inside a block,
`linear` of a vector with root mean square `1` through weights of spread
`0.08` gives outputs with spread about `0.08 * sqrt(n_embd)`: `0.32` for
`n_embd = 16`. Each block adds about a third of the stream's scale, and
the logits start with the same spread. A softmax over logits that close
to each other is near uniform, so the first loss is near `log(vocab_size)`,
the value section 2 of `docs/LOSS.md` gives for uniform logits. A larger
spread makes confident wrong guesses at step zero and a larger first
loss. The value is the one the reference `microgpt.py` uses. The init
spread is a lab swap, and the constant has a comment that says so.

## 3. Which nodes are parameters

> Code: `GPT.params`
> Tests: group 3 (params), group 6 (graph)

One forward pass of the test case makes tens of thousands of nodes. Only
936 of them are parameters. The rest are intermediate results: sums,
products, exponentials, each made fresh by the forward pass and dropped
after the backward pass. An update to an intermediate node has no
effect, because the next forward pass makes it again from the
parameters. The optimizer must know which nodes are parameters, and the
graph alone does not say.

The model answers with a **flat parameter list**, `params`, made one
time at construction. It holds every node of every table, in this order:

1. `wte`, row by row.
2. `wpe`, row by row.
3. For each block in order: `wq`, `wk`, `wv`, `wo`, `fc1`, `fc2`, each
   row by row.
4. `lm_head`, row by row.

The code builds a list of the tables in that order and flattens it with
one comprehension of three `for` clauses: for each table, for each row,
for each node. The optimizer walks this list. It does not walk the
graph. A graph walk would have to start from a loss, visit every
intermediate node, and ask each one whether it is a parameter. The list
is shorter, it exists before any loss does, and it fixes an order. The
order is what a later save-and-load needs: write the values in list
order, read them back in list order.

The list holds the same node objects as the tables, not copies. The
tests check it by identity: `params[0] is wte[0][0]`, and the last item
is the last node of `lm_head`. If the list held copies, the optimizer
would move nodes that the forward pass never reads, and the loss would
never change.

Every parameter is below the loss when the input uses every token at
every position. The test builds that input, `[0, 1, 2, 3]` with
`vocab_size = 4` and `context_length = 4`, walks the children from the
loss with its own helper, and checks that each of the 936 nodes is in
the set. Section 3 of `docs/AUTOGRAD.md` defines the walk. A parameter
that is not below the loss gets no gradient and never moves. The next
section says which parameters that happens to.

## 4. Embeddings and the lookup question

> Code: the first comprehension in `GPT.forward`
> Tests: group 6 (graph)

A **token** is an integer id, section 1 of `docs/LOSS.md`. It is not a
node and has no gradient slot. The model uses it to **index** into
`wte`: `wte[token]` is one row, a list of `n_embd` leaf nodes. The
position `i` indexes `wpe` the same way. The embedding of a position is
the elementwise sum of the two rows:

```python
[te + pe for te, pe in zip(self.wte[token], self.wpe[i], strict=True)]
```

`te` and `pe` are one element of the token row and one of the position
row. Each `+` makes one node with two children, the two leaves. The
lookup itself makes nothing. It selects a row that exists.

The old question was whether the selection needs an autograd operation
of its own, with a local derivative. It does not. The gradient of a
selected row has to be the gradient that arrives at the vector built
from it. The `+` node already does that: its local derivative is `1`
to each child, section 6.2 of `docs/AUTOGRAD.md`. So the gradient of the
embedding flows into `wte[token]` and `wpe[i]` with no change, through
an operation that exists. A selection node would pass the same gradient
with the same factor of `1` and add one object per element for nothing.

Three facts follow, and group 6 tests each one:

| Fact                                                     | Test                                                   |
|----------------------------------------------------------|--------------------------------------------------------|
| A `wte` row that no token selects is not in the graph    | `test_an_unused_wpe_row_is_not_below_the_loss`, for `wpe` |
| Its gradient stays at zero after `backward`              | `test_the_wte_row_of_an_unused_token_gets_gradient_zero` |
| A row selected twice gets the sum of both gradients      | `test_a_token_used_twice_sums_both_gradients_into_one_row` |

The first fact is the one that makes the second true. `backward` walks
the graph below the loss, section 9 of `docs/AUTOGRAD.md`. A row that no
add reads is not below the loss, so nothing writes to its gradient slot,
and the slot keeps the zero it was made with. Plain gradient descent
then leaves the row where it is. An optimizer that keeps a running
average of past gradients can still move it, because the average is not
zero. That is a property of the optimizer, not of this module.

The third fact is accumulation, section 5 of `docs/AUTOGRAD.md`. The row
of token `1` in `[1, 0, 1]` feeds two adds. Each add sends its gradient
to the same leaves, and `+=` sums them. The test checks the sum against a
finite difference of the float reference.

## 5. The residual stream

> Code: the `for block in self.blocks` loop in `GPT.forward`
> Tests: group 8 (causality)

The **residual stream** is the matrix of one vector per position that
runs from the embedding to the head. Each half of each block reads it,
computes something, and adds the result back:

```
residual_stream = residual_stream + half(rmsnorm(residual_stream))
```

The stream is never replaced. It is added to. The name in the code is
`residual_stream`. Section 1 of `docs/MLP.md` calls it `x` when it says
where the output of `mlp` goes.

### Why add and not replace

With a replace, `x = half(x)`, the gradient from the loss has to pass
through every half on its way back to the embedding. Each half is a
chain of products and a softmax or a ReLU, and each one scales the
gradient. Over several blocks the product of the scales goes toward
zero or away from it.

With an add, the `+` node sends the gradient to both children with a
factor of `1`. So there is one path from the loss to the embedding that
passes through nothing but adds, and the gradient on that path arrives
with no change. Each half adds its own gradient to that path. A block
that has learned nothing yet is close to the identity, and the model
starts as a working, if useless, function.

### Why the norm is on the input of the half

The code norms into a second matrix, `normed`, hands that to the half,
and adds the result to the raw stream. The raw stream is read twice:
once for the norm and once for the add. The norm result is read once.

The other order, norm after the add, puts `rmsnorm` on the one path that
was free of everything else. It also lets the scale of the stream grow
with each add, because nothing between the adds brings it back. With
the norm on the input, each half sees a vector of root mean square `1`,
whatever the stream has grown to, and the stream itself is left alone.
The norm has no learned scale, section 6 of `docs/LAYERS.md`, because
each one is followed by a `linear` that can learn the same scale.

There is one more norm, right after the embedding and before the first
block. The first block has its own norm on its input, so this one does
not change what the first half sees. It changes what the first add
reads: a stream of root mean square `1` instead of the raw sum of two
rows of spread `0.08`. Without it, the first block's output, with
spread about `0.32` from section 2, would be several times the stream
it is added to. The issue and the reference keep this norm. Dropping it
is a lab swap.

### Why the add is a comprehension

`residual_stream` and `attention_outputs` are plain lists, section 2 of
`docs/LAYERS.md`. The `+` operator on two lists is concatenation: it
gives one list of `2 * T` rows and makes no node. `Node.__add__` is the
add, and it runs when the left operand is a node. So the add has to
reach the nodes, two levels down:

```python
[
    [a + b for a, b in zip(xi, yi, strict=True)]
    for xi, yi in zip(residual_stream, attention_outputs, strict=True)
]
```

The outer clause pairs the rows. The inner clause pairs the elements
and makes one node per pair. Both `zip`s are strict, so two matrices of
different shape raise `ValueError`. The same comprehension appears
twice per block. A helper would give it a name and hide the two `zip`s,
and no lab swap touches the add, so it stays inline, as the convention
for elementwise maps says.

### Why the residual lives here

`attention` and `mlp` have no residual and no norm. Section 1 of
`docs/MLP.md` says both belong to the caller. The reason is the seam: a
block half that is a pure function of its arguments can be swapped for
another one without a change to the code around it. The model is the
one place that knows the order, and it wraps each call the same way.

### Causality through the whole model

`attention` is the only place where positions read each other, and it
reads positions `0` to `i` only, section 5 of `docs/ATTENTION.md`. Every
other step of `forward` acts on one row at a time: the embedding, each
norm, each `mlp`, each add, and the head. So the logits of position `i`
depend on tokens `0` to `i` only, through any number of blocks.

Group 8 checks it the same way the attention tests do. Change the token
at position `2`. The logits of positions `0` and `1` are the same
floats, not just close. The logits of position `2` change. A prefix of
two tokens gives the same two logits vectors as the full sequence. A
second place where positions met, such as an add that paired rows
wrong, would break the exact match.

## 6. The forward pass, step by step

> Code: `GPT.forward`
> Tests: group 4 (guards), group 5 (shape), group 7 (float reference),
> group 9 (numeric gradient check)

`forward` takes the whole sequence in one call and returns one logits
vector per position, as a `Matrix` of `T` rows of `vocab_size` nodes.
`T` is `len(tokens)`.

### The guards

Three guards run first, before any node is made, in this order:

| Guard                                 | Message                                            | Why a guard and not the free error                              |
|---------------------------------------|----------------------------------------------------|-----------------------------------------------------------------|
| `tokens` is empty                     | `tokens is empty`                                  | `attention` would raise `x is empty`, which names the wrong thing |
| `len(tokens) > context_length`        | `num_tokens=T exceeds context_length=N`            | `wpe[i]` would raise `IndexError` with no numbers                |
| a token outside `0 .. vocab_size - 1` | `token=t at position=i is not in 0...V-1`          | a negative token reads `wte` from the end and raises nothing     |

The empty check must come first. The other two read nothing from an
empty list and would pass it through. The token guard is a loop over
`tokens` with `enumerate`, so the message can name the position. It is
`T` integer comparisons against tens of thousands of nodes, so its cost
does not show. The third row is the reason the guard earns its place:
the free behavior for `-1` is silent and wrong.

### The stages

| Stage      | Code                                                      | Shape after        | Nodes made         |
|------------|-----------------------------------------------------------|--------------------|--------------------|
| Embed      | `[te + pe for te, pe in zip(wte[token], wpe[i])]` per `i` | `T x n_embd`       | `T * n_embd`       |
| Norm       | `[rmsnorm(x=row) for row in residual_stream]`             | `T x n_embd`       | per row, see below |
| Attention  | `attention(x=normed, wq=.., wk=.., wv=.., wo=.., n_head=..)` | `T x n_embd`    | section 6 of `docs/ATTENTION.md` |
| Add        | the comprehension of section 5                            | `T x n_embd`       | `T * n_embd`       |
| Norm       | as above                                                  | `T x n_embd`       |                    |
| MLP        | `[mlp(x=row, fc1=.., fc2=..) for row in normed]`          | `T x n_embd`       | section 2 of `docs/MLP.md` |
| Add        | as above                                                  | `T x n_embd`       | `T * n_embd`       |
| Final norm | as above                                                  | `T x n_embd`       |                    |
| Head       | `[linear(x=row, w=lm_head) for row in normed]`            | `T x vocab_size`   | `2 * T * vocab_size * n_embd` about |

The block loop runs the middle six rows `n_blocks` times. Each norm
makes about `3 * n_embd` nodes per row, section 6 of `docs/LAYERS.md`.
Each name says what the matrix is at that stage: `residual_stream` for
the stream, `normed` for the input of a half, `attention_outputs` and
`mlp_outputs` for what a half returned, `logits` for the result. One
name, `normed`, serves all three norms, because all three make the same
kind of thing: a matrix that one map reads once and nothing adds to.

### Whole sequence per call

One call builds the graph for every position. `attention` needs the
whole sequence to build its keys and values, and the loss of
`docs/LOSS.md` takes one logits vector per position. So the model has
no per-position entry point. Sampling, which wants one new token at a
time, calls `forward` on the whole prefix and reads the last vector.
The cost grows with the square of the length, and with
`context_length = 16` that is fine.

Every call makes a new graph. The parameters are the leaves of all of
them. `forward` does not reset gradients, and neither does `backward`,
section 9 of `docs/AUTOGRAD.md`. The optimizer does.

### The float reference

Group 7 compares `forward` against a second GPT in the test file,
written with floats and `math` only, that shares no code with the
module. It reads the values of `params` in list order, unpacks them by
the order of section 3, and runs the same stages. The two agree on the
shared case, on a second seed, and on a one-block, one-head case. A
missing norm, a lost residual, or `wte` and `wpe` swapped each change
the numbers while every shape stays right. This is the test that
catches those.

### The gradient check

Group 9 runs the check of section 10 of `docs/AUTOGRAD.md` on the
smallest model that has every part: one block, one head, `n_embd = 2`,
`vocab_size = 3`, `context_length = 3`, so 66 parameters. The loss is
`sequence_loss` of the logits, so the whole base model, from the first
lookup to the mean loss, is under one check. Every parameter is
perturbed. The targets are fixed, so no gradient is zero by
construction.

The MLP has the ReLU kink of section 2 of `docs/MLP.md`. The float
reference returns every pre-activation of every block and position,
and the test checks that each one is more than `1e-3` from zero before
it compares anything.

### What `forward` does not do

- No loss. It returns logits, and `sequence_loss` takes them.
- No sampling. It returns every position, and the caller picks one.
- No gradient reset. See above.
- No shape check beyond the three guards. A table of the wrong shape
  raises `ValueError` from a strict `zip` inside `dot`, section 3 of
  `docs/LAYERS.md`.
