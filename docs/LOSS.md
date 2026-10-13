# Loss: Design Notes

> Code: [`src/gptlab/loss.py`](../src/gptlab/loss.py)
> Tests: [`tests/gptlab/test_loss.py`](../tests/gptlab/test_loss.py)

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Cross-entropy for one position](#2-cross-entropy-for-one-position)
- [3. The mean over positions](#3-the-mean-over-positions)
- [4. Where `inf` and `nan` come from](#4-where-inf-and-nan-come-from)


## 1. The problem

The model reads a sequence of **tokens**, the units of text it works with.
The set of all tokens it knows is the **vocabulary**, and `vocab_size` is
its size. At each position, the model gives one number per token in the
vocabulary. These numbers are the **logits** of that position. A larger
logit means the model rates that token as a more likely next token. The
logits are not probabilities. Section 5 of `docs/LAYERS.md` turns them into
probabilities with `softmax`.

The training data says which token comes next at each position. That token
is the **target** of the position.

The loss is one number for the whole sequence. It is small when the model
gives a high probability to the target at every position, and large when
it does not. Training lowers it. The backward pass of section 9 of
`docs/AUTOGRAD.md` starts from one node and walks the graph behind it. The
loss is that one node, so the whole forward pass, from the first lookup to
the last logit, is in its graph.

The module has two functions:

| Function        | Takes                                        | Returns  |
|-----------------|----------------------------------------------|----------|
| `cross_entropy` | one logit vector, one target                 | one node |
| `sequence_loss` | one logit vector and one target per position | one node |

`cross_entropy` is the loss of one position. `sequence_loss` is the mean
of `cross_entropy` over the positions. The model calls only
`sequence_loss`.

## 2. Cross-entropy for one position

> Code: `cross_entropy`
> Tests: group 1 (shape), group 2 (hand case), group 3 (properties)

The function computes:

$$
\text{loss} = -\log\big(\mathrm{softmax}(\text{logits})[\text{target}]\big)
$$

In words: turn the logits into probabilities, read the probability of the
target, and return the negative of its log.

### Why the log

The model gives a probability to the target at each position. The
probability it gives to the whole sequence is the product of those
numbers. A product of many numbers below one is a very small number, and
in floats it rounds to zero. The log of a product is a sum of logs. So the
log turns the product over positions into a sum over positions, and
section 3 takes that sum. Nothing rounds to zero on the way.

### Why the minus

A probability is at most one, so its log is at most zero. The negative is
at least zero. It is zero only when the target has probability one. A loss
is a number that training lowers, so the sign is turned.

Two reference values follow from the formula:

| Logits                       | Loss              | Meaning                     |
|------------------------------|-------------------|-----------------------------|
| all equal                    | $\log(n_{\text{vocab}})$ | the model knows nothing     |
| target far above every other | about $0$                     | the model is sure and right |

The first is the loss at the start of training, when the weights are small
and random. With a vocabulary of 4 tokens it is $\log 4 = 1.386$.

### The code

The body is three lines and a guard:

| Line                                        | Makes                          |
|---------------------------------------------|--------------------------------|
| `probabilities = softmax(logits=logits)`    | the graph of section 5 of `docs/LAYERS.md` |
| `probability = probabilities[target]`       | no node                        |
| `loss = -probability.log()`                 | one `log` node, one product node, one leaf |

`softmax` is the function of `layers`. The code does not compute the
exponentials again. One softmax in the code base means one place to read
about the shift, and the shift already prevents overflow here.

The index read makes no node. It picks one node object out of the list.
That node is already in the graph, with the whole softmax behind it. The
embedding lookup of the model does the same thing with a row of a matrix.

`-x` on a node is the derived operator `__neg__` of section 7 of
`docs/AUTOGRAD.md`. It is `x * -1`. The `-1` becomes a leaf node, as
section 6.1 of that document says every float does. So the negation costs
one product node and one leaf, the same cost as the softmax shift. No new
primitive is needed, and the backward pass needs nothing new.

### The gradient

Write $p_i$ for the probability of token $i$, $t$ for the target, and $S$
for the sum of the exponentials. Then $p_t = \exp(x_t) / S$, and:

$$
\text{loss} = -\log(p_t) = \log(S) - x_t
$$

Take the derivative with respect to logit $i$:

1. $d\log(S) / dx_i$ is $\exp(x_i) / S$, which is $p_i$.
2. $dx_t / dx_i$ is $1$ if $i$ is the target and $0$ otherwise.

So:

$$
\frac{d\,\text{loss}}{dx_i} = p_i - [i = t]
$$

The bracket $[i = t]$ is $1$ when $i$ is the target and $0$ otherwise.

The gradient of each logit is its probability, minus one at the target.
This has three consequences the tests hold in place:

- Every gradient is between $-1$ and $1$, because $p_i$ is between $0$
  and $1$. A confidently wrong model gets a gradient near $-1$ at the
  target and near $1$ at the token it chose. The logits never receive a
  gradient larger than one from this loss.
- The gradients sum to zero, because the $p_i$ sum to one and the one-hot
  sums to one. A constant added to every logit does not change the loss,
  so the loss has no gradient in that direction.
- The gradient is never zero for finite logits, because $p_i$ is never
  exactly $0$ or $1$.

The autograd engine does not know this formula. It gets the same numbers
by the chain rule through `log`, the product, and the softmax graph. The
hand case test compares the two.

### The guard

`target` is a list index. Python raises `IndexError` for an index past the
end. It raises nothing for a negative index: `probabilities[-1]` is the
last probability, and the loss is a wrong number with no error. The
convention adds a guard only where its message says more than the free
error does. Here there is no free error for the negative case, so the
function checks `0 <= target < len(logits)` and raises `ValueError`. The
message names the bad value and the valid range, as the `n_head` guard in
section 4 of `docs/ATTENTION.md` does.

### The hand case

The tests use three logits chosen so that the exponentials are whole
numbers:

$$
\begin{aligned}
\text{logits} &= [0,\ \log 2,\ \log 3] \\
\exp(\text{logits}) &= [1,\ 2,\ 3], \qquad S = 6 \\
p &= [1/6,\ 2/6,\ 3/6]
\end{aligned}
$$

With target $t = 2$, the loss is $-\log(1/2) = \log 2$. The gradient is
$[1/6,\ 1/3,\ -1/2]$: the probabilities, minus one at the target.

The case is chosen so that each plausible wrong implementation gives a
different number:

| Implementation                 | Loss at target 2             |
|--------------------------------|------------------------------|
| Correct                        | $\log 2 = 0.693$             |
| Missing minus sign             | $-0.693$                     |
| Indexes target 1               | $\log 3 = 1.099$             |
| Indexes target 0               | $\log 6 = 1.792$             |
| `log` of the logit, no softmax | $-\log(\log 3) = -0.094$     |

## 3. The mean over positions

> Code: `sequence_loss`
> Tests: group 1 (shape), group 4 (the mean), group 6 (numeric gradient check)

The function computes one `cross_entropy` per position and returns their
mean:

$$
\text{loss} = \frac{\text{loss}_0 + \text{loss}_1 + \cdots + \text{loss}_{n-1}}{n}
$$

### Why the mean and not the sum

The gradient of a sum of $n$ losses is the sum of $n$ gradients. Each
position contributes a gradient to the shared weights, so the gradient of
a sum grows with $n$. The optimizer multiplies the gradient by the learning
rate. With a sum, a sequence twice as long would move the weights twice as
far per step, and the learning rate would have to change with the sequence
length. With the mean, every position contributes $1/n$ of its gradient,
and the step size does not depend on $n$. The learning rate is then one
number for every sequence length.

### The code

| Line                                                   | Makes                    |
|--------------------------------------------------------|--------------------------|
| `losses = [cross_entropy(...) for ... in zip(..., strict=True)]` | one graph of section 2 per position |
| `sum_of_losses = losses[0]`, then `+=` in a loop       | $n - 1$ sum nodes        |
| `mean_loss = sum_of_losses / len(logits_per_position)` | one product node, one leaf |

The backward pass follows the table. The product sends $1/n$ of the
incoming gradient to the sum. Each sum passes it unchanged to both of its
children. So every per-position loss receives $1/n$, and every logit
receives its gradient of section 2 divided by $n$. The logits of one
position do not change the gradient of another. Group 4 has a test for
each of these facts.

### The guards

`zip` over the logits and the targets has `strict=True`. By default `zip`
stops at the shorter list and raises nothing. With the flag, two lists of
different lengths raise `ValueError`. This is the same decision as in
`dot`, section 3 of `docs/LAYERS.md`.

Two empty lists pass the strict `zip`. The seed `losses[0]` would then
raise `IndexError` with the message `list index out of range`, which
names no list. So the first line of the function checks that the sequence
has a position and raises `ValueError` with the message
`logits_per_position is empty`. The same guard, for the same reason, is in
every function that reads element zero of an input: `dot`, `softmax`, and
`rmsnorm` in `layers`, and `split_heads` and `attention` in `attention`.

## 4. Where `inf` and `nan` come from

> Code: `-probability.log()` in `cross_entropy`
> Tests: group 5 (inf and nan)

Section 6.6 of `docs/AUTOGRAD.md` says that if a gradient becomes `inf` or
`nan`, look at `log` first. This section says exactly when that happens in
this loss, with numbers.

### The gap

Take two logits. The target logit is below the other one by a **gap**.
The shift in `softmax` subtracts the largest logit, so the target becomes
$-\text{gap}$ and the other becomes $0$. The probability of the target is
then close to $\exp(-\text{gap})$, and the loss is close to $\text{gap}$. The
loss is a plain number for every gap. The trouble is in the floats that
carry it.

A float holds numbers down to about $10^{-308}$ at full precision, and then
smaller numbers with less precision, down to about $5 \cdot 10^{-324}$. Below
that it holds `0.0`. $\exp(-\text{gap})$ crosses these limits at gaps of about
$709$ and $745$.

| Gap           | Forward pass                          | Backward pass                          |
|---------------|---------------------------------------|----------------------------------------|
| up to 709     | finite, about equal to the gap        | correct, $p - \text{onehot}$, about $[1, -1]$ |
| 710 to 745    | finite                                | $1/p$ in the propagate function of `log` overflows to `inf`. The next product makes `nan`. |
| 746 and past  | `exp` underflows to `0.0`, and `log(0.0)` raises `ValueError` | never runs |

The tests use gaps of `700`, `720`, and `800`, one in each zone.

In the middle zone nothing warns. The loss is a finite number, the
backward pass runs, and the logits hold `inf` and `nan`. The optimizer
would then write `nan` into the weights, and every later forward pass
would give `nan`. If that happens, this is the first place to look.

### Why the softmax form stays

The derivation in section 2 gives a second form of the same loss:

$$
\text{loss} = \log(S) - x_{\text{target}}
$$

This is the **log-softmax** form. It never makes the probability of the
target, so it has no zones. $\log(S)$ is finite for every gap, because the
shift keeps $S$ between $1$ and `vocab_size`. The forward pass and the
backward pass are finite for every input.

The code uses the softmax form. The issue gives the loss as
$-\log(\mathrm{softmax}(\text{logits})[\text{target}])$, which is the formula
every text gives,
and the code is the formula with one call and one index. The log-softmax
form would compute the sum of the exponentials a second time outside
`softmax`, or need a second function in `layers` that returns the log of
the sum. The reason to keep the simpler form is the one this document is
for: the zones are a fact about floats and `log` that a reader can see in
three tests, and the simpler code shows where they come from.

The model in this lab starts with small random weights, so the logits of
one position differ by a few units, and the loss is near
$\log(n_{\text{vocab}})$. A gap of `700` needs logits hundreds of units apart.
The log-softmax form is a later lab swap, when a trained model gives a
reason for it.
