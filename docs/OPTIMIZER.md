# Optimizer: Design Notes

> Code: [`src/gptlab/optimizer.py`](../src/gptlab/optimizer.py)
> Tests: [`tests/gptlab/test_optimizer.py`](../tests/gptlab/test_optimizer.py)

---

## Contents

- [1. The problem](#1-the-problem)
- [2. Plain gradient descent, and why it is not enough](#2-plain-gradient-descent-and-why-it-is-not-enough)
- [3. The two moments](#3-the-two-moments)
- [4. Bias correction](#4-bias-correction)
- [5. The decay schedule](#5-the-decay-schedule)
- [6. Zeroing the slots](#6-zeroing-the-slots)


## 1. The problem

Section 9 of `docs/AUTOGRAD.md` says that training has three stages per
step, and covers the first two. This module is the third.

| Stage          | Code                                | Reads                                 | Writes                                    |
|----------------|-------------------------------------|---------------------------------------|-------------------------------------------|
| forward pass   | `GPT.forward`, then `sequence_loss` | the forward values of the parameters  | a new graph, with the loss as its root    |
| backward pass  | `Node.backward`                     | the graph                             | the gradient slot of every node in it     |
| optimizer step | `Adam.step`                         | the gradient slot of every parameter  | the forward value of every parameter      |

The first two stages change no parameter. After them, the gradient slot
of each parameter holds $dL/dp$: how much the loss rises when the
parameter rises by one. The optimizer step reads that number and moves
the parameter the other way. The move is the only place in the code base
where a forward value changes after construction.

The optimizer finds the parameters through the flat list `params` of
section 3 of `docs/MODEL.md`. It does not walk the graph. The list holds
the same node objects as the tables of the model, so the nodes that the
optimizer moves are the nodes that the next forward pass reads. The graph
of one step is stale after `step`. The next forward pass makes a new one
from the moved parameters, and nothing reads the old one again.

The module has one class:

```
Adam(*, params, num_steps, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8)
    .step(*, t)
    .zero_grad()
```

| Name             | Kind       | Holds                                                      |
|------------------|------------|------------------------------------------------------------|
| `params`         | argument   | the `n_params` leaf nodes to move, the caller's list        |
| `num_steps`      | argument   | the number of steps in the run, for the decay of section 5 |
| `lr`             | argument   | the step size at step zero                                 |
| `beta1`          | argument   | the decay of the first moment, section 3                   |
| `beta2`          | argument   | the decay of the second moment, section 3                  |
| `eps`            | argument   | keeps the division finite, section 3                       |
| `first_moments`  | state      | one float per parameter, the running mean of the gradient  |
| `second_moments` | state      | one float per parameter, the running mean of its square    |
| `t`              | `step` arg | the step number, `0` to `num_steps - 1`                    |

The four rates keep the defaults of the issue. A caller that does not
care gets settings that work. `num_steps` has no sensible default, so it
is required.

The training loop calls the three stages in order, then clears the slots:

```
for t in range(num_steps):
    loss = sequence_loss(...)        forward pass, a new graph
    loss.backward()                  backward pass, fills the slots
    optimizer.step(t=t)              moves the parameters
    optimizer.zero_grad()            clears the slots
```

Group 7 runs this loop on the smallest graph there is. The one parameter
is $x$, the loss is $(x - 3)^2$, and twenty steps move $x$ from $0$
toward $3$. The test checks the end-to-end contract between the three
modules and nothing else. Section 2 says what the numbers of that run
show.

The module reads and writes floats only. Every line of `step` is float
arithmetic on `forward_val`, `gradient`, and the two state lists. No
line makes a node. Group 3 checks that every parameter is still a leaf
after a step. The module has no backward pass, so the test file has no
numeric gradient check. Group 7 is the end-to-end test in its place.

## 2. Plain gradient descent, and why it is not enough

> Code: the last line of `step`, `param.forward_val -= ...`
> Tests: group 3 (properties), group 7 (convergence)

The simplest optimizer step is **gradient descent**:

$$
p \leftarrow p - \mathit{lr}  g
$$

$g$ is the gradient slot of $p$. It says which way raises the loss. The
minus moves the other way. $\mathit{lr}$ is the **learning rate**, the size of
the move for a gradient of one. A negative gradient moves the parameter
up. Group 3 has a test for the sign.

Adam keeps this form. The last line of `step` subtracts
$\mathit{lr_t} \cdot \mathrm{ratio}$ from the forward value. The two changes are
in what the ratio is and in how $\mathit{lr_t}$ falls over the run. Plain gradient descent has two problems
that the ratio fixes.

### The scale of the gradient

The move of gradient descent is $\mathit{lr} g$. Its size depends on $g$. In a
model, $g$ differs by orders of magnitude from one table to another, and
from one step to the next. One $\mathit{lr}$ serves every parameter. A rate small
enough for the largest gradient moves the smallest gradient by nothing. A
rate large enough for the smallest gradient throws the largest one past
the minimum.

Adam divides the gradient by a running measure of its own size, so the
move is about $\mathit{lr}$ for every parameter. Group 3 has the test: three
parameters with gradients `0.001`, `1.0`, and `1000.0` each move by
`0.5` at step zero with `lr = 0.5`. Section 3 gives the measure.

### The noise of one step

Each step reads one sequence. The gradient of that step points partly
toward the minimum and partly toward whatever that one sequence pulls.
Adam averages the gradient over steps. The part that agrees from step
to step adds up. The part that changes sign cancels. A parameter whose
gradient is zero at this step still moves, from the average of the
earlier steps. Section 3 shows this with numbers.

### What the convergence test shows

Group 7 is not a comparison with gradient descent. On its one-parameter
quadratic with `lr = 0.3`, gradient descent reaches `3.0` to four digits
in thirteen steps. Adam reaches `2.71` in twenty. The reason is the
first fix above: the move of Adam is at most about $\mathit{lr_t}$ in size, so
it walks toward the minimum by $0.3$ per step at the start and by less as
the decay of section 5 takes hold. Gradient descent takes a step of
$0.6 (3 - x)$, which is large when $x$ is far away. On one parameter
with a well chosen rate, gradient descent wins. The two problems above
are problems of a model with hundreds of parameters, where no one rate
is well chosen for all of them.

## 3. The two moments

> Code: `Adam.__init__`, `first_moments`, `second_moments`, and the two
> update lines of `step`
> Tests: group 1 (construction), group 2 (hand case), group 3
> (properties), group 6 (float reference)

A **moment** is a running average over steps. Adam keeps two per
parameter:

$$
\begin{aligned}
m &\leftarrow \beta_1 m + (1 - \beta_1)  g && \text{first moment, of the gradient} \\
v &\leftarrow \beta_2 v + (1 - \beta_2)  g^2 && \text{second moment, of the squared gradient}
\end{aligned}
$$

$\beta_1$ and $\beta_2$ are the arguments `beta1` and `beta2`. Each moment is
an **exponential running average**. At each step it keeps a fraction
$\beta$ of the old average and adds a fraction $1 - \beta$ of the new value.
A value from $k$ steps ago has weight $\beta^k$, so the average remembers
about $1/(1 - \beta)$ steps.

| Moment | $\beta$          | Remembers about | Measures                       |
|--------|------------------|-----------------|--------------------------------|
| first  | $\beta_1 = 0.85$ | 7 steps         | the direction of the gradient  |
| second | $\beta_2 = 0.99$ | 100 steps       | the size of the gradient       |

The second moment remembers longer because the size of the gradient
changes slowly, and a long average of it is steadier. The first moment
remembers less because the direction must follow the loss as the
parameters move.

The move is:

$$
p \leftarrow p - \mathit{lr_t} \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

$\hat{m}$ and $\hat{v}$ are the corrected moments of section 4. $\epsilon$ is the
argument `eps`. For a gradient that is steady from step to step the
corrected moments are $g$ and $g^2$. $\sqrt{\hat{v}}$ is then $|g|$, the size
of the gradient, and the ratio is $g / |g|$, which is $1$ or $-1$. The move
is $\mathit{lr_t}$, whatever the size of $g$. This is the first fix of
section 2. With the defaults, the move is never larger than $\mathit{lr_t}$.

### The hand case

Group 2 is one parameter at `2.0` with `lr = 0.1`, `beta1 = 0.25`,
`beta2 = 0.75`, and `num_steps = 4`. The rates are far from the defaults
so that each step changes the moments by an amount a reader can see.
The gradient is `4.0` at step 0 and `0.0` at step 1.

| Step | $g$ | $m$    | $v$  | $\hat{m}$ | $\hat{v}$ | $\mathit{lr_t}$ | Move      | $p$         |
|------|-----|--------|------|-----------|-----------|-----------------|-----------|-------------|
| 0    | 4.0 | 3.0    | 4.0  | 4.0       | 16.0      | 0.1             | 0.1       | 1.9         |
| 1    | 0.0 | 0.75   | 3.0  | 0.8       | $48/7$    | 0.075           | 0.0229129 | 1.8770871   |

Step 0: $m = 0.75 \cdot 4 = 3$ and $v = 0.25 \cdot 16 = 4$. The corrections
divide by $1 - 0.25 = 0.75$ and $1 - 0.75 = 0.25$, which gives $4$ and
$16$. The ratio is $4/4 = 1$ and the move is $\mathit{lr}$.

Step 1: the gradient is zero, so $m = 0.25 \cdot 3 = 0.75$ and
$v = 0.75 \cdot 4 = 3$. The corrections divide by $1 - 0.0625$ and
$1 - 0.5625$. The ratio is $0.8 / 2.6186 = 0.3055$, the decayed rate is
$0.075$, and the move is $0.0229$.

The second row answers the question that the issue carries over. A
parameter with gradient zero at this step still moves, because its first
moment from the step before is not zero. Gradient descent moves it by
$\mathit{lr} \cdot 0 = 0$. Group 3 has the other half of the fact: a parameter with
gradient zero at step 0 does not move, because both of its moments are
zero, and a second parameter with a nonzero gradient does not move it,
because each parameter has its own two moments.

Four plausible wrong implementations give a different number on this
case:

| Wrong step                         | After step 0 | After step 1 |
|------------------------------------|--------------|--------------|
| no bias correction                 | 1.85         |              |
| correction with $t$, not $t + 1$   | raises       |              |
| `beta1` and `beta2` swapped        | 1.9          | 1.8281       |
| gradient descent, no moments       | 1.6          | 1.6          |

### The state

The two moments are two `list[float]`, one float per parameter, at the
same index as the parameter in `params`. Both start at `0.0`.
Construction reads no gradient and changes no forward value. Group 1
has the tests, and one for an empty list, which constructs, steps, and
zeroes with no error.

The moments are floats and not nodes. The update is not part of any
loss. A node per moment would have a propagate function that nothing
calls, and the list of children of each new node would hold the node of
the step before, so the graph would grow by one node per parameter per
step with no reader. The optimizer makes no node, and `step` leaves
every parameter a leaf.

The loop is `enumerate(self.params)` and writes the lists by index. A
`zip` over the three lists would give each moment a local name, and a
new float assigned to a local name does not change the list. The index
is what the write-back needs. `forward` in `model.py` uses `enumerate`
in the same way, to pair a token with its position.

The names are `first_moments` and `second_moments`, the words of the
Adam paper and of this section. Karpathy's `m` and `v` name the letters
of the formula, not the thing the list holds.

### Where `eps` sits

The denominator is $\sqrt{\hat{v}} + \epsilon$. Section 6 of `docs/LAYERS.md`
puts `eps` inside the root of `rmsnorm`. The two placements differ
because the powers differ:

| Function  | Power  | At zero, with `eps` outside | Choice                               |
|-----------|--------|-----------------------------|--------------------------------------|
| `rmsnorm` | $-1/2$ | $0^{-1/2}$ raises           | inside, $(x + \epsilon)^{-1/2}$       |
| `step`    | $1/2$  | $0^{1/2}$ is $0$            | outside, $\sqrt{v} + \epsilon$        |

A positive power of zero is zero and raises nothing. The sum with `eps`
then keeps the division finite when the second moment is zero, which it
is for a parameter whose gradient has been zero at every step so far.
With both moments zero the ratio is $0 / \epsilon = 0$ and the parameter
does not move. The form with `eps` outside the root is the one of the
Adam paper and of Karpathy's code. The form with `eps` inside would also
be finite; this one is the reference form.

### What `step` does not do

`step` does not clear the gradient slots; section 6 says who does. It
does not copy `params`; the nodes move in place, and group 3 checks by
identity that the list holds the same objects after a step. It does not
return a value.

## 4. Bias correction

> Code: `corrected_first_moment`, `corrected_second_moment` in `step`
> Tests: group 2 (hand case), group 6 (float reference)

Both moments start at zero. At step 0, the first moment is
$\beta_1 \cdot 0 + (1 - \beta_1) g$, which is $0.15 g$ with the defaults.
The running average is biased toward its zero start, and the bias is
largest at the first step.

Put a steady gradient $g$ into the first-moment update $t + 1$ times:

$$
\begin{aligned}
m &= (1 - \beta_1)  g  (1 + \beta_1 + \beta_1^2 + \cdots + \beta_1^t) \\
&= (1 - \beta_1^{t+1})  g
\end{aligned}
$$

The factor $1 - \beta_1^{t+1}$ is the part of the full weight that
the average has collected so far. Division by it gives $g$ at every
step. The same holds for the second moment with $\beta_2$:

$$
\hat{m} = \frac{m}{1 - \beta_1^{t+1}},
\qquad
\hat{v} = \frac{v}{1 - \beta_2^{t+1}}
$$

The factors go to one as $t$ grows, and the correction fades:

| $t$   | $1 - \beta_1^{t+1}$ | $1 - \beta_2^{t+1}$ |
|-------|------------------------|------------------------|
| 0     | 0.15                   | 0.01                   |
| 1     | 0.2775                 | 0.0199                 |
| 9     | 0.8031                 | 0.0956                 |
| 49    | 0.9997                 | 0.395                  |
| 99    | 1.0                    | 0.634                  |
| 499   | 1.0                    | 0.9934                 |

The second moment needs the correction for hundreds of steps. Without
it, $\sqrt{v}$ at step 0 is $0.1 |g|$, the first moment is $0.15 g$,
and the ratio is $1.5$ instead of $1$. The first move would be
$1.5 \mathit{lr}$. On the hand case of section 3, the uncorrected step 0
gives $1.85$ and the corrected one gives $1.9$.

### Why `t + 1`

`t` is zero-based, as `range(num_steps)` gives it to the training loop.
The paper counts steps from one. With $t$ in the exponent, the factor at
the first step is $1 - 1 = 0$ and the division raises. The code uses
`t + 1`, so that step 0 divides by $1 - \beta$, which is the paper's
first step.

### Why the corrected moments are locals

The two corrected values are locals of one step. The lists keep the
uncorrected averages. The correction depends on `t` only and serves one
move. Written back into the list, the next step would correct a value
that is already corrected, and the average would grow by the factor at
every step. Group 6 holds this in place: six steps of the module match
six steps of a float Adam that shares no code with it, with the
defaults and with other rates.

## 5. The decay schedule

> Code: `decayed_lr` and the guard in `step`
> Tests: group 4 (decay), group 6 (float reference)

The learning rate falls in a straight line to zero over the run:

$$
\mathit{lr_t} = \mathit{lr} \left(1 - \frac{t}{n_{\text{steps}}}\right)
$$

| $t$                         | $\mathit{lr_t}$                       |
|-----------------------------|---------------------------------------|
| $0$                         | $\mathit{lr}$                         |
| $n_{\text{steps}}/2$     | $\mathit{lr}/2$                       |
| $n_{\text{steps}} - 1$   | $\mathit{lr}/n_{\text{steps}}$     |
| $n_{\text{steps}}$       | $0$, never taken                      |

The rate reaches zero one step after the last. The last step still
moves, by $\mathit{lr}/n_{\text{steps}}$. A rate of exactly zero at the last step
would waste the step.

Section 3 says why the decay is needed. The move of Adam is about
$\mathit{lr_t}$ in size, whatever the size of the gradient. Near a minimum the
gradient is small, but the ratio of the moments is not, so a constant rate
keeps the parameter moving by moves of about $\mathit{lr}$ from one side of
the minimum to the other. The decay makes the last moves small, so the
parameter settles. Group 4 has the tests: with the same gradient at every
step, the moves are $\mathit{lr_t}$ at every step, so the first move is
$\mathit{lr}$, the last is $\mathit{lr}/n_{\text{steps}}$, and the list of
moves is the line.

A straight line to zero is one line of code and one number to set.
Karpathy's `microgpt.py` uses it. A warmup, a cosine, or a constant rate
each needs more settings and gives this lab no more to see with a run
of a few hundred steps.

`num_steps` is an argument of the constructor and not of `step`. The
length of the run is fixed before the first step, as `lr` is, and
`step` then takes one number per call. The code computes $\mathit{lr_t}$ one
time per step, outside the loop over the parameters, because it does
not depend on the parameter.

### The guard

Past `num_steps`, $\mathit{lr_t}$ is zero and then negative, and the parameter
moves up the gradient with no error. `step` checks `t` against
`range(num_steps)` and raises `ValueError`. The message names `t`, the
range, and `num_steps`:

```
step t=6 is not in 0...5 (num_steps=6)
```

`t` is one letter. Unlike `target` in section 2 of `docs/LOSS.md`,
the value alone does not say where the limit came from, so the message
says it. Group 4 has one test for `t = num_steps` and one for `t = -1`.

## 6. Zeroing the slots

> Code: `Adam.zero_grad`
> Tests: group 5 (zeroing)

Section 5 of `docs/AUTOGRAD.md` says that each propagate function adds
to the gradient slot of a child with `+=`, because a node that two
operations read gets one contribution from each. Section 9 of that
document says that `backward` does not clear the slots before the walk,
and that the optimizer owns the zeroing. `zero_grad` is that method. It
sets the gradient slot of every parameter to `0.0`. It changes no
forward value and no moment.

Group 5 has the fact from both sides. Two calls of `backward` on the
same graph with no zeroing give $[8.0, 4.0]$, twice the true gradient
$[4.0, 2.0]$. With `zero_grad` between the two calls, the second gives
$[4.0, 2.0]$ again.

### Why the optimizer and not `backward`

`backward` knows the graph. The graph of a step holds only the
parameters that the step read. Section 4 of `docs/MODEL.md` says which
parameters are left out: the rows of `wte` for tokens that are not in
the input, and the rows of `wpe` past the length of the input. If
`backward` cleared the slots of the nodes it walks, a parameter outside
the graph would keep the gradient of the last step that read it, and
`step` would move it again from that stale number at every later step.
The optimizer holds every parameter through `params`, in the graph or
not, so it can clear every slot. `backward` keeps one job: fill the
slots from the graph as it is.

### Why a method of its own

`step` could clear each slot after it reads it. Karpathy's loop does.
The code keeps `zero_grad` separate, so that the training loop of
section 1 shows the contract in three lines: fill the slots, move the
parameters, clear the slots. When a later optimizer replaces `Adam`,
the loop stays the same and the contract stays visible.

### Where in the loop

Group 7 calls `zero_grad` after `step`. The slots are then zero between
steps, as they are at construction, and the next `backward` starts from
zero. A call before `backward` also works, because the slots start at
zero. The document shows the order of the test.
