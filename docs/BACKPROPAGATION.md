# Backpropagation: Design Notes

---

## 1. The problem

Training changes the weights of a model to decrease a loss function. To know
the direction of the change for a weight, we must know the derivative of the
loss with respect to that weight.

In pure Python, a weight is a float. A float keeps no record of its history.
Thus, we must make a data structure.

## 2. What a value must carry

The loss is one number. It is the last result of a long chain of operations:

weights -> embeddings -> attention -> MLP -> logits -> softmax -> loss

To calculate the derivative of the loss with respect to a weight in the chain,
we use the **chain rule**. For each link in the chain, the chain rule must
know two things: the operation that made the link, and the inputs to that
operation.

Thus, we make a node class. Each scalar in the model is a node. Each node
holds these fields:

| Field                   | Purpose                                                     |
|-------------------------|-------------------------------------------------------------|
| forward value           | the number computed in the forward pass                     |
| gradient slot           | dL/d(this value), filled in during backprop, starts at zero |
| parents                 | pointers to the values that produced this one               |
| local backward function | given this node's gradient, push gradient to parents        |

Important: this applies to **each** value in the forward pass, not only to
the weights. Intermediate results (the output of a multiply, an attention
score, a softmax probability) must also keep their history. The path from
the loss back to a weight goes through them.

## 3. The graph

A weight does not connect directly to another weight. A weight connects to
the value that it helped to make. That value connects to the next value, and
so on, up to the loss.

- The result is a **directed acyclic graph** (DAG). The values are the
  nodes. The operations are the edges.
- The **loss** is the root.
- The **weights** are leaves. These leaves are the values that we can
  adjust.
- All other nodes are intermediate results.

We do not make the graph before the forward pass. The forward pass makes the
graph **as a side effect**: each `+` and each `*` on node objects makes a new
node. The new node records its parents. Thus, we must overload the arithmetic
operators of the node class to do this record.

## 4. Local derivatives

Backprop comes to a node `c` with the value `dL/dc`. The node must compute
`dL/d(parent)` for each parent and send it to that parent. The chain rule
gives:

```
dL/d(parent) = dL/dc * dc/d(parent)
```

The second factor, `dc/d(parent)`, is the **local derivative**. It depends only on the
operation that made `c`.

### Addition: `c = a + b`

- `dc/da = 1`, `dc/db = 1`
- `c` sends `dL/dc` to `a` and to `b` with no change.
- The values of `a` and `b` are not necessary.

### Multiplication: `c = a * b`

- `dc/da = b`, `dc/db = a`
- `c` sends `dL/dc * b` to `a`. `c` sends `dL/dc * a` to `b`.
- `c` must keep a reference to each parent. `a` alone does not know the
  value of `b`.

### Power: `c = a ** n`

- `n` is a plain number, not a node. It is fixed during the forward pass.
- `dc/da = n * a ** (n - 1)` (the power rule).
- `c` sends `dL/dc * n * a ** (n - 1)` to `a`. There is one parent, so
  there is one send.
- `c` must keep a reference to `a` and must also keep the number `n`. `n` is
  not a parent. It is not in the graph. The backward function must capture
  it in some other way.
- Example: `c = a ** 2` with `a = 3`. Then `dc/da = 2 * 3 = 6`.

### Where the derivative logic lives

The computation occurs **at `c`**, not at the parents. `c` knows the
operation that made it. `c` knows each parent. `c` receives the incoming
gradient. A parent knows none of these things. If a parent had to compute its
own gradient, it would have to know each operation that it was part of.

Thus, each node carries a small function: "given my gradient, update the
gradients of my parents." Addition makes one version of this function.
Multiplication makes a different version. The generic backward pass calls
these functions in the correct sequence. It does not have to know the
operation of any node.

The function takes one argument: the gradient of its own node. The backward
pass reads that gradient from the node and passes it in. Thus, the function
does not need a reference to the node that carries it. We can write the
function before we make that node, and pass it to the constructor. A leaf
carries a function that does nothing.

Each new operation that the model needs (exponent, log, power, a
nonlinearity, ...) is one more local derivative. We package each one in the
same way.

## 5. Accumulate the gradients

One value can be the input to more than one operation. If weight `w` is used
two times, `c = w * x` and `d = w * y`, then `c` and `d` will each send a
gradient to `w`. The true gradient is the **sum** of all paths (multivariable
chain rule):

```
dL/dw = (contribution from c) + (contribution from d)
```

## 6. Visitation order

A node can send gradient to its parents only when its own gradient is final.
Thus, each node that depends on a value must be processed before that value.

- Do a **post-order depth-first search** from the loss. This gives a list in
  which each node comes after all nodes that it depends on (a topological
  order). The loss is last.
- **Reverse** the list. Then walk the list and call the local backward
  function of each node. The loss is now first.
- Keep a **visited set**. Then a node that is reachable by more than one path
  (for example, a weight that is used two times) is in the list exactly one
  time.

## 7. Backprop does not touch forward values

Each node has two numbers: the forward value and the gradient slot. Backprop
writes only to the gradient slot. It does not change the forward value.

This rule must be strict for this reason: the local derivative of a
multiplication is the *forward value of the other parent*. If a forward value
changed during the backward pass, each later node that reads it would compute
its gradient from a number that the forward pass did not use. All gradients
are defined with respect to the forward pass as it occurred. Thus, forward
values must stay frozen until the full backward pass is complete.

The weight update is a **separate pass**. It is the task of the optimizer.
There are three stages:

1. **Forward pass**: compute the values and make the graph.
2. **Backward pass**: fill the gradient slots.
3. **Optimizer step**: adjust the forward values of the weights with their
   gradient slots. Then set the gradient slots to zero for the next step.

## 8. Which nodes are parameters

The graph has thousands of nodes, but only some of them are weights. Each
forward pass computes the intermediate results again from zero and then
discards them. Thus, an "update" to an intermediate result has no effect.
The optimizer must know which nodes are parameters. The graph alone does not
give this information.

Use a **plain list of parameters**, made one time when the model is 
constructed. The optimizer walks this list. It does not traverse the whole graph. 
This is simpler. It also tells us how to save and load a model.

### Embeddings

Input tokens are integers. They are not parameters, and they get no gradient
slot. We use them to **index** into the embedding table. The embedding table
is a list of lists of node objects. Those nodes are parameters.

Only the rows that we look up in a step are in the graph. The forward pass
does not touch the other rows. Thus, their gradient at the end of backprop is
zero, and plain gradient descent does not change them, however, this can change
depending on the optimizer implementation.

## 9. Operator overloading

The forward pass makes the graph through arithmetic on node objects. Python
calls a method on the node class for each operator. We write these methods.
Python does not know which attribute to add or multiply. The body of the
method tells it.

### How Python chooses the method

1. Python looks at the left operand first.
2. For `a + 2.0`, the left operand is a node. Python calls `a.__add__(2.0)`.
3. For `2.0 + a`, the left operand is a float. Python calls
   `float.__add__(2.0, a)`. The float does not know the node class. It
   returns `NotImplemented`.
4. Python then calls the **reflected** method on the right operand:
   `a.__radd__(2.0)`. The `r` means the operands are in the other order.

### Floats become leaf nodes

Inside `__add__`, `other` can be a float. A float has no `data`, no gradient
slot, and no parents. The backward function of the output node will write to
`other.grad`. This fails on a float.

Thus, the first line of each primitive wraps the float:

```python
if not isinstance(other, Node):
    other = Node(other)
```

The wrapped float is a **leaf**. It has no parents. It gets a gradient slot
that nothing reads. We accept this small waste over repetitive second code 
paths for floats in each primitive and in each backward function.

### Three primitives, seven combinations

Only `__add__`, `__mul__`, and `__pow__` have backward functions. The other
operators are built from these three. Each combination makes one or more
nodes in the graph, each with its own parents and backward function.

| Expression | Method          | Body                            |
|------------|-----------------|---------------------------------|
| `a + b`    | `__add__`       | primitive                       |
| `a * b`    | `__mul__`       | primitive                       |
| `a ** 2`   | `__pow__`       | primitive, number exponent only |
| `2.0 + a`  | `__radd__`      | `self + other`                  |
| `2.0 * a`  | `__rmul__`      | `self * other`                  |
| `-a`       | `__neg__`       | `self * -1`                     |
| `a - b`    | `__sub__`       | `self + (-other)`               |
| `2.0 - a`  | `__rsub__`      | `(-self) + other`               |
| `a / b`    | `__truediv__`   | `self * other ** -1`            |
| `2.0 / a`  | `__rtruediv__`  | `other * self ** -1`            |

Note on `__pow__`: the exponent is a plain number, not a node. A node
exponent would need a different local derivative (it includes the log of the
base). The GPT only raises values to fixed powers, so we do not build the
general case.
