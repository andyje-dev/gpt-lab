# Design Notes

One document per module. Each document follows the order of its code, and
each numbered section names the code and the tests that it explains.

Read the documents in this order. Each one uses the words that the documents
before it define.

| Order | Document                     | What it explains                                                                                                                                                                                                                                                       |
|-------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | [AUTOGRAD.md](AUTOGRAD.md)   | The scalar autograd engine. One `Node` holds one float, its children, and a propagate function. The six primitives, the derived operators, topological order, and the backward pass. Defines "node", "leaf", "children", "propagate function", and "scalar operation". |
| 2     | [LAYERS.md](LAYERS.md)       | The layer functions `dot`, `linear`, `softmax`, `rmsnorm`, and `transpose`, and the plain-list `Vector` and `Matrix` types. Why a seeded loop and not `sum`, why the softmax shift is a float, why `eps` is inside the root, and why `transpose` makes no node. Defines "vector" and "matrix". |
| 3     | [ATTENTION.md](ATTENTION.md) | Causal multi-head self-attention over a whole sequence. `attend` as the core, one query over many keys and values. `split_heads` as slices. The scaled score, the weighted sum as a `linear` over a transpose, the causal mask as a slice bound, and the output projection.                                     |
| 4     | [MLP.md](MLP.md)             | The two-layer feed-forward block for one position. Why attention alone is linear inside a position. Expand to `n_hidden` with `linear`, cut with ReLU, contract back. Why ReLU and not GELU, why the cut sits between the maps, what a cut unit does to the gradient, and the kink guard in the gradient check.                                           |

## How to read a document

The sections of a document follow the order of its code. Read one section.
Then read the code that the section names. No section depends on a later
section.

Each numbered section starts with two lines:

- `Code:` names the function, method, alias, or lines that the section
  explains. It never gives line numbers.
- `Tests:` names the test group that holds the section in place. A comment
  at the top of each test group points back to the section.

A section number with no document name points to the document you are in.
A reference to another document always names it.

Each document ends with a section on the numeric gradient check. The method
is defined one time, in section 10 of `AUTOGRAD.md`, and the later documents
say only what is specific to their module.
