"""Causal multi-head self-attention over a sequence.

``attend`` is the core: one query reads a list of keys and values and
returns one vector. ``split_heads`` cuts a matrix into one matrix per head.
``attention`` takes the whole sequence, projects it, splits it into heads,
and calls ``attend`` once per head per position with the keys and values of
positions ``0`` to ``i`` only. The causal mask is a slice bound, so no
masked score enters the graph. The design notes and the reason for each
choice are in docs/ATTENTION.md.
"""

from gptlab.layers import Matrix, Vector, dot, linear, softmax, transpose


def attend(*, q: Vector, keys: Matrix, values: Matrix, scale: float) -> Vector:
    """Return the weighted sum of ``values``, weighted by how ``q`` scores each key.

    The score of key ``j`` is ``dot(x=q, y=keys[j]) * scale``. The weights are the
    softmax of the scores. The function does not know about positions, heads,
    or the causal rule. The caller chooses which keys and values to pass.

    Args:
        q: The query. One vector with ``head_dim`` nodes.
        keys: One key per row. Each row has ``head_dim`` nodes.
        values: One value per row, in the same order as ``keys``. Each row
            has ``head_dim`` nodes.
        scale: A plain float that multiplies each score.

    Returns:
        One vector with ``head_dim`` nodes.

    Raises:
        ValueError: If a key does not have the same length as ``q``, or if
            ``keys`` and ``values`` do not have the same number of rows.
    """
    scores: Vector = [dot(x=q, y=key) * scale for key in keys]
    weights: Vector = softmax(logits=scores)

    return linear(x=weights, w=transpose(m=values))


def split_heads(*, m: Matrix, n_head: int) -> list[Matrix]:
    """Return ``m`` cut into ``n_head`` matrices, one per head.

    Head ``h`` holds elements ``h * head_dim`` to ``(h + 1) * head_dim - 1``
    of every row, where ``head_dim`` is the row length divided by
    ``n_head``. The function makes no node. Each slice holds the same node
    objects as the row it came from.

    Args:
        m: One row per position. ``n_head`` must divide the row length.
        n_head: The number of heads.

    Returns:
        A list of ``n_head`` matrices. Each has one row per position, and
        each row has ``head_dim`` nodes.

    Raises:
        ValueError: If ``m`` is empty.
    """
    if len(m) == 0:
        raise ValueError("m is empty")
    head_dim: int = len(m[0]) // n_head

    return [
        [row[h * head_dim : (h + 1) * head_dim] for row in m] for h in range(n_head)
    ]


def attention(
    *,
    x: Matrix,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    n_head: int,
) -> Matrix:
    """Return causal multi-head self-attention over the sequence ``x``.

    Args:
        x: One vector per position. Each vector has ``n_embd`` nodes.
        wq: The query weights, ``n_embd x n_embd``.
        wk: The key weights, ``n_embd x n_embd``.
        wv: The value weights, ``n_embd x n_embd``.
        wo: The output weights, ``n_embd x n_embd``.
        n_head: The number of heads. It must divide ``n_embd``.

    Returns:
        One vector per position, each with ``n_embd`` nodes.

    Raises:
        ValueError: If ``x`` is empty, or if ``n_head`` does not divide
            ``n_embd``.
    """
    if len(x) == 0:
        raise ValueError("x is empty")
    n_embd: int = len(x[0])
    if n_embd % n_head != 0:
        raise ValueError(f"n_head={n_head} does not divide n_embd={n_embd}")
    head_dim: int = n_embd // n_head
    scale: float = head_dim**-0.5

    q: list[Matrix] = split_heads(m=[linear(x=xi, w=wq) for xi in x], n_head=n_head)
    k: list[Matrix] = split_heads(m=[linear(x=xi, w=wk) for xi in x], n_head=n_head)
    v: list[Matrix] = split_heads(m=[linear(x=xi, w=wv) for xi in x], n_head=n_head)

    outputs: Matrix = []
    for i in range(len(x)):
        joined: Vector = []
        for h in range(n_head):
            joined.extend(
                attend(q=q[h][i], keys=k[h][: i + 1], values=v[h][: i + 1], scale=scale)
            )
        outputs.append(linear(x=joined, w=wo))

    return outputs
