"""Unit tests for causal multi-head self-attention.

Each test group maps to a section of docs/ATTENTION.md.

The functions are imported by name. Until all three exist, the import fails
and pytest reports one collection error. Once they exist, each test fails
or passes on its own.
"""

import math
import random

import pytest

from gptlab.attention import attend, attention, split_heads
from gptlab.autograd import Node
from gptlab.layers import Matrix, Vector


def _nodes(*, values: list[float]) -> Vector:
    """Return one leaf Node for each float."""
    return [Node(forward_val=v) for v in values]


def _matrix(*, values: list[list[float]]) -> Matrix:
    """Return one leaf Node for each float, in rows."""
    return [_nodes(values=row) for row in values]


def _values(*, nodes: Vector) -> list[float]:
    """Return the forward value of each Node."""
    return [n.forward_val for n in nodes]


def _random_floats(*, rng: random.Random, n: int) -> list[float]:
    """Return ``n`` floats drawn from a standard normal."""
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _random_matrix(*, rng: random.Random, n: int) -> list[list[float]]:
    """Return an ``n x n`` matrix of floats drawn from a standard normal."""
    return [_random_floats(rng=rng, n=n) for _ in range(n)]


# --- Float reference ---------------------------------------------------------
#
# A second attention, written with floats and ``math`` only. It shares no code
# with the module under test. The forward tests compare against it where a
# hand computation is too long, and the gradient check perturbs it.


def _float_linear(*, x: list[float], w: list[list[float]]) -> list[float]:
    """Return the dot product of ``x`` with each row of ``w``."""
    return [sum(xi * wi for xi, wi in zip(x, row, strict=True)) for row in w]


def _float_softmax(*, scores: list[float]) -> list[float]:
    """Return the softmax of the scores, with no shift."""
    exps = [math.exp(s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _float_attend(
    *,
    q: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    scale: float,
) -> list[float]:
    """Return one query's weighted sum of the values, with floats only."""
    scores = [sum(a * b for a, b in zip(q, key, strict=True)) * scale for key in keys]
    weights = _float_softmax(scores=scores)
    return [
        sum(w * value[t] for w, value in zip(weights, values, strict=True))
        for t in range(len(values[0]))
    ]


def _float_attention(
    *,
    x: list[list[float]],
    wq: list[list[float]],
    wk: list[list[float]],
    wv: list[list[float]],
    wo: list[list[float]],
    n_head: int,
) -> list[list[float]]:
    """Return causal multi-head self-attention over ``x``, with floats only."""
    n_embd = len(x[0])
    head_dim = n_embd // n_head
    q = [_float_linear(x=xi, w=wq) for xi in x]
    k = [_float_linear(x=xi, w=wk) for xi in x]
    v = [_float_linear(x=xi, w=wv) for xi in x]

    outputs = []
    for i in range(len(x)):
        heads: list[float] = []
        for h in range(n_head):
            lo, hi = h * head_dim, (h + 1) * head_dim
            scores = [
                sum(a * b for a, b in zip(q[i][lo:hi], k[j][lo:hi], strict=True))
                / math.sqrt(head_dim)
                for j in range(i + 1)
            ]
            weights = _float_softmax(scores=scores)
            heads += [
                sum(weights[j] * v[j][lo + t] for j in range(i + 1))
                for t in range(head_dim)
            ]
        outputs.append(_float_linear(x=heads, w=wo))
    return outputs


# --- Shared small case --------------------------------------------------------
#
# 3 positions, n_embd = 4, 2 heads, so head_dim = 2. The weights are random
# but fixed by the seed, so each test sees the same numbers.

N_EMBD = 4
N_HEAD = 2
T = 3


def _case(*, seed: int = 0):
    """Return (x, wq, wk, wv, wo) as floats for the shared small case."""
    rng = random.Random(seed)
    x = [_random_floats(rng=rng, n=N_EMBD) for _ in range(T)]
    wq, wk, wv, wo = (_random_matrix(rng=rng, n=N_EMBD) for _ in range(4))
    return x, wq, wk, wv, wo


def _run(*, x, wq, wk, wv, wo, n_head: int = N_HEAD) -> Matrix:
    """Build Nodes from the floats and run the module under test."""
    return attention(
        x=[_nodes(values=xi) for xi in x],
        wq=_matrix(values=wq),
        wk=_matrix(values=wk),
        wv=_matrix(values=wv),
        wo=_matrix(values=wo),
        n_head=n_head,
    )


# --- Group 1: attend (Section 2) ---------------------------------------------
#
# One query, three keys, three values, head_dim = 2. The scale is 1.0 unless
# a test says otherwise, so the scores are plain dot products.

Q_VALS = [1.0, 0.0]
K_VALS = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
V_VALS = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def _attend(*, q, keys, values, scale: float = 1.0) -> Vector:
    """Build Nodes from the floats and run ``attend``."""
    return attend(
        q=_nodes(values=q),
        keys=_matrix(values=keys),
        values=_matrix(values=values),
        scale=scale,
    )


def test_attend_matches_hand_computed_case() -> None:
    # Scores are q . k: 1, 0, -1. Softmax of those is the weights. The output
    # is the weighted sum of the value rows.
    e = [math.exp(1.0), math.exp(0.0), math.exp(-1.0)]
    w = [ei / sum(e) for ei in e]
    expected = [
        w[0] * 1.0 + w[1] * 3.0 + w[2] * 5.0,
        w[0] * 2.0 + w[1] * 4.0 + w[2] * 6.0,
    ]
    out = _attend(q=Q_VALS, keys=K_VALS, values=V_VALS)
    assert _values(nodes=out) == pytest.approx(expected)


def test_attend_matches_float_reference() -> None:
    rng = random.Random(3)
    q = _random_floats(rng=rng, n=2)
    keys = [_random_floats(rng=rng, n=2) for _ in range(4)]
    values = [_random_floats(rng=rng, n=2) for _ in range(4)]
    scale = 2**-0.5
    out = _attend(q=q, keys=keys, values=values, scale=scale)
    expected = _float_attend(q=q, keys=keys, values=values, scale=scale)
    assert _values(nodes=out) == pytest.approx(expected)


def test_attend_output_length_is_value_width() -> None:
    assert len(_attend(q=Q_VALS, keys=K_VALS, values=V_VALS)) == 2


def test_attend_returns_nodes() -> None:
    out = _attend(q=Q_VALS, keys=K_VALS, values=V_VALS)
    assert all(isinstance(y, Node) for y in out)


def test_attend_with_one_key_returns_its_value() -> None:
    # Softmax of one score is exactly 1, so the output is the one value.
    out = _attend(q=Q_VALS, keys=K_VALS[:1], values=V_VALS[:1])
    assert _values(nodes=out) == V_VALS[0]


def test_attend_scale_changes_the_weights() -> None:
    # A larger scale sharpens the softmax, so the best-matching key, the
    # first one, pulls the output closer to its value.
    soft = _values(nodes=_attend(q=Q_VALS, keys=K_VALS, values=V_VALS, scale=0.1))
    sharp = _values(nodes=_attend(q=Q_VALS, keys=K_VALS, values=V_VALS, scale=10.0))
    assert abs(sharp[0] - V_VALS[0][0]) < abs(soft[0] - V_VALS[0][0])


def test_attend_rejects_a_key_of_the_wrong_length() -> None:
    with pytest.raises(ValueError):
        _attend(q=Q_VALS, keys=[[1.0, 0.0, 0.0]], values=V_VALS[:1])


def test_attend_rejects_unequal_key_and_value_counts() -> None:
    with pytest.raises(ValueError):
        _attend(q=Q_VALS, keys=K_VALS, values=V_VALS[:2])


# --- Group 2: split_heads (Section 3) ----------------------------------------

# 2 rows of 4 elements, 2 heads: head 0 holds elements 0 and 1, head 1 holds
# elements 2 and 3, of every row.
M_VALS = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
HEAD_VALS = [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]


def test_split_heads_matches_hand_computed_case() -> None:
    heads = split_heads(m=_matrix(values=M_VALS), n_head=2)
    assert [[_values(nodes=row) for row in head] for head in heads] == HEAD_VALS


def test_split_heads_shape_is_heads_by_rows_by_head_dim() -> None:
    heads = split_heads(m=_matrix(values=M_VALS), n_head=2)
    assert len(heads) == 2
    assert all(len(head) == 2 for head in heads)
    assert all(len(row) == 2 for head in heads for row in head)


def test_split_heads_makes_no_node() -> None:
    # Section 3: a slice holds the same node objects as the row it came from.
    m = _matrix(values=M_VALS)
    heads = split_heads(m=m, n_head=2)
    for h, head in enumerate(heads):
        for r, row in enumerate(head):
            for t, node in enumerate(row):
                assert node is m[r][h * 2 + t]


def test_split_heads_with_one_head_is_the_whole_matrix() -> None:
    m = _matrix(values=M_VALS)
    (head,) = split_heads(m=m, n_head=1)
    assert head == m


# --- Group 3: shape (Sections 4 and 6) ---------------------------------------


def test_output_has_one_vector_per_position() -> None:
    x, wq, wk, wv, wo = _case()
    out = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    assert len(out) == T


def test_each_output_vector_has_length_n_embd() -> None:
    x, wq, wk, wv, wo = _case()
    out = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    assert all(len(y) == N_EMBD for y in out)


def test_output_holds_nodes() -> None:
    x, wq, wk, wv, wo = _case()
    out = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    assert all(isinstance(yi, Node) for y in out for yi in y)


def test_forward_matches_float_reference() -> None:
    x, wq, wk, wv, wo = _case()
    expected = _float_attention(x=x, wq=wq, wk=wk, wv=wv, wo=wo, n_head=N_HEAD)
    out = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    for y, e in zip(out, expected, strict=True):
        assert _values(nodes=y) == pytest.approx(e)


def test_n_head_must_divide_n_embd() -> None:
    # n_embd = 4. 3 heads would give head_dim = 1 and drop one dimension.
    x, wq, wk, wv, wo = _case()
    with pytest.raises(ValueError):
        _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo, n_head=3)


# --- Group 4: one position (Section 6) ---------------------------------------
#
# With one position, softmax sees one score and returns 1. The weighted sum
# is v itself, so the output is wo @ (wv @ x). The result does not depend on
# wq or wk.


def test_one_position_is_wo_of_wv_of_x() -> None:
    x, wq, wk, wv, wo = _case()
    (y,) = _run(x=x[:1], wq=wq, wk=wk, wv=wv, wo=wo)
    expected = _float_linear(x=_float_linear(x=x[0], w=wv), w=wo)
    assert _values(nodes=y) == pytest.approx(expected)


def test_one_position_ignores_wq_and_wk() -> None:
    x, wq, wk, wv, wo = _case()
    _, wq2, wk2, _, _ = _case(seed=1)
    (y,) = _run(x=x[:1], wq=wq, wk=wk, wv=wv, wo=wo)
    (y2,) = _run(x=x[:1], wq=wq2, wk=wk2, wv=wv, wo=wo)
    assert _values(nodes=y) == pytest.approx(_values(nodes=y2))


# --- Group 5: causality (Section 5) ------------------------------------------
#
# Change the input at position 2. Positions 0 and 1 read nothing from it, so
# their outputs are the same floats, not just close. Position 2 does change.


def test_a_later_position_does_not_change_earlier_outputs() -> None:
    x, wq, wk, wv, wo = _case()
    changed = [row[:] for row in x]
    changed[2] = [v + 1.0 for v in changed[2]]

    base = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    out = _run(x=changed, wq=wq, wk=wk, wv=wv, wo=wo)

    assert _values(nodes=out[0]) == _values(nodes=base[0])
    assert _values(nodes=out[1]) == _values(nodes=base[1])
    assert _values(nodes=out[2]) != pytest.approx(_values(nodes=base[2]))


def test_an_earlier_position_does_change_later_outputs() -> None:
    x, wq, wk, wv, wo = _case()
    changed = [row[:] for row in x]
    changed[0] = [v + 1.0 for v in changed[0]]

    base = _run(x=x, wq=wq, wk=wk, wv=wv, wo=wo)
    out = _run(x=changed, wq=wq, wk=wk, wv=wv, wo=wo)

    assert _values(nodes=out[1]) != pytest.approx(_values(nodes=base[1]))
    assert _values(nodes=out[2]) != pytest.approx(_values(nodes=base[2]))


# --- Group 6: numeric gradient check (Section 7) -----------------------------
#
# Same method as layers test group 6: compare the autograd gradient of each
# input against a central finite difference, (f(v + h) - f(v - h)) / (2h).
#
# The case is 2 positions and 2 heads, n_embd = 4. Every float is an input:
# the 8 elements of x and the 64 elements of the four weight matrices. The
# loss is a weighted sum of all output elements with fixed, distinct
# coefficients, so that no gradient is zero by construction.

H = 1e-6
TOL = 1e-6
GRAD_T = 2
COEFFS = [0.3, -1.2, 2.0, 0.7, -0.4, 1.5, 0.9, -2.1]


def _unpack(*, v: list, t: int = GRAD_T, n: int = N_EMBD):
    """Split one flat list into (x, wq, wk, wv, wo)."""
    x = [v[i * n : (i + 1) * n] for i in range(t)]
    mats = []
    offset = t * n
    for _ in range(4):
        mats.append([v[offset + r * n : offset + (r + 1) * n] for r in range(n)])
        offset += n * n
    return x, *mats


def _weighted(*, outputs: list[list], coeffs: list[float]):
    """Return the weighted sum of every element of the outputs."""
    flat = [o for row in outputs for o in row]
    total = flat[0] * coeffs[0]
    for out, c in zip(flat[1:], coeffs[1:], strict=True):
        total = total + out * c
    return total


def _node_loss(*, v: Vector) -> Node:
    """Return the weighted loss of the module under test."""
    x, wq, wk, wv, wo = _unpack(v=v)
    out = attention(x=x, wq=wq, wk=wk, wv=wv, wo=wo, n_head=N_HEAD)
    return _weighted(outputs=out, coeffs=COEFFS)


def _float_loss(*, v: list[float]) -> float:
    """Return the same loss as ``_node_loss`` with floats only."""
    x, wq, wk, wv, wo = _unpack(v=v)
    out = _float_attention(x=x, wq=wq, wk=wk, wv=wv, wo=wo, n_head=N_HEAD)
    return _weighted(outputs=out, coeffs=COEFFS)


def test_backward_matches_finite_difference() -> None:
    rng = random.Random(2)
    inputs = _random_floats(rng=rng, n=GRAD_T * N_EMBD + 4 * N_EMBD * N_EMBD)

    nodes = _nodes(values=inputs)
    loss = _node_loss(v=nodes)
    assert isinstance(loss, Node)
    assert loss.forward_val == pytest.approx(_float_loss(v=inputs))
    loss.backward()

    for i, node in enumerate(nodes):
        up = inputs[:i] + [inputs[i] + H] + inputs[i + 1 :]
        down = inputs[:i] + [inputs[i] - H] + inputs[i + 1 :]
        numeric = (_float_loss(v=up) - _float_loss(v=down)) / (2 * H)
        assert node.gradient == pytest.approx(numeric, abs=TOL), f"input {i}"
