"""Unit tests for the two-layer feed-forward block.

Each test group maps to a section of docs/MLP.md.

The function is imported by name. Until it exists, the import fails and
pytest reports one collection error. Once it exists, each test fails or
passes on its own.
"""

import random

import pytest

from gptlab.autograd import Node
from gptlab.layers import Matrix, Vector
from gptlab.mlp import mlp


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


def _random_matrix(*, rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    """Return a ``rows x cols`` matrix of floats drawn from a standard normal."""
    return [_random_floats(rng=rng, n=cols) for _ in range(rows)]


# --- Float reference ---------------------------------------------------------
#
# A second MLP, written with floats only. It shares no code with the module
# under test. The forward tests compare against it where a hand computation
# is too long, and the gradient check perturbs it.


def _float_linear(*, x: list[float], w: list[list[float]]) -> list[float]:
    """Return the dot product of ``x`` with each row of ``w``."""
    return [sum(xi * wi for xi, wi in zip(x, row, strict=True)) for row in w]


def _float_mlp(*, x: list[float], fc1: list[list[float]], fc2: list[list[float]]) -> list[float]:
    """Return ``fc2 @ relu(fc1 @ x)`` with floats only."""
    hidden = [max(0.0, h) for h in _float_linear(x=x, w=fc1)]
    return _float_linear(x=hidden, w=fc2)


# --- Shared small case --------------------------------------------------------
#
# n_embd = 2, so the hidden size is 4. The weights are random but fixed by
# the seed, so each test sees the same numbers.

N_EMBD = 2
N_HIDDEN = 4 * N_EMBD


def _case(*, seed: int = 0):
    """Return (x, fc1, fc2) as floats for the shared small case."""
    rng = random.Random(seed)
    x = _random_floats(rng=rng, n=N_EMBD)
    fc1 = _random_matrix(rng=rng, rows=N_HIDDEN, cols=N_EMBD)
    fc2 = _random_matrix(rng=rng, rows=N_EMBD, cols=N_HIDDEN)
    return x, fc1, fc2


def _run(*, x, fc1, fc2) -> Vector:
    """Build Nodes from the floats and run the module under test."""
    return mlp(x=_nodes(values=x), fc1=_matrix(values=fc1), fc2=_matrix(values=fc2))


# --- Group 1: shape (Section 2) ----------------------------------------------


def test_output_has_length_n_embd() -> None:
    x, fc1, fc2 = _case()
    assert len(_run(x=x, fc1=fc1, fc2=fc2)) == N_EMBD


def test_output_holds_nodes() -> None:
    x, fc1, fc2 = _case()
    assert all(isinstance(y, Node) for y in _run(x=x, fc1=fc1, fc2=fc2))


def test_rejects_fc1_row_of_wrong_length() -> None:
    # A row of fc1 must have the same length as x.
    x, fc1, fc2 = _case()
    bad = [row[:] for row in fc1]
    bad[0] = bad[0] + [1.0]
    with pytest.raises(ValueError):
        _run(x=x, fc1=bad, fc2=fc2)


def test_rejects_fc2_row_of_wrong_length() -> None:
    # A row of fc2 must have the same length as the hidden vector.
    x, fc1, fc2 = _case()
    bad = [row[:] for row in fc2]
    bad[0] = bad[0][:-1]
    with pytest.raises(ValueError):
        _run(x=x, fc1=fc1, fc2=bad)


# --- Group 2: hand case (Section 2) ------------------------------------------
#
# x = [1, -1]. The four rows of fc1 give pre-activations 1, -1, 1, 2. The
# second unit is negative and ReLU cuts it, so hidden = [1, 0, 1, 2].
#
# The first row of fc2 sums every hidden unit. The second row of fc2 reads
# only the cut unit. So the output is [4, 0]. Without the ReLU it would be
# [3, -1], and both elements would differ.

X_VALS = [1.0, -1.0]
FC1_VALS = [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, -1.0]]
FC2_VALS = [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
HAND_EXPECTED = [4.0, 0.0]
HAND_WITHOUT_RELU = [3.0, -1.0]


def test_hand_case_matches() -> None:
    out = _run(x=X_VALS, fc1=FC1_VALS, fc2=FC2_VALS)
    assert _values(nodes=out) == HAND_EXPECTED


def test_hand_case_cuts_the_negative_unit() -> None:
    # The hand case is built so that a missing ReLU changes both outputs.
    out = _values(nodes=_run(x=X_VALS, fc1=FC1_VALS, fc2=FC2_VALS))
    assert out[0] != HAND_WITHOUT_RELU[0]
    assert out[1] != HAND_WITHOUT_RELU[1]


def test_cut_unit_passes_no_gradient_to_its_weights() -> None:
    # The second row of fc1 feeds only the cut unit. ReLU sends no gradient
    # through a unit that is at or below zero, so those weights get zero.
    x = _nodes(values=X_VALS)
    fc1 = _matrix(values=FC1_VALS)
    fc2 = _matrix(values=FC2_VALS)
    y = mlp(x=x, fc1=fc1, fc2=fc2)
    (y[0] + y[1]).backward()
    assert [w.gradient for w in fc1[1]] == [0.0, 0.0]


def test_live_unit_passes_gradient_to_its_weights() -> None:
    # The first row of fc1 feeds a unit with pre-activation 1. Its gradient
    # is x scaled by the weight fc2 gives that unit: fc2[0][0] + fc2[1][0]
    # = 1, so the gradient is x itself.
    x = _nodes(values=X_VALS)
    fc1 = _matrix(values=FC1_VALS)
    fc2 = _matrix(values=FC2_VALS)
    y = mlp(x=x, fc1=fc1, fc2=fc2)
    (y[0] + y[1]).backward()
    assert [w.gradient for w in fc1[0]] == pytest.approx(X_VALS)


# --- Group 3: float reference (Section 2) ------------------------------------


def test_forward_matches_float_reference() -> None:
    x, fc1, fc2 = _case()
    expected = _float_mlp(x=x, fc1=fc1, fc2=fc2)
    assert _values(nodes=_run(x=x, fc1=fc1, fc2=fc2)) == pytest.approx(expected)


def test_forward_matches_float_reference_on_a_second_seed() -> None:
    x, fc1, fc2 = _case(seed=1)
    expected = _float_mlp(x=x, fc1=fc1, fc2=fc2)
    assert _values(nodes=_run(x=x, fc1=fc1, fc2=fc2)) == pytest.approx(expected)


# --- Group 4: numeric gradient check (AUTOGRAD.md section 10) ----------------
#
# Same method as layers test group 6: compare the autograd gradient of each
# input against a central finite difference, (f(v + h) - f(v - h)) / (2h).
#
# Every float is an input: the 2 elements of x, the 8 of fc1, and the 8 of
# fc2. The loss is a weighted sum of the output elements with fixed,
# distinct coefficients, so that no gradient is zero by construction.
#
# ReLU has a kink at zero. A finite difference that straddles the kink does
# not match the autograd gradient. The test checks that every pre-activation
# of the seeded case is well away from zero before it compares anything.

H = 1e-6
TOL = 1e-6
KINK_MARGIN = 1e-3
COEFFS = [0.3, -1.2]


def _unpack(*, v: list, n: int = N_EMBD):
    """Split one flat list into (x, fc1, fc2)."""
    x = v[:n]
    offset = n
    fc1 = [v[offset + r * n : offset + (r + 1) * n] for r in range(4 * n)]
    offset += 4 * n * n
    fc2 = [v[offset + r * 4 * n : offset + (r + 1) * 4 * n] for r in range(n)]
    return x, fc1, fc2


def _weighted(*, outputs: list, coeffs: list[float]):
    """Return the weighted sum of the outputs."""
    total = outputs[0] * coeffs[0]
    for out, c in zip(outputs[1:], coeffs[1:], strict=True):
        total = total + out * c
    return total


def _node_loss(*, v: Vector) -> Node:
    """Return the weighted loss of the module under test."""
    x, fc1, fc2 = _unpack(v=v)
    return _weighted(outputs=mlp(x=x, fc1=fc1, fc2=fc2), coeffs=COEFFS)


def _float_loss(*, v: list[float]) -> float:
    """Return the same loss as ``_node_loss`` with floats only."""
    x, fc1, fc2 = _unpack(v=v)
    return _weighted(outputs=_float_mlp(x=x, fc1=fc1, fc2=fc2), coeffs=COEFFS)


def test_backward_matches_finite_difference() -> None:
    rng = random.Random(2)
    inputs = _random_floats(rng=rng, n=N_EMBD + 4 * N_EMBD * N_EMBD + N_EMBD * 4 * N_EMBD)

    x, fc1, _ = _unpack(v=inputs)
    assert all(abs(h) > KINK_MARGIN for h in _float_linear(x=x, w=fc1))

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
