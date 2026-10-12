"""Unit tests for the layer functions: linear, softmax, rmsnorm.

Each test group maps to a section of docs/LAYERS.md.

The module is imported as a whole, not by name. A function that does not
exist yet fails only its own tests, so the file works as a TDD target while
the module is built one function at a time.
"""

import math

import pytest

from gptlab import layers
from gptlab.autograd import Node


def _nodes(values: list[float]) -> list[Node]:
    return [Node(v) for v in values]


def _values(nodes: list[Node]) -> list[float]:
    return [n.forward_val for n in nodes]


# --- Group 1: linear (Section 3) ---------------------------------------------

# x has 3 inputs, w has 2 rows. Expected values are computed by hand:
#   y[0] = 1*1 + 2*0   + 3*(-1)  = -2.0
#   y[1] = 1*2 + 2*1   + 3*0.5   =  5.5
X_VALS = [1.0, 2.0, 3.0]
W_VALS = [[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]]
Y_VALS = [-2.0, 5.5]


def test_linear_matches_hand_computed_2x3_case() -> None:
    x = _nodes(X_VALS)
    w = [_nodes(row) for row in W_VALS]
    assert _values(layers.linear(x, w)) == pytest.approx(Y_VALS)


def test_linear_output_length_is_number_of_rows() -> None:
    x = _nodes([1.0, 2.0])
    w = [_nodes([1.0, 1.0]) for _ in range(5)]
    assert len(layers.linear(x, w)) == 5


def test_linear_returns_nodes() -> None:
    x = _nodes([1.0, 2.0])
    w = [_nodes([3.0, 4.0])]
    assert all(isinstance(y, Node) for y in layers.linear(x, w))


def test_linear_rejects_a_row_with_the_wrong_length() -> None:
    # zip without strict=True would truncate and return a wrong number.
    x = _nodes([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        layers.linear(x, [_nodes([1.0, 1.0])])
    with pytest.raises(ValueError):
        layers.linear(x, [_nodes([1.0, 1.0, 1.0, 1.0])])


def test_linear_graph_holds_only_the_named_operations() -> None:
    # y = x0 * w0 + x1 * w1 is 4 leaves, 2 products, 1 sum: 7 nodes.
    # Builtin sum would add a leaf Node(0) and one more sum: 9 nodes.
    x = _nodes([1.0, 2.0])
    w = [_nodes([3.0, 4.0])]
    (y,) = layers.linear(x, w)
    assert len(y._topological_order()) == 7


# --- Group 2: softmax (Section 4) --------------------------------------------


def test_softmax_matches_hand_computed_case() -> None:
    # exp(0) = 1 and exp(log 2) = 2, so the probabilities are 1/3 and 2/3.
    p = layers.softmax(_nodes([0.0, math.log(2.0)]))
    assert _values(p) == pytest.approx([1 / 3, 2 / 3])


def test_softmax_sums_to_one() -> None:
    p = layers.softmax(_nodes([0.5, -1.0, 2.0, 0.0]))
    assert sum(_values(p)) == pytest.approx(1.0)


def test_softmax_is_unchanged_by_a_constant_shift() -> None:
    logits = [0.5, -1.0, 2.0]
    base = _values(layers.softmax(_nodes(logits)))
    shifted = _values(layers.softmax(_nodes([v + 100.0 for v in logits])))
    assert shifted == pytest.approx(base)


def test_softmax_does_not_overflow_on_large_logits() -> None:
    # math.exp(1000.0) raises OverflowError without the max shift.
    p = layers.softmax(_nodes([1000.0, 1000.0, 1001.0]))
    assert sum(_values(p)) == pytest.approx(1.0)
    assert all(math.isfinite(v) for v in _values(p))


def test_softmax_output_length_matches_input() -> None:
    assert len(layers.softmax(_nodes([1.0, 2.0, 3.0]))) == 3


def test_softmax_returns_nodes() -> None:
    assert all(isinstance(p, Node) for p in layers.softmax(_nodes([1.0, 2.0])))


# --- Group 3: rmsnorm (Section 5) --------------------------------------------


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def test_rmsnorm_matches_hand_computed_case() -> None:
    # mean(x ** 2) = (9 + 16) / 2 = 12.5. Each element is divided by
    # sqrt(12.5 + eps).
    eps = 1e-5
    y = layers.rmsnorm(_nodes([3.0, 4.0]), eps=eps)
    scale = (12.5 + eps) ** -0.5
    assert _values(y) == pytest.approx([3.0 * scale, 4.0 * scale])


def test_rmsnorm_output_has_unit_rms() -> None:
    y = layers.rmsnorm(_nodes([1.5, -0.5, 2.0, 0.25]))
    assert _rms(_values(y)) == pytest.approx(1.0, abs=1e-5)


def test_rmsnorm_is_unchanged_by_a_positive_scale() -> None:
    x = [1.5, -0.5, 2.0]
    base = _values(layers.rmsnorm(_nodes(x)))
    scaled = _values(layers.rmsnorm(_nodes([v * 10.0 for v in x])))
    assert scaled == pytest.approx(base, abs=1e-5)


def test_rmsnorm_of_zero_vector_does_not_divide_by_zero() -> None:
    y = layers.rmsnorm(_nodes([0.0, 0.0, 0.0]))
    assert _values(y) == [0.0, 0.0, 0.0]


def test_rmsnorm_of_zero_vector_has_finite_gradients() -> None:
    # Section 5: with eps inside the root, the scale is eps ** -0.5 and each
    # x[i] receives exactly coefficient * scale. Nothing is inf or nan.
    eps = 1e-5
    x = _nodes([0.0, 0.0, 0.0])
    loss = _weighted(layers.rmsnorm(x, eps=eps), COEFFS)
    loss.backward()
    scale = eps**-0.5
    for node, c in zip(x, COEFFS, strict=True):
        assert math.isfinite(node.gradient)
        assert node.gradient == pytest.approx(c * scale)


def test_rmsnorm_output_length_matches_input() -> None:
    assert len(layers.rmsnorm(_nodes([1.0, 2.0, 3.0]))) == 3


def test_rmsnorm_returns_nodes() -> None:
    assert all(isinstance(y, Node) for y in layers.rmsnorm(_nodes([1.0, 2.0])))


# --- Group 4: numeric gradient check (Section 6) -----------------------------
#
# Same method as autograd test group 7: compare the autograd gradient of each
# input against a central finite difference, (f(v + h) - f(v - h)) / (2h).
#
# Each layer returns a vector. A gradient check needs a scalar. The loss is a
# weighted sum of the outputs with fixed, distinct coefficients. A plain sum
# would not do: the plain sum of a softmax is always 1, so every gradient
# through it would be zero and the test would prove nothing.
#
# The float reference for each layer is written here with math functions. It
# does not call the layer under test, so the two computations share no code.

H = 1e-6
TOL = 1e-6
COEFFS = [0.3, -1.2, 2.0]


def _weighted(outputs: list, coeffs: list[float]):
    total = outputs[0] * coeffs[0]
    for out, c in zip(outputs[1:], coeffs[1:]):
        total = total + out * c
    return total


# Each case is (inputs, node_loss, float_loss).
#   inputs:     the flat list of float inputs to perturb, one at a time.
#   node_loss:  builds the loss from Nodes with the layer under test.
#   float_loss: the same loss from floats, with no Node and no layer code.


def _linear_node_loss(v: list[Node]) -> Node:
    x, w = v[:3], [v[3:6], v[6:9]]
    return _weighted(layers.linear(x, w), COEFFS[:2])


def _linear_float_loss(v: list[float]) -> float:
    x, w = v[:3], [v[3:6], v[6:9]]
    y = [sum(xi * wi for xi, wi in zip(x, row)) for row in w]
    return _weighted(y, COEFFS[:2])


def _softmax_node_loss(v: list[Node]) -> Node:
    return _weighted(layers.softmax(v), COEFFS)


def _softmax_float_loss(v: list[float]) -> float:
    exps = [math.exp(vi) for vi in v]
    total = sum(exps)
    return _weighted([e / total for e in exps], COEFFS)


def _rmsnorm_node_loss(v: list[Node]) -> Node:
    return _weighted(layers.rmsnorm(v), COEFFS)


def _rmsnorm_float_loss(v: list[float]) -> float:
    scale = (sum(vi * vi for vi in v) / len(v) + 1e-5) ** -0.5
    return _weighted([vi * scale for vi in v], COEFFS)


@pytest.mark.parametrize(
    ("inputs", "node_loss", "float_loss"),
    [
        (
            [1.5, -0.5, 2.0, 0.2, -1.0, 0.7, 1.1, 0.4, -0.3],
            _linear_node_loss,
            _linear_float_loss,
        ),
        ([0.5, -1.0, 2.0], _softmax_node_loss, _softmax_float_loss),
        ([1.5, -0.5, 2.0], _rmsnorm_node_loss, _rmsnorm_float_loss),
    ],
    ids=["linear", "softmax", "rmsnorm"],
)
def test_backward_matches_finite_difference(inputs, node_loss, float_loss) -> None:
    nodes = _nodes(inputs)
    loss = node_loss(nodes)
    assert isinstance(loss, Node)
    assert loss.forward_val == pytest.approx(float_loss(inputs))
    loss.backward()

    for i, node in enumerate(nodes):
        up = inputs[:i] + [inputs[i] + H] + inputs[i + 1 :]
        down = inputs[:i] + [inputs[i] - H] + inputs[i + 1 :]
        numeric = (float_loss(up) - float_loss(down)) / (2 * H)
        assert node.gradient == pytest.approx(numeric, abs=TOL), f"input {i}"
