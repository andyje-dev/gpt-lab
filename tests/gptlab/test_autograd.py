"""Unit tests for the scalar autograd Node class.

Each test group maps to a section of docs/AUTOGRAD.md.
"""

import math

import pytest

from gptlab.autograd import Node

# --- Group 1: leaf construction (Section 2) ----------------------------------


def test_leaf_stores_forward_value() -> None:
    assert Node(3.0).forward_val == 3.0


def test_leaf_has_no_children() -> None:
    assert Node(3.0).children == ()


def test_leaf_gradient_starts_at_zero() -> None:
    assert Node(3.0).gradient == 0.0


def test_leaf_propagate_is_a_safe_no_op() -> None:
    leaf = Node(3.0)
    leaf.propagate(1.0)
    assert leaf.gradient == 0.0


# --- Group 2: forward values for every operator (Section 7) ------------------

A, B = 1.5, -4.0


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # primitives
        (lambda a, b: a + b, A + B),
        (lambda a, b: a * b, A * B),
        (lambda a, _: a**2, A**2),
        (lambda a, _: a**-1, A**-1),
        (lambda a, _: a**0.5, A**0.5),
        (lambda a, _: a.exp(), math.exp(A)),
        (lambda a, _: a.log(), math.log(A)),
        (lambda a, _: a.relu(), A),
        (lambda _, b: b.relu(), 0.0),
        # node with float, both orders
        (lambda a, _: a + B, A + B),
        (lambda a, _: B + a, B + A),
        (lambda a, _: a * B, A * B),
        (lambda a, _: B * a, B * A),
        (lambda a, _: a - B, A - B),
        (lambda a, _: B - a, B - A),
        (lambda a, _: a / B, A / B),
        (lambda a, _: B / a, B / A),
        # derived, node with node
        (lambda a, _: -a, -A),
        (lambda a, b: a - b, A - B),
        (lambda a, b: a / b, A / B),
    ],
    ids=[
        "add",
        "mul",
        "pow_int",
        "pow_neg",
        "pow_frac",
        "exp",
        "log",
        "relu_positive",
        "relu_negative",
        "add_float",
        "radd_float",
        "mul_float",
        "rmul_float",
        "sub_float",
        "rsub_float",
        "truediv_float",
        "rtruediv_float",
        "neg",
        "sub",
        "truediv",
    ],
)
def test_forward_value(expr, expected: float) -> None:
    result = expr(Node(A), Node(B))
    assert isinstance(result, Node)
    assert result.forward_val == pytest.approx(expected)


# --- Group 3: graph construction (Sections 3 and 6) --------------------------


def test_add_records_both_operands_as_children() -> None:
    a, b = Node(1.0), Node(2.0)
    assert (a + b).children == (a, b)


def test_mul_records_both_operands_as_children() -> None:
    a, b = Node(1.0), Node(2.0)
    assert (a * b).children == (a, b)


def test_pow_records_only_the_base_as_child() -> None:
    a = Node(2.0)
    assert (a**3).children == (a,)


@pytest.mark.parametrize(
    "expr",
    [lambda a: a.exp(), lambda a: a.log(), lambda a: a.relu()],
    ids=["exp", "log", "relu"],
)
def test_unary_primitive_records_only_its_input_as_child(expr) -> None:
    a = Node(2.0)
    assert expr(a).children == (a,)


@pytest.mark.parametrize(
    "expr",
    [lambda a: a + 2.0, lambda a: a * 2.0],
    ids=["add", "mul"],
)
def test_float_operand_is_wrapped_as_a_leaf(expr) -> None:
    a = Node(1.0)
    result = expr(a)
    assert len(result.children) == 2
    wrapped = result.children[1]
    assert isinstance(wrapped, Node)
    assert wrapped.forward_val == 2.0
    assert wrapped.children == ()


# --- Group 4: local derivatives in isolation (Section 6) ---------------------

SEED = 0.25  # pre-existing gradient on the children; must survive the call
G = 2.5  # incoming gradient dL/dc; must be multiplied, not ignored


def _seeded(*nodes: Node) -> None:
    for n in nodes:
        n.gradient = SEED


def test_add_propagate_passes_gradient_through_unchanged() -> None:
    a, b = Node(3.0), Node(-7.0)
    c = a + b
    _seeded(a, b)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G)
    assert b.gradient == pytest.approx(SEED + G)


def test_mul_propagate_swaps_operand_values() -> None:
    a, b = Node(3.0), Node(-7.0)
    c = a * b
    _seeded(a, b)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G * -7.0)
    assert b.gradient == pytest.approx(SEED + G * 3.0)


@pytest.mark.parametrize(
    ("base", "n"),
    [(3.0, 2), (3.0, -1), (4.0, 0.5), (2.0, 0)],
    ids=["square", "reciprocal", "sqrt", "zeroth"],
)
def test_pow_propagate_uses_power_rule(base: float, n: float) -> None:
    a = Node(base)
    c = a**n
    _seeded(a)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G * n * base ** (n - 1))


def test_exp_propagate_reuses_its_own_forward_value() -> None:
    # d/da exp(a) = exp(a) = c. Nothing is recomputed.
    a = Node(3.0)
    c = a.exp()
    _seeded(a)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G * c.forward_val)


def test_log_propagate_uses_reciprocal_of_input() -> None:
    # d/da log(a) = 1 / a
    a = Node(3.0)
    c = a.log()
    _seeded(a)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G / 3.0)


@pytest.mark.parametrize(
    ("value", "slope"),
    [(3.0, 1.0), (-7.0, 0.0), (0.0, 0.0)],
    ids=["positive", "negative", "zero"],
)
def test_relu_propagate_passes_gradient_only_when_positive(
    value: float, slope: float
) -> None:
    # d/da max(0, a) = 1 if a > 0 else 0. At a == 0 we choose 0.
    a = Node(value)
    c = a.relu()
    _seeded(a)
    c.propagate(G)
    assert a.gradient == pytest.approx(SEED + G * slope)


# --- Group 5: accumulation across consumers (Section 5) ----------------------


def test_node_used_twice_accumulates_both_contributions() -> None:
    w, x, y = Node(2.0), Node(3.0), Node(5.0)
    c = w * x
    d = w * y
    c.propagate(1.0)
    d.propagate(1.0)
    assert w.gradient == pytest.approx(3.0 + 5.0)


def test_node_multiplied_by_itself_accumulates_both_sides() -> None:
    w = Node(3.0)
    c = w * w
    c.propagate(1.0)
    assert w.gradient == pytest.approx(2 * 3.0)


# --- Group 6: the full backward pass (Sections 8 and 9) ----------------------


def test_backward_on_a_leaf_sets_its_own_gradient_to_one() -> None:
    # dL/dL = 1. This is the seed the walk starts from.
    a = Node(3.0)
    a.backward()
    assert a.gradient == pytest.approx(1.0)


def test_backward_reaches_the_children_of_one_operation() -> None:
    # c = a * b -> dc/da = b = -7, dc/db = a = 3
    a, b = Node(3.0), Node(-7.0)
    c = a * b
    c.backward()
    assert c.gradient == pytest.approx(1.0)
    assert a.gradient == pytest.approx(-7.0)
    assert b.gradient == pytest.approx(3.0)


def test_backward_chains_through_two_levels() -> None:
    # d = (a * b) + 1 -> dd/dc = 1, dc/da = b = 3, so dd/da = 3 and dd/db = 2.
    # c must run before a and b receive anything; if the order were wrong,
    # a and b would be visited while c.gradient is still 0.
    a, b = Node(2.0), Node(3.0)
    c = a * b
    d = c + 1.0
    d.backward()
    assert c.gradient == pytest.approx(1.0)
    assert a.gradient == pytest.approx(3.0)
    assert b.gradient == pytest.approx(2.0)


def test_backward_sums_both_paths_in_a_diamond() -> None:
    # L = a * b + a ** 2, with a = 3, b = 4
    # dL/da = b + 2a = 4 + 6 = 10, dL/db = a = 3
    # Path 1 (through a * b) gives 4. Path 2 (through a ** 2) gives 6.
    # Each path must contribute exactly once. A missing visited set or a
    # wrong order yields 4, 6, or something larger than 10.
    a, b = Node(3.0), Node(4.0)
    loss = a * b + a**2
    loss.backward()
    assert a.gradient == pytest.approx(10.0)
    assert b.gradient == pytest.approx(3.0)


def test_backward_through_a_node_consumed_by_a_deeper_node() -> None:
    # L = c * (c + 1) where c = a * 2, a = 3 -> c = 6
    # dL/dc = (c + 1) + c = 13, dc/da = 2, so dL/da = 26
    # c is used directly by L and also by the intermediate (c + 1).
    # The intermediate must finish before c sends anything to a.
    a = Node(3.0)
    c = a * 2.0
    loss = c * (c + 1.0)
    loss.backward()
    assert c.gradient == pytest.approx(13.0)
    assert a.gradient == pytest.approx(26.0)


def test_backward_through_shared_node_and_separate_wrapped_leaves() -> None:
    # L = (a + 1) * (a + 1), a = 2 -> L = 9
    # dL/da = 2 * (a + 1) = 6
    # The two `+ 1.0` calls make two different wrapped leaves, but a is shared.
    a = Node(2.0)
    loss = (a + 1.0) * (a + 1.0)
    loss.backward()
    assert a.gradient == pytest.approx(6.0)


def test_backward_visits_each_node_once() -> None:
    # Count how many times each node's backward function runs.
    # In a diamond, a is reachable by two paths but must appear in the
    # topological order exactly one time.
    a, b = Node(3.0), Node(4.0)
    loss = a * b + a**2
    calls: dict[int, int] = {}
    for node in (loss, *loss.children, a, b):
        original = node.propagate

        def counting(g: float, node: Node = node, original=original) -> None:
            calls[id(node)] = calls.get(id(node), 0) + 1
            original(g)

        node.propagate = counting
    loss.backward()
    assert all(count == 1 for count in calls.values()), calls


def test_backward_does_not_reset_gradients_between_calls() -> None:
    # Section 9: zeroing gradients is the job of the optimizer, not backward.
    # A second call on the same graph adds the same contributions again.
    a, b = Node(3.0), Node(-7.0)
    c = a * b
    c.backward()
    c.backward()
    assert a.gradient == pytest.approx(2 * -7.0)
    assert b.gradient == pytest.approx(2 * 3.0)


def test_backward_leaves_forward_values_frozen() -> None:
    a, b = Node(3.0), Node(-7.0)
    c = a * b + a**2
    before = [n.forward_val for n in (a, b, c)]
    c.backward()
    assert [n.forward_val for n in (a, b, c)] == before


def test_backward_survives_a_deep_chain() -> None:
    # A GPT forward pass is thousands of operations deep. A recursive DFS
    # would hit Python's default recursion limit (1000) long before that.
    a = Node(1.0)
    x = a
    depth = 5_000
    for _ in range(depth):
        x = x + 1.0
    x.backward()
    assert a.gradient == pytest.approx(1.0)


# --- Group 7: numeric gradient check (Section 10) ----------------------------
#
# Independent of the hand derivations above: compare the autograd result
# against a central finite difference, (f(x + h) - f(x - h)) / (2h).
# With h = 1e-6 the truncation error is O(h^2), so a 1e-6 tolerance is honest.

H = 1e-6
TOL = 1e-6


def _exp(x: float | Node) -> float | Node:
    return x.exp() if isinstance(x, Node) else math.exp(x)


def _log(x: float | Node) -> float | Node:
    return x.log() if isinstance(x, Node) else math.log(x)


def _relu(x: float | Node) -> float | Node:
    return x.relu() if isinstance(x, Node) else max(0.0, x)


def _loss_arithmetic(a: float | Node, b: float | Node) -> float | Node:
    # Uses every operator: +, *, **, -, /, unary -, and float on both sides.
    return (a * b - 2.0) / (a**2 + 1.0) + (3.0 - b) * -a + 1.0 / b


def _loss_nonlinear(a: float | Node, b: float | Node) -> float | Node:
    # A two-class softmax cross-entropy with a as the target logit, plus a
    # ReLU term. Uses exp, log, and relu in the shape the model will use them.
    # a * b is never near 0 for the sample points, so the ReLU kink is avoided.
    p = _exp(a) / (_exp(a) + _exp(b))
    return -_log(p) + _relu(a * b)


@pytest.mark.parametrize(
    "loss_fn",
    [_loss_arithmetic, _loss_nonlinear],
    ids=["arithmetic", "nonlinear"],
)
@pytest.mark.parametrize(
    ("a_val", "b_val"),
    [(1.5, -4.0), (0.3, 2.0), (-2.0, 0.7)],
)
def test_backward_matches_finite_difference(
    loss_fn, a_val: float, b_val: float
) -> None:
    a, b = Node(a_val), Node(b_val)
    loss = loss_fn(a, b)
    assert isinstance(loss, Node)
    loss.backward()

    numeric_da = (loss_fn(a_val + H, b_val) - loss_fn(a_val - H, b_val)) / (2 * H)
    numeric_db = (loss_fn(a_val, b_val + H) - loss_fn(a_val, b_val - H)) / (2 * H)

    assert a.gradient == pytest.approx(numeric_da, abs=TOL)
    assert b.gradient == pytest.approx(numeric_db, abs=TOL)
