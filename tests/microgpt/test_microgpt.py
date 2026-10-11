"""Unit tests for the scalar autograd Node class.

Each test group maps to a section of docs/BACKPROPAGATION.md.
"""

import pytest

from microgpt.microgpt import Node

# --- Group 1: leaf construction (Section 2) ----------------------------------


def test_leaf_stores_forward_value() -> None:
    assert Node(3.0).forward_val == 3.0


def test_leaf_has_no_parents() -> None:
    assert Node(3.0).parents == ()


def test_leaf_gradient_starts_at_zero() -> None:
    assert Node(3.0).gradient == 0.0


def test_leaf_backward_is_a_safe_no_op() -> None:
    leaf = Node(3.0)
    leaf.backward_func(1.0)
    assert leaf.gradient == 0.0


# --- Group 2: forward values for every operator (Section 9) ------------------

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


# --- Group 3: graph construction (Sections 3 and 9) --------------------------


def test_add_records_both_operands_as_parents() -> None:
    a, b = Node(1.0), Node(2.0)
    assert (a + b).parents == (a, b)


def test_mul_records_both_operands_as_parents() -> None:
    a, b = Node(1.0), Node(2.0)
    assert (a * b).parents == (a, b)


def test_pow_records_only_the_base_as_parent() -> None:
    a = Node(2.0)
    assert (a**3).parents == (a,)


@pytest.mark.parametrize(
    "expr",
    [lambda a: a + 2.0, lambda a: a * 2.0],
    ids=["add", "mul"],
)
def test_float_operand_is_wrapped_as_a_leaf(expr) -> None:
    a = Node(1.0)
    result = expr(a)
    assert len(result.parents) == 2
    wrapped = result.parents[1]
    assert isinstance(wrapped, Node)
    assert wrapped.forward_val == 2.0
    assert wrapped.parents == ()


# --- Group 4: local derivatives in isolation (Section 4) ---------------------

SEED = 0.25  # pre-existing gradient on the parents; must survive the call
G = 2.5  # incoming gradient dL/dc; must be multiplied, not ignored


def _seeded(*nodes: Node) -> None:
    for n in nodes:
        n.gradient = SEED


def test_add_backward_passes_gradient_through_unchanged() -> None:
    a, b = Node(3.0), Node(-7.0)
    c = a + b
    _seeded(a, b)
    c.backward_func(G)
    assert a.gradient == pytest.approx(SEED + G)
    assert b.gradient == pytest.approx(SEED + G)


def test_mul_backward_swaps_operand_values() -> None:
    a, b = Node(3.0), Node(-7.0)
    c = a * b
    _seeded(a, b)
    c.backward_func(G)
    assert a.gradient == pytest.approx(SEED + G * -7.0)
    assert b.gradient == pytest.approx(SEED + G * 3.0)


@pytest.mark.parametrize(
    ("base", "n"),
    [(3.0, 2), (3.0, -1), (4.0, 0.5), (2.0, 0)],
    ids=["square", "reciprocal", "sqrt", "zeroth"],
)
def test_pow_backward_uses_power_rule(base: float, n: float) -> None:
    a = Node(base)
    c = a**n
    _seeded(a)
    c.backward_func(G)
    assert a.gradient == pytest.approx(SEED + G * n * base ** (n - 1))


# --- Group 5: accumulation across consumers (Section 5) ----------------------


def test_node_used_twice_accumulates_both_contributions() -> None:
    w, x, y = Node(2.0), Node(3.0), Node(5.0)
    c = w * x
    d = w * y
    c.backward_func(1.0)
    d.backward_func(1.0)
    assert w.gradient == pytest.approx(3.0 + 5.0)


def test_node_multiplied_by_itself_accumulates_both_sides() -> None:
    w = Node(3.0)
    c = w * w
    c.backward_func(1.0)
    assert w.gradient == pytest.approx(2 * 3.0)
