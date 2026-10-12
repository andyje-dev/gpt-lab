"""Unit tests for the cross-entropy loss over a sequence.

Each test group maps to a section of docs/LOSS.md.

The functions are imported by name. Until both exist, the import fails and
pytest reports one collection error. Once they exist, each test fails or
passes on its own.
"""

import math
import random

import pytest

from gptlab.autograd import Node
from gptlab.layers import Matrix, Vector
from gptlab.loss import cross_entropy, sequence_loss


def _nodes(*, values: list[float]) -> Vector:
    """Return one leaf Node for each float."""
    return [Node(forward_val=v) for v in values]


def _matrix(*, values: list[list[float]]) -> Matrix:
    """Return one leaf Node for each float, in rows."""
    return [_nodes(values=row) for row in values]


def _gradients(*, nodes: Vector) -> list[float]:
    """Return the gradient of each Node."""
    return [n.gradient for n in nodes]


def _random_floats(*, rng: random.Random, n: int) -> list[float]:
    """Return ``n`` floats drawn from a standard normal."""
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _random_matrix(*, rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    """Return a ``rows x cols`` matrix of floats drawn from a standard normal."""
    return [_random_floats(rng=rng, n=cols) for _ in range(rows)]


# --- Float reference ---------------------------------------------------------
#
# A second loss, written with floats and math only. It shares no code with
# the module under test. The forward tests compare against it where a hand
# computation is too long, and the gradient check perturbs it.


def _float_softmax(*, logits: list[float]) -> list[float]:
    """Return the softmax of ``logits`` with floats only."""
    shift = max(logits)
    exponentials = [math.exp(v - shift) for v in logits]
    total = sum(exponentials)
    return [e / total for e in exponentials]


def _float_cross_entropy(*, logits: list[float], target: int) -> float:
    """Return ``-log(softmax(logits)[target])`` with floats only."""
    return -math.log(_float_softmax(logits=logits)[target])


def _float_sequence_loss(*, logits_per_position: list[list[float]], targets: list[int]) -> float:
    """Return the mean of the per-position losses with floats only."""
    losses = [
        _float_cross_entropy(logits=logits, target=target)
        for logits, target in zip(logits_per_position, targets, strict=True)
    ]
    return sum(losses) / len(losses)


# --- Shared small case --------------------------------------------------------
#
# A vocabulary of 4 tokens and a sequence of 3 positions. The logits are
# random but fixed by the seed, so each test sees the same numbers. The
# targets are fixed and distinct from the largest logit of each position in
# every seed used here, so no loss is near zero.

VOCAB_SIZE = 4
N_POSITIONS = 3
TARGETS = [2, 0, 3]


def _case(*, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Return (logits_per_position, targets) as floats for the shared case."""
    rng = random.Random(seed)
    logits_per_position = _random_matrix(rng=rng, rows=N_POSITIONS, cols=VOCAB_SIZE)
    return logits_per_position, TARGETS


# --- Group 1: shape (Sections 2 and 3) ---------------------------------------


def test_cross_entropy_returns_one_node() -> None:
    logits, targets = _case()
    loss = cross_entropy(logits=_nodes(values=logits[0]), target=targets[0])
    assert isinstance(loss, Node)


def test_cross_entropy_rejects_a_target_past_the_last_logit() -> None:
    # The guard raises before list indexing would.
    logits, _ = _case()
    with pytest.raises(ValueError):
        cross_entropy(logits=_nodes(values=logits[0]), target=VOCAB_SIZE)


def test_cross_entropy_rejects_a_negative_target() -> None:
    # Without the guard, a negative target indexes from the end and raises
    # nothing. This is the case the guard is for.
    logits, _ = _case()
    with pytest.raises(ValueError):
        cross_entropy(logits=_nodes(values=logits[0]), target=-1)


def test_sequence_loss_returns_one_node() -> None:
    logits, targets = _case()
    loss = sequence_loss(logits_per_position=_matrix(values=logits), targets=targets)
    assert isinstance(loss, Node)


def test_sequence_loss_rejects_more_targets_than_positions() -> None:
    logits, targets = _case()
    with pytest.raises(ValueError):
        sequence_loss(logits_per_position=_matrix(values=logits), targets=targets + [0])


def test_sequence_loss_rejects_fewer_targets_than_positions() -> None:
    logits, targets = _case()
    with pytest.raises(ValueError):
        sequence_loss(logits_per_position=_matrix(values=logits), targets=targets[:-1])


def test_sequence_loss_rejects_an_empty_sequence() -> None:
    # Two empty lists pass the strict zip. The seeded sum then reads
    # losses[0]. The guard says the sequence is empty.
    with pytest.raises(ValueError):
        sequence_loss(logits_per_position=[], targets=[])


# --- Group 2: hand case (Section 2) ------------------------------------------
#
# logits = [0, log 2, log 3]. The exponentials are 1, 2, 3 and they sum to
# 6, so the probabilities are 1/6, 2/6, 3/6.
#
# Target 2 gives -log(1/2) = log 2. Target 1 gives log 3. Target 0 gives
# log 6. A missing minus sign gives -log 2. A log of the logit with no
# softmax gives -log(log 3), which is negative. Every plausible mistake
# gives a different number.
#
# The gradient of the loss with respect to logit i is p_i - [i == target].
# For target 2 that is [1/6, 2/6, 3/6 - 1] = [1/6, 1/3, -1/2].

HAND_LOGITS = [0.0, math.log(2.0), math.log(3.0)]
HAND_LOSSES = [math.log(6.0), math.log(3.0), math.log(2.0)]
HAND_GRADIENT_AT_TARGET_2 = [1 / 6, 1 / 3, -1 / 2]


def test_hand_case_matches() -> None:
    loss = cross_entropy(logits=_nodes(values=HAND_LOGITS), target=2)
    assert loss.forward_val == pytest.approx(math.log(2.0))


def test_hand_case_gives_a_different_loss_for_each_target() -> None:
    losses = [cross_entropy(logits=_nodes(values=HAND_LOGITS), target=t).forward_val for t in range(3)]
    assert losses == pytest.approx(HAND_LOSSES)


def test_hand_case_gradient_is_probability_minus_one_hot() -> None:
    logits = _nodes(values=HAND_LOGITS)
    cross_entropy(logits=logits, target=2).backward()
    assert _gradients(nodes=logits) == pytest.approx(HAND_GRADIENT_AT_TARGET_2)


# --- Group 3: properties (Section 2) -----------------------------------------


def test_uniform_logits_give_log_vocab_size() -> None:
    # Every token has probability 1 / vocab_size, so the loss is log(vocab_size)
    # for any target and any shared value.
    logits = _nodes(values=[0.7] * 5)
    loss = cross_entropy(logits=logits, target=3)
    assert loss.forward_val == pytest.approx(math.log(5.0))


def test_loss_is_near_zero_when_the_target_logit_is_far_above_the_others() -> None:
    logits = _nodes(values=[20.0, 0.0, 0.0, 0.0])
    loss = cross_entropy(logits=logits, target=0)
    assert loss.forward_val == pytest.approx(0.0, abs=1e-6)


def test_loss_is_positive_for_finite_logits() -> None:
    # A probability is below one, so -log of it is above zero.
    logits, targets = _case()
    for row, target in zip(logits, targets, strict=True):
        loss = cross_entropy(logits=_nodes(values=row), target=target)
        assert loss.forward_val > 0.0


def test_loss_is_unchanged_by_a_constant_shift_of_the_logits() -> None:
    logits, targets = _case()
    base = cross_entropy(logits=_nodes(values=logits[0]), target=targets[0])
    shifted = cross_entropy(logits=_nodes(values=[v + 100.0 for v in logits[0]]), target=targets[0])
    assert shifted.forward_val == pytest.approx(base.forward_val)


def test_gradients_over_the_logits_sum_to_zero() -> None:
    # The loss does not change under a constant shift, so the sum of the
    # gradients is zero. Also p sums to one and the one-hot sums to one.
    logits, targets = _case()
    nodes = _nodes(values=logits[0])
    cross_entropy(logits=nodes, target=targets[0]).backward()
    assert sum(_gradients(nodes=nodes)) == pytest.approx(0.0, abs=1e-12)


def test_cross_entropy_matches_float_reference() -> None:
    logits, targets = _case()
    for row, target in zip(logits, targets, strict=True):
        expected = _float_cross_entropy(logits=row, target=target)
        loss = cross_entropy(logits=_nodes(values=row), target=target)
        assert loss.forward_val == pytest.approx(expected)


# --- Group 4: the mean (Section 3) -------------------------------------------


def test_two_positions_give_the_average_of_the_two_single_losses() -> None:
    logits, targets = _case()
    first = cross_entropy(logits=_nodes(values=logits[0]), target=targets[0])
    second = cross_entropy(logits=_nodes(values=logits[1]), target=targets[1])
    loss = sequence_loss(logits_per_position=_matrix(values=logits[:2]), targets=targets[:2])
    expected = (first.forward_val + second.forward_val) / 2
    assert loss.forward_val == pytest.approx(expected)


def test_one_position_equals_cross_entropy() -> None:
    logits, targets = _case()
    single = cross_entropy(logits=_nodes(values=logits[0]), target=targets[0])
    loss = sequence_loss(logits_per_position=_matrix(values=logits[:1]), targets=targets[:1])
    assert loss.forward_val == pytest.approx(single.forward_val)


def test_repeating_a_position_does_not_change_the_loss() -> None:
    # A sum would grow with the number of copies. A mean does not.
    logits, targets = _case()
    once = sequence_loss(logits_per_position=_matrix(values=logits[:1]), targets=targets[:1])
    four = sequence_loss(logits_per_position=_matrix(values=logits[:1] * 4), targets=targets[:1] * 4)
    assert four.forward_val == pytest.approx(once.forward_val)


def test_each_position_gradient_is_its_own_divided_by_the_count() -> None:
    # The mean scales every per-position gradient by 1 / n_positions. The
    # logits of one position do not change the gradient of another.
    logits, targets = _case()
    rows = _matrix(values=logits)
    sequence_loss(logits_per_position=rows, targets=targets).backward()
    for row, floats, target in zip(rows, logits, targets, strict=True):
        alone = _nodes(values=floats)
        cross_entropy(logits=alone, target=target).backward()
        expected = [g / N_POSITIONS for g in _gradients(nodes=alone)]
        assert _gradients(nodes=row) == pytest.approx(expected)


def test_sequence_loss_matches_float_reference() -> None:
    logits, targets = _case()
    expected = _float_sequence_loss(logits_per_position=logits, targets=targets)
    loss = sequence_loss(logits_per_position=_matrix(values=logits), targets=targets)
    assert loss.forward_val == pytest.approx(expected)


def test_sequence_loss_matches_float_reference_on_a_second_seed() -> None:
    logits, targets = _case(seed=1)
    expected = _float_sequence_loss(logits_per_position=logits, targets=targets)
    loss = sequence_loss(logits_per_position=_matrix(values=logits), targets=targets)
    assert loss.forward_val == pytest.approx(expected)


# --- Group 5: inf and nan (Section 4) ----------------------------------------
#
# Two logits, the target far below the other. The gap sets the probability
# of the target, exp(-gap) up to rounding. Three zones:
#
#   gap <= 709   exp(-gap) is a normal float. Loss and gradient are right.
#   710 .. 745   exp(-gap) is subnormal. The loss is finite but 1 / p in the
#                propagate function of log overflows, and inf and nan reach
#                the logits.
#   gap >= 746   exp(-gap) underflows to 0.0, and log(0.0) raises in the
#                forward pass.

GAP_FINITE = 700.0
GAP_BROKEN_GRADIENT = 720.0
GAP_RAISES = 800.0


def test_a_gap_of_700_gives_a_finite_loss_about_equal_to_the_gap() -> None:
    logits = _nodes(values=[0.0, -GAP_FINITE])
    loss = cross_entropy(logits=logits, target=1)
    assert loss.forward_val == pytest.approx(GAP_FINITE)
    loss.backward()
    assert _gradients(nodes=logits) == pytest.approx([1.0, -1.0])


def test_a_gap_of_720_gives_a_finite_loss_but_not_a_finite_gradient() -> None:
    logits = _nodes(values=[0.0, -GAP_BROKEN_GRADIENT])
    loss = cross_entropy(logits=logits, target=1)
    assert math.isfinite(loss.forward_val)
    loss.backward()
    assert not all(math.isfinite(g) for g in _gradients(nodes=logits))


def test_a_gap_of_800_raises_in_the_forward_pass() -> None:
    logits = _nodes(values=[0.0, -GAP_RAISES])
    with pytest.raises(ValueError):
        cross_entropy(logits=logits, target=1)


# --- Group 6: numeric gradient check (AUTOGRAD.md section 10) ----------------
#
# Same method as layers test group 6: compare the autograd gradient of each
# input against a central finite difference, (f(v + h) - f(v - h)) / (2h).
#
# Every logit is an input: 3 positions times 4 tokens. The module returns
# one scalar, so the check uses it as the loss with no weighting. The
# gradient of a logit is (p - onehot) / n_positions, which is never zero
# for finite logits, so nothing is zero by construction.

H = 1e-6
TOL = 1e-6


def _unpack(*, v: list, n: int = VOCAB_SIZE) -> list:
    """Split one flat list into rows of ``n``."""
    return [v[r * n : (r + 1) * n] for r in range(len(v) // n)]


def _node_loss(*, v: Vector) -> Node:
    """Return the loss of the module under test."""
    return sequence_loss(logits_per_position=_unpack(v=v), targets=TARGETS)


def _float_loss(*, v: list[float]) -> float:
    """Return the same loss as ``_node_loss`` with floats only."""
    return _float_sequence_loss(logits_per_position=_unpack(v=v), targets=TARGETS)


def test_backward_matches_finite_difference() -> None:
    rng = random.Random(2)
    inputs = _random_floats(rng=rng, n=N_POSITIONS * VOCAB_SIZE)

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
