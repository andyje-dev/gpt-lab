"""Unit tests for the Adam optimizer.

Each test group maps to a section of docs/OPTIMIZER.md.

The class is imported by name. Until it exists, the import fails and pytest
reports one collection error. Once it exists, each test fails or passes on
its own.
"""

import random
from itertools import pairwise

import pytest

from gptlab.autograd import Node
from gptlab.layers import Vector
from gptlab.optimizer import Adam


def _nodes(*, values: list[float]) -> Vector:
    """Return one leaf Node for each float."""
    return [Node(forward_val=v) for v in values]


def _values(*, nodes: Vector) -> list[float]:
    """Return the forward value of each Node."""
    return [n.forward_val for n in nodes]


def _gradients(*, nodes: Vector) -> list[float]:
    """Return the gradient of each Node."""
    return [n.gradient for n in nodes]


def _set_gradients(*, nodes: Vector, values: list[float]) -> None:
    """Write one float into the gradient slot of each Node."""
    for node, value in zip(nodes, values, strict=True):
        node.gradient = value


def _random_floats(*, rng: random.Random, n: int) -> list[float]:
    """Return ``n`` floats drawn from a standard normal."""
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _random_matrix(*, rng: random.Random, rows: int, cols: int) -> list[list[float]]:
    """Return a ``rows x cols`` matrix of floats drawn from a standard normal."""
    return [_random_floats(rng=rng, n=cols) for _ in range(rows)]


# --- Float reference ---------------------------------------------------------
#
# A second Adam, written with floats only. It shares no code with the module
# under test. It takes the start values and one gradient list per step, and
# returns the values after every step. Group 6 compares against it.


def _float_adam(
    *,
    values: list[float],
    gradients_per_step: list[list[float]],
    num_steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> list[float]:
    """Return ``values`` after one Adam step per row of ``gradients_per_step``."""
    values = list(values)
    first = [0.0 for _ in values]
    second = [0.0 for _ in values]
    for t, gradients in enumerate(gradients_per_step):
        lr_t = lr * (1 - t / num_steps)
        for i, g in enumerate(gradients):
            first[i] = beta1 * first[i] + (1 - beta1) * g
            second[i] = beta2 * second[i] + (1 - beta2) * g * g
            m_hat = first[i] / (1 - beta1 ** (t + 1))
            v_hat = second[i] / (1 - beta2 ** (t + 1))
            values[i] -= lr_t * m_hat / (v_hat**0.5 + eps)
    return values


# --- Shared small case --------------------------------------------------------
#
# Five parameters and six steps. The gradients are random but fixed by the
# seed, so each test sees the same numbers. The optimizer never sees a loss
# here: the tests write the gradient slots and call step. The defaults of
# the issue are repeated here so that a test can pin them.

N_PARAMS = 5
N_STEPS = 6
DEFAULT_LR = 0.01
DEFAULT_BETA1 = 0.85
DEFAULT_BETA2 = 0.99
DEFAULT_EPS = 1e-8


def _case(*, seed: int = 0) -> tuple[list[float], list[list[float]]]:
    """Return (start values, one gradient list per step) for the shared case."""
    rng = random.Random(seed)
    values = _random_floats(rng=rng, n=N_PARAMS)
    gradients_per_step = _random_matrix(rng=rng, rows=N_STEPS, cols=N_PARAMS)
    return values, gradients_per_step


# --- Group 1: construction (Section 3) ---------------------------------------


def test_construction_changes_no_forward_value() -> None:
    values, _ = _case()
    params = _nodes(values=values)
    Adam(params=params, num_steps=N_STEPS)
    assert _values(nodes=params) == values


def test_construction_changes_no_gradient() -> None:
    values, (gradients, *_) = _case()
    params = _nodes(values=values)
    _set_gradients(nodes=params, values=gradients)
    Adam(params=params, num_steps=N_STEPS)
    assert _gradients(nodes=params) == gradients


def test_construction_accepts_an_empty_list() -> None:
    optimizer = Adam(params=[], num_steps=N_STEPS)
    optimizer.step(t=0)
    optimizer.zero_grad()


# --- Group 2: hand case (Sections 3 and 4) -----------------------------------
#
# One parameter at 2.0. lr = 0.1, beta1 = 0.25, beta2 = 0.75, num_steps = 4.
#
# Step 0, gradient 4:
#   m = 0.75 * 4 = 3           m_hat = 3 / (1 - 0.25) = 4
#   v = 0.25 * 16 = 4          v_hat = 4 / (1 - 0.75) = 16
#   lr_0 = 0.1                 move = 0.1 * 4 / (4 + eps) = 0.1
#   p = 1.9
#
# Step 1, gradient 0:
#   m = 0.25 * 3 = 0.75        m_hat = 0.75 / (1 - 0.0625) = 0.8
#   v = 0.75 * 4 = 3           v_hat = 3 / (1 - 0.5625) = 48 / 7
#   lr_1 = 0.1 * (1 - 1/4)     move = 0.075 * 0.8 / sqrt(48 / 7) = 0.0229129
#   p = 1.8770871
#
# Without bias correction, step 0 gives 1.85. With the betas swapped, step 1
# gives 1.8281. Plain SGD gives 1.6 and then does not move at step 1.

HAND_START = 2.0
HAND_LR = 0.1
HAND_BETA1 = 0.25
HAND_BETA2 = 0.75
HAND_NUM_STEPS = 4
HAND_GRADIENT_0 = 4.0
HAND_AFTER_0 = 1.9
HAND_AFTER_1 = 1.8770871


def _hand_optimizer(*, param: Node) -> Adam:
    """Return the optimizer of the hand case over one parameter."""
    return Adam(
        params=[param],
        num_steps=HAND_NUM_STEPS,
        lr=HAND_LR,
        beta1=HAND_BETA1,
        beta2=HAND_BETA2,
        eps=DEFAULT_EPS,
    )


def test_one_step_matches_the_hand_computation() -> None:
    param = Node(forward_val=HAND_START)
    optimizer = _hand_optimizer(param=param)
    param.gradient = HAND_GRADIENT_0
    optimizer.step(t=0)
    assert param.forward_val == pytest.approx(HAND_AFTER_0)


def test_a_zero_gradient_at_step_1_still_moves_the_parameter() -> None:
    param = Node(forward_val=HAND_START)
    optimizer = _hand_optimizer(param=param)
    param.gradient = HAND_GRADIENT_0
    optimizer.step(t=0)
    param.gradient = 0.0
    optimizer.step(t=1)
    assert param.forward_val == pytest.approx(HAND_AFTER_1)


# --- Group 3: properties (Sections 2 and 3) ----------------------------------


def test_the_first_move_has_size_lr_whatever_the_gradient() -> None:
    params = _nodes(values=[0.0, 0.0, 0.0])
    optimizer = Adam(params=params, num_steps=N_STEPS, lr=0.5)
    _set_gradients(nodes=params, values=[0.001, 1.0, 1000.0])
    optimizer.step(t=0)
    # eps shows at the fifth digit of the smallest gradient: 0.001 / (0.001 + 1e-8).
    assert _values(nodes=params) == pytest.approx([-0.5, -0.5, -0.5], rel=1e-4)


def test_a_negative_gradient_moves_the_parameter_up() -> None:
    params = _nodes(values=[1.0])
    optimizer = Adam(params=params, num_steps=N_STEPS, lr=0.5)
    _set_gradients(nodes=params, values=[-2.0])
    optimizer.step(t=0)
    assert params[0].forward_val == pytest.approx(1.5)


def test_a_zero_gradient_at_step_0_does_not_move_the_parameter() -> None:
    params = _nodes(values=[1.0, 1.0])
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _set_gradients(nodes=params, values=[3.0, 0.0])
    optimizer.step(t=0)
    assert params[0].forward_val != 1.0
    assert params[1].forward_val == 1.0


def test_each_parameter_has_its_own_moments() -> None:
    params = _nodes(values=[1.0, 1.0])
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _set_gradients(nodes=params, values=[3.0, 0.0])
    optimizer.step(t=0)
    _set_gradients(nodes=params, values=[0.0, 0.0])
    optimizer.step(t=1)
    assert params[0].forward_val != pytest.approx(1.0 - DEFAULT_LR)
    assert params[1].forward_val == 1.0


def test_step_makes_no_node() -> None:
    params = _nodes(values=[1.0, 2.0])
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _set_gradients(nodes=params, values=[3.0, -1.0])
    optimizer.step(t=0)
    for param in params:
        assert isinstance(param.forward_val, float)
        assert param.children == ()


def test_step_moves_the_nodes_in_place() -> None:
    params = _nodes(values=[1.0, 2.0])
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _set_gradients(nodes=params, values=[3.0, -1.0])
    before = list(params)
    optimizer.step(t=0)
    assert all(a is b for a, b in zip(params, before, strict=True))


def test_step_leaves_the_gradient_slots_as_they_are() -> None:
    params = _nodes(values=[1.0, 2.0])
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _set_gradients(nodes=params, values=[3.0, -1.0])
    optimizer.step(t=0)
    assert _gradients(nodes=params) == [3.0, -1.0]


# --- Group 4: decay (Section 5) ----------------------------------------------
#
# With the same gradient at every step, the corrected moments are g and g*g,
# so the move at step t is lr_t = lr * (1 - t / num_steps), up to eps.

DECAY_LR = 0.1
DECAY_NUM_STEPS = 100


def _moves(*, num_steps: int) -> list[float]:
    """Return the size of the move at each step under a constant gradient."""
    param = Node(forward_val=0.0)
    optimizer = Adam(params=[param], num_steps=num_steps, lr=DECAY_LR)
    moves: list[float] = []
    for t in range(num_steps):
        before = param.forward_val
        param.gradient = 1.0
        optimizer.step(t=t)
        moves.append(before - param.forward_val)
    return moves


def test_the_first_move_is_lr() -> None:
    moves = _moves(num_steps=DECAY_NUM_STEPS)
    assert moves[0] == pytest.approx(DECAY_LR)


def test_the_last_move_is_near_zero() -> None:
    moves = _moves(num_steps=DECAY_NUM_STEPS)
    assert moves[-1] == pytest.approx(DECAY_LR / DECAY_NUM_STEPS)
    assert moves[-1] < DECAY_LR * 0.02


def test_the_moves_shrink_linearly() -> None:
    moves = _moves(num_steps=DECAY_NUM_STEPS)
    expected = [DECAY_LR * (1 - t / DECAY_NUM_STEPS) for t in range(DECAY_NUM_STEPS)]
    assert moves == pytest.approx(expected)


def test_rejects_a_step_past_num_steps() -> None:
    optimizer = Adam(params=_nodes(values=[0.0]), num_steps=N_STEPS)
    with pytest.raises(ValueError, match=f"t={N_STEPS}"):
        optimizer.step(t=N_STEPS)


def test_rejects_a_negative_step() -> None:
    optimizer = Adam(params=_nodes(values=[0.0]), num_steps=N_STEPS)
    with pytest.raises(ValueError, match="t=-1"):
        optimizer.step(t=-1)


# --- Group 5: zeroing (Section 6) --------------------------------------------


def test_zero_grad_clears_every_slot() -> None:
    values, (gradients, *_) = _case()
    params = _nodes(values=values)
    _set_gradients(nodes=params, values=gradients)
    optimizer = Adam(params=params, num_steps=N_STEPS)
    optimizer.zero_grad()
    assert _gradients(nodes=params) == [0.0] * N_PARAMS


def test_zero_grad_changes_no_forward_value() -> None:
    values, (gradients, *_) = _case()
    params = _nodes(values=values)
    _set_gradients(nodes=params, values=gradients)
    optimizer = Adam(params=params, num_steps=N_STEPS)
    optimizer.zero_grad()
    assert _values(nodes=params) == values


def test_a_second_backward_after_zero_grad_matches_the_first() -> None:
    a = Node(forward_val=2.0)
    b = Node(forward_val=3.0)
    optimizer = Adam(params=[a, b], num_steps=N_STEPS)

    (a * b + a).backward()
    first = _gradients(nodes=[a, b])
    assert first == [4.0, 2.0]

    optimizer.zero_grad()
    (a * b + a).backward()
    assert _gradients(nodes=[a, b]) == first


def test_backward_without_zero_grad_adds_again() -> None:
    a = Node(forward_val=2.0)
    b = Node(forward_val=3.0)
    (a * b + a).backward()
    (a * b + a).backward()
    assert _gradients(nodes=[a, b]) == [8.0, 4.0]


# --- Group 6: float reference (Sections 3, 4, and 5) -------------------------


def _run(*, optimizer: Adam, params: Vector, gradients_per_step: list[list[float]]) -> None:
    """Write each gradient list into the slots and take one step, in order."""
    for t, gradients in enumerate(gradients_per_step):
        _set_gradients(nodes=params, values=gradients)
        optimizer.step(t=t)


def test_six_steps_match_the_float_reference_with_the_defaults() -> None:
    values, gradients_per_step = _case()
    params = _nodes(values=values)
    optimizer = Adam(params=params, num_steps=N_STEPS)
    _run(optimizer=optimizer, params=params, gradients_per_step=gradients_per_step)
    expected = _float_adam(
        values=values,
        gradients_per_step=gradients_per_step,
        num_steps=N_STEPS,
        lr=DEFAULT_LR,
        beta1=DEFAULT_BETA1,
        beta2=DEFAULT_BETA2,
        eps=DEFAULT_EPS,
    )
    assert _values(nodes=params) == pytest.approx(expected)


def test_six_steps_match_the_float_reference_with_other_hyperparameters() -> None:
    values, gradients_per_step = _case(seed=1)
    params = _nodes(values=values)
    optimizer = Adam(params=params, num_steps=N_STEPS, lr=0.3, beta1=0.5, beta2=0.9, eps=1e-6)
    _run(optimizer=optimizer, params=params, gradients_per_step=gradients_per_step)
    expected = _float_adam(
        values=values,
        gradients_per_step=gradients_per_step,
        num_steps=N_STEPS,
        lr=0.3,
        beta1=0.5,
        beta2=0.9,
        eps=1e-6,
    )
    assert _values(nodes=params) == pytest.approx(expected)


def test_a_longer_run_matches_the_float_reference_part_way() -> None:
    values, gradients_per_step = _case(seed=2)
    params = _nodes(values=values)
    optimizer = Adam(params=params, num_steps=50)
    _run(optimizer=optimizer, params=params, gradients_per_step=gradients_per_step)
    expected = _float_adam(
        values=values,
        gradients_per_step=gradients_per_step,
        num_steps=50,
        lr=DEFAULT_LR,
        beta1=DEFAULT_BETA1,
        beta2=DEFAULT_BETA2,
        eps=DEFAULT_EPS,
    )
    assert _values(nodes=params) == pytest.approx(expected)


# --- Group 7: convergence (Sections 1 and 2) ---------------------------------
#
# The three stages of training on the smallest graph there is. The loss is
# (x - 3) ** 2, so the minimum is at x = 3. Each step makes a new graph,
# runs backward, takes one step, and zeroes the slot.

CONVERGE_NUM_STEPS = 20
CONVERGE_LR = 0.3
CONVERGE_TARGET = 3.0


def _train(*, start: float) -> list[float]:
    """Return the value of x after each of the training steps."""
    x = Node(forward_val=start)
    optimizer = Adam(params=[x], num_steps=CONVERGE_NUM_STEPS, lr=CONVERGE_LR)
    history: list[float] = []
    for t in range(CONVERGE_NUM_STEPS):
        loss = (x - CONVERGE_TARGET) ** 2
        loss.backward()
        optimizer.step(t=t)
        optimizer.zero_grad()
        history.append(x.forward_val)
    return history


def test_twenty_steps_move_x_toward_3() -> None:
    history = _train(start=0.0)
    assert abs(history[-1] - CONVERGE_TARGET) < 0.5


def test_the_loss_falls_from_9_to_below_a_quarter() -> None:
    history = _train(start=0.0)
    assert (history[-1] - CONVERGE_TARGET) ** 2 < 0.25


def test_every_step_moves_x_up_from_0() -> None:
    history = _train(start=0.0)
    assert history[0] > 0.0
    assert all(b > a for a, b in pairwise(history))


def test_twenty_steps_move_x_down_from_6() -> None:
    history = _train(start=6.0)
    assert abs(history[-1] - CONVERGE_TARGET) < 0.5
