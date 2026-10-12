"""Unit tests for the GPT model: parameters and the forward pass.

Each test group maps to a section of docs/MODEL.md.

The names are imported by name. Until both exist, the import fails and
pytest reports one collection error. Once they exist, each test fails or
passes on its own.
"""

import math
import statistics

import pytest

from gptlab.autograd import Node
from gptlab.layers import Matrix, Vector
from gptlab.loss import sequence_loss
from gptlab.model import GPT, Config


def _values(*, nodes: Vector) -> list[float]:
    """Return the forward value of each Node."""
    return [n.forward_val for n in nodes]


def _gradients(*, nodes: Vector) -> list[float]:
    """Return the gradient of each Node."""
    return [n.gradient for n in nodes]


def _shape(*, m: Matrix) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a matrix whose rows all have one length."""
    cols = {len(row) for row in m}
    assert len(cols) == 1, "rows have different lengths"
    return len(m), cols.pop()


def _reachable(*, root: Node) -> set[Node]:
    """Return every node below ``root``, found by a walk over ``children``."""
    seen: set[Node] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(node.children)
    return seen


# --- Float reference ---------------------------------------------------------
#
# A second GPT, written with floats and ``math`` only. It shares no code with
# the module under test. The forward tests compare against it, and the
# gradient check perturbs it. The float parameters come from one flat list,
# unpacked in the order that section 3 of the outline fixes:
# wte, wpe, then wq wk wv wo fc1 fc2 per block, then lm_head.

LAYER_NAMES = ("wq", "wk", "wv", "wo", "fc1", "fc2")


def _float_linear(*, x: list[float], w: list[list[float]]) -> list[float]:
    """Return the dot product of ``x`` with each row of ``w``."""
    return [sum(xi * wi for xi, wi in zip(x, row, strict=True)) for row in w]


def _float_rmsnorm(*, x: list[float], eps: float = 1e-5) -> list[float]:
    """Return ``x`` divided by its root mean square, with ``eps`` inside the root."""
    scale = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * scale for v in x]


def _float_softmax(*, scores: list[float]) -> list[float]:
    """Return the softmax of the scores."""
    shift = max(scores)
    exps = [math.exp(s - shift) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


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


def _float_mlp(
    *, x: list[float], fc1: list[list[float]], fc2: list[list[float]]
) -> tuple[list[float], list[float]]:
    """Return ``(fc2 @ relu(fc1 @ x), fc1 @ x)``: the output and the pre-activations."""
    pre_activations = _float_linear(x=x, w=fc1)
    hidden = [max(0.0, h) for h in pre_activations]
    return _float_linear(x=hidden, w=fc2), pre_activations


def _float_forward(
    *,
    tokens: list[int],
    wte: list[list[float]],
    wpe: list[list[float]],
    blocks: list[dict[str, list[list[float]]]],
    lm_head: list[list[float]],
    n_head: int,
) -> tuple[list[list[float]], list[float]]:
    """Return ``(logits, pre_activations)`` of the float GPT.

    ``logits`` has one list per position. ``pre_activations`` holds every
    MLP pre-activation of every block and position, for the ReLU kink guard
    of the gradient check.
    """
    x = [
        [t + p for t, p in zip(wte[token], wpe[i], strict=True)]
        for i, token in enumerate(tokens)
    ]
    x = [_float_rmsnorm(x=xi) for xi in x]
    pre_activations: list[float] = []
    for block in blocks:
        normed = [_float_rmsnorm(x=xi) for xi in x]
        attended = _float_attention(
            x=normed,
            wq=block["wq"],
            wk=block["wk"],
            wv=block["wv"],
            wo=block["wo"],
            n_head=n_head,
        )
        x = [
            [a + b for a, b in zip(xi, yi, strict=True)]
            for xi, yi in zip(x, attended, strict=True)
        ]
        normed = [_float_rmsnorm(x=xi) for xi in x]
        mlp_out = []
        for xi in normed:
            out, pre = _float_mlp(x=xi, fc1=block["fc1"], fc2=block["fc2"])
            mlp_out.append(out)
            pre_activations.extend(pre)
        x = [
            [a + b for a, b in zip(xi, yi, strict=True)]
            for xi, yi in zip(x, mlp_out, strict=True)
        ]
    x = [_float_rmsnorm(x=xi) for xi in x]
    return [_float_linear(x=xi, w=lm_head) for xi in x], pre_activations


def _float_cross_entropy(*, logits: list[float], target: int) -> float:
    """Return ``-log(softmax(logits)[target])`` with floats only."""
    return -math.log(_float_softmax(scores=logits)[target])


def _float_sequence_loss(*, logits: list[list[float]], targets: list[int]) -> float:
    """Return the mean cross-entropy over the positions with floats only."""
    losses = [
        _float_cross_entropy(logits=lg, target=t)
        for lg, t in zip(logits, targets, strict=True)
    ]
    return sum(losses) / len(losses)


def _count(*, config: Config) -> int:
    """Return the parameter count from the shapes of section 3."""
    return (
        2 * config.vocab_size * config.n_embd
        + config.context_length * config.n_embd
        + config.n_blocks * 12 * config.n_embd**2
    )


def _unpack(*, v: list, config: Config):
    """Split one flat list into ``(wte, wpe, blocks, lm_head)`` by the fixed order."""
    n = config.n_embd
    offset = 0

    def take(*, rows: int, cols: int) -> list:
        nonlocal offset
        m = [v[offset + r * cols : offset + (r + 1) * cols] for r in range(rows)]
        offset += rows * cols
        return m

    wte = take(rows=config.vocab_size, cols=n)
    wpe = take(rows=config.context_length, cols=n)
    blocks = []
    for _ in range(config.n_blocks):
        block = {}
        for name in ("wq", "wk", "wv", "wo"):
            block[name] = take(rows=n, cols=n)
        block["fc1"] = take(rows=4 * n, cols=n)
        block["fc2"] = take(rows=n, cols=4 * n)
        blocks.append(block)
    lm_head = take(rows=config.vocab_size, cols=n)
    assert offset == len(v), "flat list has leftover values"
    return wte, wpe, blocks, lm_head


# --- Shared small case --------------------------------------------------------
#
# vocab_size = 4 and context_length = 4, so one input can hold every token at
# every position. n_embd = 6, not 4, so that a table made with its two sizes
# swapped has the wrong shape and not a square that hides the swap. Two
# blocks, so a per-block bug cannot hide in block 0. The parameters come
# from the seed, so each test sees the same numbers.

CONFIG = Config(vocab_size=4, n_blocks=2, n_embd=6, n_head=2, context_length=4)
SEED = 0
TOKENS = [0, 1, 2, 3]
TARGETS = [1, 2, 3, 0]


def _model(*, config: Config = CONFIG, seed: int = SEED) -> GPT:
    """Return the model under test."""
    return GPT(config=config, seed=seed)


def _float_logits(
    *, model: GPT, config: Config, tokens: list[int]
) -> list[list[float]]:
    """Return the float reference logits for the parameters that ``model`` holds."""
    wte, wpe, blocks, lm_head = _unpack(v=_values(nodes=model.params), config=config)
    logits, _ = _float_forward(
        tokens=tokens,
        wte=wte,
        wpe=wpe,
        blocks=blocks,
        lm_head=lm_head,
        n_head=config.n_head,
    )
    return logits


# --- Group 1: config (Section 2) ---------------------------------------------


def test_config_stores_the_five_sizes() -> None:
    assert (
        CONFIG.vocab_size,
        CONFIG.n_blocks,
        CONFIG.n_embd,
        CONFIG.n_head,
        CONFIG.context_length,
    ) == (4, 2, 6, 2, 4)


def test_n_head_must_divide_n_embd_at_construction() -> None:
    # n_embd = 4. 3 heads would give head_dim = 1 and drop one dimension.
    # The error comes before any forward pass, from Config or from GPT.
    with pytest.raises(ValueError):
        _model(
            config=Config(
                vocab_size=4, n_blocks=1, n_embd=4, n_head=3, context_length=4
            )
        )


# --- Group 2: init (Section 2) -----------------------------------------------

INIT_STD = 0.08


def test_wte_is_vocab_size_by_n_embd() -> None:
    assert _shape(m=_model().wte) == (CONFIG.vocab_size, CONFIG.n_embd)


def test_wpe_is_context_length_by_n_embd() -> None:
    assert _shape(m=_model().wpe) == (CONFIG.context_length, CONFIG.n_embd)


def test_there_is_one_block_record_per_block() -> None:
    assert len(_model().blocks) == CONFIG.n_blocks


def test_attention_weights_are_square() -> None:
    n = CONFIG.n_embd
    for block in _model().blocks:
        for name in ("wq", "wk", "wv", "wo"):
            assert _shape(m=getattr(block, name)) == (n, n), name


def test_mlp_weights_expand_by_four_and_contract() -> None:
    n = CONFIG.n_embd
    for block in _model().blocks:
        assert _shape(m=block.fc1) == (4 * n, n)
        assert _shape(m=block.fc2) == (n, 4 * n)


def test_lm_head_is_vocab_size_by_n_embd() -> None:
    assert _shape(m=_model().lm_head) == (CONFIG.vocab_size, CONFIG.n_embd)


def test_init_is_gaussian_with_std_0_08() -> None:
    # 3840 parameters. The standard error of the sample std is about 0.001,
    # and of the sample mean about 0.0013, so the tolerances are loose.
    wide = Config(vocab_size=16, n_blocks=1, n_embd=16, n_head=4, context_length=16)
    values = _values(nodes=_model(config=wide).params)
    assert len(values) == 3840
    assert statistics.mean(values) == pytest.approx(0.0, abs=0.01)
    assert statistics.stdev(values) == pytest.approx(INIT_STD, abs=0.005)


def test_same_seed_gives_the_same_parameters() -> None:
    a = _values(nodes=_model(seed=7).params)
    b = _values(nodes=_model(seed=7).params)
    assert a == b


def test_a_different_seed_gives_different_parameters() -> None:
    a = _values(nodes=_model(seed=7).params)
    b = _values(nodes=_model(seed=8).params)
    assert a != b


# --- Group 3: params (Section 3) ---------------------------------------------


def test_params_count_matches_the_formula() -> None:
    assert len(_model().params) == _count(config=CONFIG) == 936


def test_params_count_matches_the_formula_on_a_second_config() -> None:
    config = Config(vocab_size=5, n_blocks=3, n_embd=6, n_head=3, context_length=7)
    assert len(_model(config=config).params) == _count(config=config)


def test_params_holds_nodes() -> None:
    assert all(isinstance(p, Node) for p in _model().params)


def test_params_holds_each_node_one_time() -> None:
    params = _model().params
    assert len({id(p) for p in params}) == len(params)


def test_params_holds_the_table_nodes_not_copies() -> None:
    # The optimizer moves the nodes in params. The forward pass reads the
    # tables. They must be the same objects.
    model = _model()
    assert model.params[0] is model.wte[0][0]
    assert model.params[-1] is model.lm_head[-1][-1]
    inner = model.blocks[-1].fc2[0][0]
    assert any(p is inner for p in model.params)


def test_params_starts_with_wte_then_wpe() -> None:
    # The first vocab_size * n_embd nodes are wte, row by row. The next
    # context_length * n_embd nodes are wpe. Section 3 fixes the order.
    model = _model()
    n_wte = CONFIG.vocab_size * CONFIG.n_embd
    flat_wte = [node for row in model.wte for node in row]
    flat_wpe = [node for row in model.wpe for node in row]
    assert all(a is b for a, b in zip(model.params[:n_wte], flat_wte, strict=True))
    assert all(
        a is b
        for a, b in zip(
            model.params[n_wte : n_wte + len(flat_wpe)], flat_wpe, strict=True
        )
    )


# --- Group 4: guards (Section 6) ---------------------------------------------


def test_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError):
        _model().forward(tokens=[])


def test_rejects_a_sequence_longer_than_context_length() -> None:
    with pytest.raises(ValueError):
        _model().forward(tokens=[0] * (CONFIG.context_length + 1))


def test_rejects_a_token_past_the_vocabulary() -> None:
    with pytest.raises(ValueError):
        _model().forward(tokens=[0, CONFIG.vocab_size])


def test_rejects_a_negative_token() -> None:
    # A negative index reads a row from the end of wte and raises nothing
    # on its own. The guard must catch it.
    with pytest.raises(ValueError):
        _model().forward(tokens=[0, -1])


# --- Group 5: shape (Section 6) ----------------------------------------------


def test_forward_returns_one_vector_per_position() -> None:
    assert len(_model().forward(tokens=[0, 1, 2])) == 3


def test_each_logits_vector_has_length_vocab_size() -> None:
    out = _model().forward(tokens=[0, 1, 2])
    assert all(len(logits) == CONFIG.vocab_size for logits in out)


def test_forward_returns_nodes() -> None:
    out = _model().forward(tokens=[0, 1, 2])
    assert all(isinstance(lg, Node) for logits in out for lg in logits)


def test_forward_accepts_a_full_block() -> None:
    assert len(_model().forward(tokens=TOKENS)) == CONFIG.context_length


def test_forward_accepts_one_position() -> None:
    assert len(_model().forward(tokens=[2])) == 1


def test_a_repeated_token_is_allowed() -> None:
    assert len(_model().forward(tokens=[1, 1, 1])) == 3


# --- Group 6: graph (Sections 3 and 4) ---------------------------------------
#
# The loss is one node. Every node below it is in its graph. With every
# token used and every position filled, every parameter must be below the
# loss. A wte row that no token selects is not in the graph, so backward
# never reaches it and its gradient stays at zero.


def test_every_parameter_is_below_the_loss_when_all_tokens_are_used() -> None:
    model = _model()
    loss = sequence_loss(
        logits_per_position=model.forward(tokens=TOKENS), targets=TARGETS
    )
    below = _reachable(root=loss)
    assert all(p in below for p in model.params)


def test_an_unused_wpe_row_is_not_below_the_loss() -> None:
    # Three positions, four wpe rows. Row 3 is not read.
    model = _model()
    loss = sequence_loss(
        logits_per_position=model.forward(tokens=[0, 1, 2]), targets=[1, 2, 3]
    )
    below = _reachable(root=loss)
    assert not any(p in below for p in model.wpe[3])
    assert all(p in below for p in model.wpe[2])


def test_the_wte_row_of_an_unused_token_gets_gradient_zero() -> None:
    # Token 3 is not in the input. Its row is not in the graph.
    model = _model()
    loss = sequence_loss(
        logits_per_position=model.forward(tokens=[0, 1, 2]), targets=[1, 2, 0]
    )
    loss.backward()
    assert _gradients(nodes=model.wte[3]) == [0.0] * CONFIG.n_embd
    assert any(g != 0.0 for g in _gradients(nodes=model.wte[0]))


def test_a_token_used_twice_sums_both_gradients_into_one_row() -> None:
    # The same row of wte feeds positions 0 and 2. Its gradient is the sum
    # of what each position sends. The test compares with the float
    # reference: perturb one element of the row and both positions move.
    model = _model()
    tokens = [1, 0, 1]
    targets = [0, 1, 0]
    loss = sequence_loss(
        logits_per_position=model.forward(tokens=tokens), targets=targets
    )
    loss.backward()

    inputs = _values(nodes=model.params)
    i = CONFIG.n_embd * 1  # first element of wte[1]
    h = 1e-6

    def float_loss(*, v: list[float]) -> float:
        wte, wpe, blocks, lm_head = _unpack(v=v, config=CONFIG)
        logits, _ = _float_forward(
            tokens=tokens,
            wte=wte,
            wpe=wpe,
            blocks=blocks,
            lm_head=lm_head,
            n_head=CONFIG.n_head,
        )
        return _float_sequence_loss(logits=logits, targets=targets)

    up = inputs[:i] + [inputs[i] + h] + inputs[i + 1 :]
    down = inputs[:i] + [inputs[i] - h] + inputs[i + 1 :]
    numeric = (float_loss(v=up) - float_loss(v=down)) / (2 * h)
    assert model.wte[1][0].gradient == pytest.approx(numeric, abs=1e-6)


# --- Group 7: float reference (Section 6) ------------------------------------


def test_forward_matches_float_reference() -> None:
    model = _model()
    out = model.forward(tokens=TOKENS)
    expected = _float_logits(model=model, config=CONFIG, tokens=TOKENS)
    for logits, e in zip(out, expected, strict=True):
        assert _values(nodes=logits) == pytest.approx(e)


def test_forward_matches_float_reference_on_a_second_seed() -> None:
    model = _model(seed=1)
    tokens = [3, 1, 0]
    out = model.forward(tokens=tokens)
    expected = _float_logits(model=model, config=CONFIG, tokens=tokens)
    for logits, e in zip(out, expected, strict=True):
        assert _values(nodes=logits) == pytest.approx(e)


def test_forward_matches_float_reference_with_one_block_and_one_head() -> None:
    config = Config(vocab_size=3, n_blocks=1, n_embd=2, n_head=1, context_length=3)
    model = _model(config=config, seed=2)
    out = model.forward(tokens=[2, 0])
    expected = _float_logits(model=model, config=config, tokens=[2, 0])
    for logits, e in zip(out, expected, strict=True):
        assert _values(nodes=logits) == pytest.approx(e)


# --- Group 8: causality (Section 5) ------------------------------------------
#
# Change the token at position 2. Positions 0 and 1 read nothing from it, so
# their logits are the same floats, not just close. Position 2 does change.
# Attention is the only place where positions meet, and the model must not
# add a second one in the residual add or the norms.


def test_a_later_token_does_not_change_earlier_logits() -> None:
    model = _model()
    base = model.forward(tokens=[0, 1, 2, 3])
    out = model.forward(tokens=[0, 1, 3, 3])
    assert _values(nodes=out[0]) == _values(nodes=base[0])
    assert _values(nodes=out[1]) == _values(nodes=base[1])
    assert _values(nodes=out[2]) != pytest.approx(_values(nodes=base[2]))


def test_an_earlier_token_does_change_later_logits() -> None:
    model = _model()
    base = model.forward(tokens=[0, 1, 2, 3])
    out = model.forward(tokens=[1, 1, 2, 3])
    assert _values(nodes=out[1]) != pytest.approx(_values(nodes=base[1]))
    assert _values(nodes=out[3]) != pytest.approx(_values(nodes=base[3]))


def test_a_prefix_gives_the_same_logits_as_the_full_sequence() -> None:
    # The first two positions of a four-token input see the same inputs as
    # a two-token input. So the logits are the same floats.
    model = _model()
    full = model.forward(tokens=[0, 1, 2, 3])
    prefix = model.forward(tokens=[0, 1])
    assert _values(nodes=prefix[0]) == _values(nodes=full[0])
    assert _values(nodes=prefix[1]) == _values(nodes=full[1])


# --- Group 9: numeric gradient check (AUTOGRAD.md section 10) ----------------
#
# Same method as blocks test group 6: compare the autograd gradient of each
# parameter against a central finite difference, (f(v + h) - f(v - h)) / (2h).
#
# The model is as small as it gets: one block, one head, n_embd = 2,
# vocab_size = 3, context_length = 3, so 66 parameters. The loss is the
# sequence_loss of the logits, so the whole base model is under one check.
# Every parameter is an input. The targets are fixed, so no gradient is
# zero by construction.
#
# ReLU has a kink at zero. Before the comparison, the test checks that every
# MLP pre-activation of the case is well away from zero.

H = 1e-6
TOL = 1e-6
KINK_MARGIN = 1e-3
GRAD_CONFIG = Config(vocab_size=3, n_blocks=1, n_embd=2, n_head=1, context_length=3)
GRAD_SEED = 3
GRAD_TOKENS = [0, 1, 2]
GRAD_TARGETS = [1, 2, 0]


def _float_loss(*, v: list[float]) -> tuple[float, list[float]]:
    """Return ``(loss, pre_activations)`` of the float GPT on the flat list ``v``."""
    wte, wpe, blocks, lm_head = _unpack(v=v, config=GRAD_CONFIG)
    logits, pre_activations = _float_forward(
        tokens=GRAD_TOKENS,
        wte=wte,
        wpe=wpe,
        blocks=blocks,
        lm_head=lm_head,
        n_head=GRAD_CONFIG.n_head,
    )
    return _float_sequence_loss(logits=logits, targets=GRAD_TARGETS), pre_activations


def test_backward_matches_finite_difference() -> None:
    model = _model(config=GRAD_CONFIG, seed=GRAD_SEED)
    inputs = _values(nodes=model.params)
    assert len(inputs) == 66

    expected, pre_activations = _float_loss(v=inputs)
    assert all(abs(p) > KINK_MARGIN for p in pre_activations)

    loss = sequence_loss(
        logits_per_position=model.forward(tokens=GRAD_TOKENS), targets=GRAD_TARGETS
    )
    assert isinstance(loss, Node)
    assert loss.forward_val == pytest.approx(expected)
    loss.backward()

    for i, node in enumerate(model.params):
        up = inputs[:i] + [inputs[i] + H] + inputs[i + 1 :]
        down = inputs[:i] + [inputs[i] - H] + inputs[i + 1 :]
        numeric = (_float_loss(v=up)[0] - _float_loss(v=down)[0]) / (2 * H)
        assert node.gradient == pytest.approx(numeric, abs=TOL), f"param {i}"
