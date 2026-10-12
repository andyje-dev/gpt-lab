"""Two-layer feed-forward block for one position.

``mlp`` maps a vector to a hidden vector with one linear map, cuts every
negative element to zero with ReLU, and maps it back with a second linear
map. The hidden width is set by the caller through the shape of the
weights, not by this module. The function acts on one position and has no
data about the other positions. Both maps are calls of ``linear``, and the
ReLU is the primitive of the autograd engine, so the backward pass needs
nothing new. The design notes and the reason for each choice are in
docs/MLP.md.
"""

from gptlab.layers import Matrix, Vector, linear


def mlp(*, x: Vector, fc1: Matrix, fc2: Matrix) -> Vector:
    """Return ``fc2 @ relu(fc1 @ x)`` for one position.

    The hidden vector has one node per row of ``fc1``, so the caller sets
    the hidden width by the shape of ``fc1``. Each hidden node is the ReLU
    of one dot product, so a hidden unit with a negative pre-activation is
    zero and passes no gradient. There is no bias term, no residual
    connection, and no normalization. Those belong to the caller.

    Args:
        x: The input vector for one position, with ``n_in`` nodes.
        fc1: The first weights, ``n_hidden x n_in``. One row per hidden
            unit. Each row must have the same length as ``x``.
        fc2: The second weights, ``n_out x n_hidden``. One row per output
            unit. Each row must have the same length as the hidden vector,
            which is the number of rows of ``fc1``.

    Returns:
        A vector with one node for each row of ``fc2``, so ``n_out`` nodes.
        In a transformer block ``n_out`` equals ``n_in`` so that the result
        can be added to ``x``, but this function does not require it.

    Raises:
        ValueError: If a row of ``fc1`` does not have the same length as
            ``x``, or if a row of ``fc2`` does not have the same length as
            the hidden vector.
    """
    pre_activations: Vector = linear(x=x, w=fc1)
    hidden: Vector = [pre_activation.relu() for pre_activation in pre_activations]
    outputs: Vector = linear(x=hidden, w=fc2)

    return outputs
