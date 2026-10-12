"""Layer functions: dot, linear, softmax, rmsnorm, transpose.

Each function takes lists of nodes and returns nodes. The scalar operations
of the autograd engine make each output node, so the backward pass needs
nothing new. ``transpose`` makes no node at all. It only moves references.
The design notes and the reason for each choice are in docs/LAYERS.md.
"""

from gptlab.autograd import Node

Vector = list[Node]
Matrix = list[Vector]


def dot(*, x: Vector, y: Vector) -> Node:
    """Return the dot product of ``x`` and ``y``.

    The graph is one product per element and a chain of sums seeded from the
    first product. ``linear`` calls this function once per row.

    Args:
        x: The first vector. It must have at least one node.
        y: The second vector. It must have the same length as ``x``.

    Returns:
        One node.

    Raises:
        ValueError: If ``x`` is empty, or if ``x`` and ``y`` do not have the
            same length.
    """
    if len(x) == 0:
        raise ValueError("x is empty")
    dot_product: Node = x[0] * y[0]
    for xi, yi in zip(x[1:], y[1:], strict=True):
        dot_product += xi * yi

    return dot_product


def linear(*, x: Vector, w: Matrix) -> Vector:
    """Return the dot product of ``x`` with each row of ``w``.

    Each output is one call of ``dot``. There is no bias term.

    Args:
        x: The input vector. It must have at least one node.
        w: The weight matrix. Each row must have the same length as ``x``.

    Returns:
        A vector with one node for each row of ``w``.

    Raises:
        ValueError: If a row of ``w`` does not have the same length as ``x``.
    """
    outputs: Vector = [dot(x=x, y=row) for row in w]

    return outputs


def softmax(*, logits: Vector) -> Vector:
    """Return a probability distribution over the logits.

    The function subtracts the largest logit as a plain float before it
    applies exp. This prevents overflow. The shift cancels in the division,
    so it is not a node and it receives no gradient.

    Args:
        logits: The input vector. It must have at least one node.

    Returns:
        A vector with the same length as ``logits``. Each node is between
        zero and one. The nodes sum to one.

    Raises:
        ValueError: If ``logits`` is empty.
    """
    if len(logits) == 0:
        raise ValueError("logits is empty")
    max_logit = max(logit.forward_val for logit in logits)

    exponentials: Vector = [(logit - max_logit).exp() for logit in logits]

    denominator = exponentials[0]
    for exponential in exponentials[1:]:
        denominator += exponential
    inverse = denominator**-1

    probabilities: Vector = [exponential * inverse for exponential in exponentials]

    return probabilities


def rmsnorm(*, x: Vector, eps: float = 1e-5) -> Vector:
    """Divide each element of ``x`` by the root-mean-square of ``x``.

    There is no learned scale. The ``eps`` term is inside the root. A zero
    vector gives zeros, and there is no division by zero in the forward pass
    or in the backward pass.

    Args:
        x: The input vector. It must have at least one node.
        eps: A small constant. The function adds it to the mean square
            before the square root. The default is 1e-5.

    Returns:
        A vector with the same length as ``x``.

    Raises:
        ValueError: If ``x`` is empty.
    """
    if len(x) == 0:
        raise ValueError("x is empty")
    squares: Vector = [xi**2 for xi in x]

    sum_of_squares: Node = squares[0]
    for square in squares[1:]:
        sum_of_squares += square

    mean_square = sum_of_squares / len(x)

    inverse_square_root = (mean_square + eps) ** -0.5

    outputs: Vector = [xi * inverse_square_root for xi in x]

    return outputs


def transpose(*, m: Matrix) -> Matrix:
    """Return ``m`` with its rows and columns swapped.

    Element ``[r][c]`` of the result is element ``[c][r]`` of ``m``. The
    function makes no node. Each row of the result holds the same node
    objects as a column of ``m``.

    Args:
        m: The matrix. Every row must have the same length.

    Returns:
        A matrix with one row for each column of ``m``.

    Raises:
        ValueError: If the rows of ``m`` do not all have the same length.
    """
    return [list(col) for col in zip(*m, strict=True)]
