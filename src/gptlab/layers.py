"""Layer functions: linear, softmax, rmsnorm.

Each function takes a list of nodes and returns a list of nodes. The scalar
operations of the autograd engine make each output node, so the backward pass
needs nothing new. The design notes and the reason for each choice are in
docs/LAYERS.md.
"""

from gptlab.autograd import Node

Vector = list[Node]
Matrix = list[list[Node]]


def linear(x: Vector, w: Matrix) -> Vector:
    """Return the dot product of ``x`` with each row of ``w``.

    There is no bias term.

    Args:
        x: The input vector. It must have at least one node.
        w: The weight matrix. Each row must have the same length as ``x``.

    Returns:
        A vector with one node for each row of ``w``.

    Raises:
        ValueError: If a row of ``w`` does not have the same length as ``x``.
    """
    outputs: Vector = []

    for row in w:
        dot = x[0] * row[0]
        for xi, wi in zip(x[1:], row[1:], strict=True):
            dot += xi * wi
        outputs.append(dot)

    return outputs


def softmax(logits: Vector) -> Vector:
    """Return a probability distribution over the logits.

    The function subtracts the largest logit as a plain float before it
    applies exp. This prevents overflow. The shift cancels in the division,
    so it is not a node and it receives no gradient.

    Args:
        logits: The input vector. It must have at least one node.

    Returns:
        A vector with the same length as ``logits``. Each node is between
        zero and one. The nodes sum to one.
    """
    max_logit = max(logit.forward_val for logit in logits)

    exponentials: Vector = [(logit - max_logit).exp() for logit in logits]

    denominator = exponentials[0]
    for exponential in exponentials[1:]:
        denominator += exponential
    inverse = denominator**-1

    probabilities: Vector = [exponential * inverse for exponential in exponentials]

    return probabilities


def rmsnorm(x: Vector, eps: float = 1e-5) -> Vector:
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
    """
    squares: Vector = [xi**2 for xi in x]

    sum_of_squares: Node = squares[0]
    for square in squares[1:]:
        sum_of_squares += square

    mean_square = sum_of_squares / len(x)

    inverse_square_root = (mean_square + eps) ** -0.5

    outputs: Vector = [xi * inverse_square_root for xi in x]

    return outputs
