"""Layer functions: linear, softmax, rmsnorm.

Each function takes a list of nodes and returns a list of nodes. Every node
is made by a scalar operation from the autograd engine, so the backward pass
needs nothing new. Design notes and the reasoning behind each choice are in
docs/LAYERS.md.
"""

from gptlab.autograd import Node

Vector = list[Node]
Matrix = list[list[Node]]


def linear(x: Vector, w: Matrix) -> Vector:
    """Return the dot product of x with each row of w. No bias."""
    outputs: Vector = []

    for row in w:
        dot = x[0] * row[0]
        for xi, wi in zip(x[1:], row[1:], strict=True):
            dot += xi * wi
        outputs.append(dot)

    return outputs


def softmax(logits: Vector) -> Vector:
    """Return a probability distribution over the logits.

    The largest logit is subtracted as a plain float before exp, so exp
    cannot overflow. The shift cancels in the division and is not a node.
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
    """Divide each element by the root-mean-square of x. No learned scale.

    eps is inside the root, so a zero vector gives zeros with no division
    by zero in the forward or the backward pass.
    """
    squares: Vector = [xi**2 for xi in x]

    sum_of_squares: Node = squares[0]
    for square in squares[1:]:
        sum_of_squares += square

    mean_square = sum_of_squares / len(x)

    inverse_square_root = (mean_square + eps) ** -0.5

    outputs: Vector = [xi * inverse_square_root for xi in x]

    return outputs
