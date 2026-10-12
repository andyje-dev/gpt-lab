"""Cross-entropy loss over a sequence.

``cross_entropy`` turns the logits of one position into probabilities with
``softmax`` and returns the negative log of the probability of the target
token. ``sequence_loss`` calls it once per position and returns the mean,
so the loss is one node and the root of the graph that ``backward`` walks.
The softmax is the function of ``layers``, and the log and the negation
are operators of the autograd engine, so the backward pass needs nothing
new. The design notes and the reason for each choice are in docs/LOSS.md.
"""

from gptlab.autograd import Node
from gptlab.layers import Vector, softmax


def cross_entropy(*, logits: Vector, target: int) -> Node:
    """Return ``-log(softmax(logits)[target])`` for one position.

    The function makes the probabilities with ``softmax``, reads the one at
    index ``target`` by plain list indexing, and returns its negative log.
    The indexing makes no node. The gradient of the loss with respect to
    logit ``i`` is ``p[i] - 1`` if ``i`` is the target and ``p[i]``
    otherwise. The function does not average, and it does not clip. A
    target probability that rounds to zero makes ``log`` raise, and one
    that is subnormal makes the gradient ``inf`` or ``nan``.

    Args:
        logits: One node per token of the vocabulary, so ``vocab_size``
            nodes. It must have at least one node.
        target: The index of the correct token. It must be in
            ``0 .. vocab_size - 1``.

    Returns:
        One node, greater than zero for finite logits.

    Raises:
        ValueError: If ``target`` is negative or past the last logit, or if
            ``logits`` is empty.
    """
    if not (0 <= target < len(logits)):
        raise ValueError(f"target={target} is not in 0...{len(logits) - 1}")
    probabilities: Vector = softmax(logits=logits)
    probability: Node = probabilities[target]
    loss: Node = -probability.log()
    return loss


def sequence_loss(*, logits_per_position: list[Vector], targets: list[int]) -> Node:
    """Return the mean of ``cross_entropy`` over the positions of a sequence.

    The function calls ``cross_entropy`` once per position, sums the losses
    in a loop seeded from the first one, and divides by the number of
    positions. The mean, not the sum, so that the gradient does not grow
    with the sequence length. The division is one multiplication by a
    float, so the graph holds the per-position losses, the sums, and one
    product.

    Args:
        logits_per_position: One vector per position, each with
            ``vocab_size`` nodes. It must have at least one position.
        targets: One index per position, the correct token at that
            position. It must have the same length as
            ``logits_per_position``.

    Returns:
        One node, the mean loss.

    Raises:
        ValueError: If ``logits_per_position`` is empty, if ``targets`` does
            not have the same length, or if a target is out of range for
            its logits.
    """
    if len(logits_per_position) == 0:
        raise ValueError("logits_per_position is empty")
    losses: Vector = [
        cross_entropy(logits=logits, target=target)
        for logits, target in zip(logits_per_position, targets, strict=True)
    ]
    sum_of_losses: Node = losses[0]
    for loss in losses[1:]:
        sum_of_losses += loss
    mean_loss: Node = sum_of_losses / len(logits_per_position)
    return mean_loss
