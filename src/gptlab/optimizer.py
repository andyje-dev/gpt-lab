"""Adam optimizer with learning-rate decay.

``Adam`` holds the flat parameter list of a model and two running averages
per parameter: the first moment of the gradient and the second moment, its
square. ``step`` reads each gradient slot, updates the two averages, corrects
them for the zero start, and moves each parameter against the corrected
ratio with a learning rate that falls in a straight line to zero over the
run. ``zero_grad`` sets every gradient slot back to zero, the job that
``backward`` leaves to the optimizer. Every line writes a float. The module
makes no node, so the backward pass needs nothing new. The design notes and
the reason for each choice are in docs/OPTIMIZER.md.
"""

from gptlab.autograd import Node


class Adam:
    """Adam over a flat list of parameter nodes, with linear learning-rate decay.

    The optimizer walks ``params`` and nothing else. The list is the caller's list,
    not a copy, so the nodes that a model holds in its tables are the nodes that move.

    Attributes:
        params: The parameter nodes to move, ``n_params`` of them.
        num_steps: The number of steps in the run. The learning rate reaches
            zero one step after the last one.
        lr: The learning rate at step zero.
        beta1: The decay of the first moment per step.
        beta2: The decay of the second moment per step.
        eps: A small float added to the root of the second moment so that
            the division is finite when the second moment is zero.
        first_moments: One float per parameter, the running average of the
            gradient. All zero at construction.
        second_moments: One float per parameter, the running average of the
            squared gradient. All zero at construction.
    """

    def __init__(
        self,
        *,
        params: list[Node],
        num_steps: int,
        lr: float = 0.01,
        beta1: float = 0.85,
        beta2: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        """Keep the parameters and the settings, and start both moments at zero.

        Construction reads no gradient and changes no forward value.

        Args:
            params: The parameter nodes to move, in any order. The same
                objects as the model's tables hold.
            num_steps: The number of steps in the run. It must be positive.
            lr: The learning rate at step zero. The default is the one of
                the issue.
            beta1: The decay of the first moment, in ``0 .. 1``.
            beta2: The decay of the second moment, in ``0 .. 1``.
            eps: The float added to the root of the second moment.
        """
        self.params: list[Node] = params
        self.num_steps: int = num_steps
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps
        self.first_moments: list[float] = [0.0 for _ in self.params]
        self.second_moments: list[float] = [0.0 for _ in self.params]

    def step(self, *, t: int) -> None:
        """Move every parameter one Adam step at step number ``t``.

        For each parameter with gradient ``g``, the method updates the two
        running averages, ``m = beta1 * m + (1 - beta1) * g`` and
        ``v = beta2 * v + (1 - beta2) * g * g``, divides each by
        ``1 - beta ** (t + 1)`` to undo the zero start, and subtracts
        ``lr_t * m_hat / (sqrt(v_hat) + eps)`` from the forward value, with
        ``lr_t = lr * (1 - t / num_steps)``. The corrected values are
        locals of this one step. The running averages stay in the lists.

        The method reads the gradient slots and does not change them. It
        writes the forward value of each leaf in place and makes no node.
        The graph of the step that filled the slots is stale after the
        call; the next forward pass makes a new one.

        Args:
            t: The step number, from ``0`` to ``num_steps - 1``, as the
                training loop counts it. The bias correction uses ``t + 1``
                so that step zero divides by a nonzero number.

        Raises:
            ValueError: If ``t`` is negative or not less than ``num_steps``.
        """
        if not 0 <= t < self.num_steps:
            raise ValueError(f"step t={t} is not in 0...{self.num_steps - 1} (num_steps={self.num_steps})")

        decayed_lr: float = self.lr * (1 - t / self.num_steps)
        for i, param in enumerate(self.params):
            gradient: float = param.gradient
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * gradient
            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * gradient**2
            corrected_first_moment: float = self.first_moments[i] / (1 - self.beta1 ** (t + 1))
            corrected_second_moment: float = self.second_moments[i] / (1 - self.beta2 ** (t + 1))
            param.forward_val -= (
                decayed_lr * corrected_first_moment / (corrected_second_moment**0.5 + self.eps)
            )

    def zero_grad(self) -> None:
        """Set the gradient slot of every parameter to zero.

        ``backward`` adds into the slots and never clears them, so the
        training loop calls this method once per step. The method changes
        no forward value and no moment.
        """
        for param in self.params:
            param.gradient = 0.0
