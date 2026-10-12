"""Scalar autograd engine.

Each Node holds one float, the nodes that made it, and a closure. The closure
sends the gradient of the node one step back to those nodes. The design
notes, the derivations, and the reason for each choice are in
docs/AUTOGRAD.md.
"""

import math
from collections.abc import Callable
from typing import Literal


class Node:
    """One scalar value in a computation graph.

    A Node records how it was made. The forward pass makes the graph. The
    backward pass walks the graph in reverse and sends a gradient to each
    Node. Each arithmetic operator accepts a float or a Node. A float becomes
    a leaf Node with no children.

    Attributes:
        forward_val: The value of this node from the forward pass.
        children: The nodes that made this node. A leaf has no children.
        propagate: A closure that sends the gradient of this node one step
            back to its children. Its only argument is the gradient of this
            node.
        gradient: The derivative of the output with respect to this node. It
            starts at zero. Each call to ``backward`` adds to it.
    """

    def __init__(
        self,
        forward_val: float,
        children: tuple[Node, ...] = (),
        propagate: Callable[[float], None] = lambda g: None,
    ) -> None:
        """Initialize a node with a gradient of zero.

        Args:
            forward_val: The value of this node from the forward pass.
            children: The nodes that made this node. The default is an empty
                tuple, which makes a leaf.
            propagate: A closure that sends the gradient of this node one
                step back to its children. The default does nothing, which
                is correct for a leaf.
        """
        self.forward_val: float = forward_val
        self.children: tuple[Node, ...] = children
        self.propagate: Callable[[float], None] = propagate
        self.gradient: float = 0.0

    def __repr__(self) -> str:
        """Return a string that shows the forward value and the gradient."""
        return f"{self.__class__.__name__}(forward_val={self.forward_val}, gradient={self.gradient})"

    def __add__(self, other: float | Node) -> Node:
        """Return a node for ``self + other``.

        Args:
            other: A Node or a float. A float becomes a leaf Node.

        Returns:
            A new Node with both operands as children. In the backward pass,
            each operand receives the incoming gradient with no change.
        """
        other = other if isinstance(other, Node) else Node(other)

        def propagate(g: float) -> None:
            self.gradient += g
            other.gradient += g

        return Node(
            forward_val=self.forward_val + other.forward_val,
            children=(self, other),
            propagate=propagate,
        )

    def __mul__(self, other: float | Node) -> Node:
        """Return a node for ``self * other``.

        Args:
            other: A Node or a float. A float becomes a leaf Node.

        Returns:
            A new Node with both operands as children. In the backward pass,
            each operand receives the incoming gradient times the forward
            value of the other operand.
        """
        other = other if isinstance(other, Node) else Node(other)

        def propagate(g: float) -> None:
            self.gradient += g * other.forward_val
            other.gradient += g * self.forward_val

        return Node(
            forward_val=self.forward_val * other.forward_val,
            children=(self, other),
            propagate=propagate,
        )

    def __pow__(self, exponent: float) -> Node:
        """Return a node for ``self ** exponent``.

        Args:
            exponent: A float constant. It is not a Node, so it receives no
                gradient.

        Returns:
            A new Node with ``self`` as its only child. The backward pass uses
            the power rule. It multiplies the incoming gradient by
            ``exponent * self ** (exponent - 1)``.
        """

        def propagate(g: float) -> None:
            self.gradient += g * exponent * self.forward_val ** (exponent - 1)

        return Node(
            forward_val=self.forward_val**exponent,
            children=(self,),
            propagate=propagate,
        )

    def exp(self) -> Node:
        """Return a node for ``e ** self``.

        Returns:
            A new Node with ``self`` as its only child. The derivative of exp
            is exp. The backward pass multiplies the incoming gradient by the
            forward value of the result, so it computes nothing again.
        """
        result = math.exp(self.forward_val)

        def propagate(g: float) -> None:
            self.gradient += g * result

        return Node(
            forward_val=result,
            children=(self,),
            propagate=propagate,
        )

    def log(self) -> Node:
        """Return a node for the natural logarithm of ``self``.

        Returns:
            A new Node with ``self`` as its only child. The backward pass
            divides the incoming gradient by the forward value of ``self``.

        Raises:
            ValueError: If the forward value of ``self`` is zero or negative.
        """

        def propagate(g: float) -> None:
            self.gradient += g / self.forward_val

        return Node(
            forward_val=math.log(self.forward_val),
            children=(self,),
            propagate=propagate,
        )

    def relu(self) -> Node:
        """Return a node for ``max(0, self)``.

        Returns:
            A new Node with ``self`` as its only child. The backward pass
            sends the incoming gradient through only if the forward value of
            ``self`` is greater than zero. At zero, the gradient is zero.
        """

        def propagate(g: float) -> None:
            if self.forward_val > 0:
                self.gradient += g

        return Node(
            forward_val=max(0.0, self.forward_val),
            children=(self,),
            propagate=propagate,
        )

    def __radd__(self, other: float | Node) -> Node:
        """Return ``other + self``. Addition commutes, so this is ``self + other``."""
        return self + other

    def __rmul__(self, other: float | Node) -> Node:
        """Return ``other * self``. Multiplication commutes, so this is ``self * other``."""
        return self * other

    def __neg__(self) -> Node:
        """Return ``-self`` as ``self * -1``."""
        return self * -1

    def __sub__(self, other: float | Node) -> Node:
        """Return ``self - other`` as ``self + (-other)``."""
        return self + -other

    def __rsub__(self, other: float | Node) -> Node:
        """Return ``other - self`` as ``(-self) + other``."""
        return -self + other

    def __truediv__(self, other: float | Node) -> Node:
        """Return ``self / other`` as ``self * other ** -1``."""
        return self * other**-1

    def __rtruediv__(self, other: float | Node) -> Node:
        """Return ``other / self`` as ``other * self ** -1``."""
        return other * self**-1

    def _topological_order(self) -> list[Node]:
        """Return this node and all nodes below it, in topological order.

        The walk is an iterative depth-first search with an explicit stack.
        It does not use recursion, so a deep graph does not reach the Python
        recursion limit. Each node appears one time only, after all of its
        children. This node is the last item.

        Returns:
            A list of nodes. Each node comes after all of its children.
        """
        topological_order: list[Node] = []
        visited: set[Node] = set()
        stack: list[tuple[Node, Literal["expand", "append"]]] = [(self, "expand")]
        while stack:
            node, tag = stack.pop()
            if tag == "append":
                topological_order.append(node)
            elif tag == "expand":
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, "append"))
                for c in node.children:
                    stack.append((c, "expand"))
        return topological_order

    def backward(self) -> None:
        """Run the backward pass from this node.

        This method sets the gradient of this node to one. Then it walks the
        topological order in reverse and calls ``propagate`` on each node.
        Each node has its full gradient before it sends anything to its
        children.

        This method does not reset gradients. A second call adds the same
        contributions again. The optimizer zeroes the gradients, not this
        method.
        """
        self.gradient = 1.0
        for n in reversed(self._topological_order()):
            n.propagate(n.gradient)
