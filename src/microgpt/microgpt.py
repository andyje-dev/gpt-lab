import math
from collections.abc import Callable
from typing import Literal


class Node:
    def __init__(
        self,
        forward_val: float,
        parents: tuple[Node, ...] = (),
        backward_func: Callable[[float], None] = lambda g: None,
    ) -> None:
        self.forward_val: float = forward_val
        self.parents: tuple[Node, ...] = parents
        self.backward_func: Callable[[float], None] = backward_func
        self.gradient: float = 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(forward_val={self.forward_val}, gradient={self.gradient})"

    def __add__(self, other: float | Node) -> Node:
        other = other if isinstance(other, Node) else Node(other)

        def backward(g: float) -> None:
            self.gradient += g
            other.gradient += g

        return Node(
            forward_val=self.forward_val + other.forward_val,
            parents=(self, other),
            backward_func=backward,
        )

    def __mul__(self, other: float | Node) -> Node:
        other = other if isinstance(other, Node) else Node(other)

        def backward(g: float) -> None:
            self.gradient += g * other.forward_val
            other.gradient += g * self.forward_val

        return Node(
            forward_val=self.forward_val * other.forward_val,
            parents=(self, other),
            backward_func=backward,
        )

    def __pow__(self, other: float) -> Node:
        def backward(g: float) -> None:
            self.gradient += g * other * self.forward_val ** (other - 1)

        return Node(
            forward_val=self.forward_val**other, parents=(self,), backward_func=backward
        )

    def exp(self) -> Node:
        result = math.exp(self.forward_val)

        def backward(g: float) -> None:
            self.gradient += g * result

        return Node(
            forward_val=result,
            parents=(self,),
            backward_func=backward,
        )

    def log(self) -> Node:
        def backward(g: float) -> None:
            self.gradient += g / self.forward_val

        return Node(
            forward_val=math.log(self.forward_val),
            parents=(self,),
            backward_func=backward,
        )

    def relu(self) -> Node:
        def backward(g: float) -> None:
            if self.forward_val > 0:
                self.gradient += g

        return Node(
            forward_val=max(0.0, self.forward_val),
            parents=(self,),
            backward_func=backward,
        )

    def __radd__(self, other: float | Node) -> Node:
        return self + other

    def __rmul__(self, other: float | Node) -> Node:
        return self * other

    def __neg__(self) -> Node:
        return self * -1

    def __sub__(self, other: float | Node) -> Node:
        return self + -other

    def __rsub__(self, other: float | Node) -> Node:
        return -self + other

    def __truediv__(self, other: float | Node) -> Node:
        return self * other**-1

    def __rtruediv__(self, other: float | Node) -> Node:
        return other * self**-1

    def _topological_order(self) -> list[Node]:
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
                for p in node.parents:
                    stack.append((p, "expand"))
        return topological_order

    def backward(self):
        self.gradient = 1.0
        for n in reversed(self._topological_order()):
            n.backward_func(n.gradient)
