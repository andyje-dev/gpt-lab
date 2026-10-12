"""GPT model: the parameters and the forward pass.

``Config`` holds the five sizes of the model. ``GPT`` makes every parameter
node one time at construction and puts all of them in one flat list,
``params``, that the optimizer walks. ``forward`` takes a list of token ids
and returns one logits vector per position: an embedding lookup, a stack
of blocks that each add attention and an MLP to the residual stream, a
final norm, and the head. Every step is a call of a function from
``layers``, ``attention``, or ``mlp``, or a plain add of nodes, so the
backward pass needs nothing new. The design notes and the reason for each
choice are in docs/MODEL.md.
"""

import random
from dataclasses import dataclass

from gptlab.attention import attention
from gptlab.autograd import Node
from gptlab.layers import Matrix, Vector, linear, rmsnorm
from gptlab.mlp import mlp


@dataclass(frozen=True, kw_only=True)
class Config:
    """The five sizes of a GPT.

    The config says what the model is. Two models with the same config
    have the same tables, the same parameter count, and the same forward
    pass. The seed of the parameters is not part of it.

    Attributes:
        vocab_size: The number of tokens the model knows. The row count of
            ``wte`` and ``lm_head``, and the length of each logits vector.
        n_embd: The width of the residual stream. The length of every
            embedding row and of every vector a block reads or returns.
        n_blocks: The number of blocks stacked in the forward pass.
        n_head: The number of attention heads. It must divide ``n_embd``.
        context_length: The most tokens one call of ``forward`` accepts.
            The row count of ``wpe``.
    """

    vocab_size: int
    n_embd: int
    n_blocks: int
    n_head: int
    context_length: int

    def __post_init__(self) -> None:
        """Check that ``n_head`` divides ``n_embd``.

        The check runs at construction, before any parameter is made.
        ``attention`` has the same guard for callers that use it alone.

        Raises:
            ValueError: If ``n_head`` does not divide ``n_embd``.
        """
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_head={self.n_head} does not divide n_embd={self.n_embd}"
            )


@dataclass(frozen=True, kw_only=True)
class Block:
    """The six weight tables of one block.

    A block is one ``attention`` call and one ``mlp`` call. It holds the
    tables and computes nothing. ``forward`` passes them straight through
    to the two functions.

    Attributes:
        wq: The query weights, ``n_embd x n_embd``.
        wk: The key weights, ``n_embd x n_embd``.
        wv: The value weights, ``n_embd x n_embd``.
        wo: The output weights, ``n_embd x n_embd``.
        fc1: The first MLP weights, ``n_hidden x n_embd``.
        fc2: The second MLP weights, ``n_embd x n_hidden``.
    """

    wq: Matrix
    wk: Matrix
    wv: Matrix
    wo: Matrix
    fc1: Matrix
    fc2: Matrix


# The standard deviation of every parameter at init.
# A later lab swap changes it here.
INIT_STD = 0.08

# The hidden width of the MLP as a multiple of n_embd. The ratio of the
# first Transformer and the first GPT models. See section 2 of docs/MLP.md.
MLP_RATIO = 4


class GPT:
    """A GPT: the parameter tables, the flat parameter list, and ``forward``.

    Attributes:
        config: The five sizes, as given at construction.
        wte: The token embeddings, ``vocab_size x n_embd``. Row ``t`` is
            the embedding of token ``t``.
        wpe: The position embeddings, ``context_length x n_embd``. Row
            ``i`` is the embedding of position ``i``.
        blocks: One ``Block`` per block, in forward order.
        lm_head: The head weights, ``vocab_size x n_embd``. Row ``t`` is
            what logit ``t`` reads from the final vector.
        params: Every parameter node one time, in a flat list: ``wte``,
            ``wpe``, then ``wq wk wv wo fc1 fc2`` of each block, then
            ``lm_head``, each row by row. The list holds the same node
            objects as the tables, not copies. The optimizer walks it.
    """

    def __init__(self, *, config: Config, seed: int) -> None:
        """Make every parameter and the flat parameter list.

        Every node is a leaf with one draw from a Gaussian of mean zero
        and standard deviation ``INIT_STD``. One ``random.Random(seed)``
        makes every draw, so the same seed gives the same parameters.

        Args:
            config: The five sizes.
            seed: The seed of the generator that draws the parameters.
        """
        self.config: Config = config
        rng: random.Random = random.Random(seed)
        n_hidden: int = MLP_RATIO * config.n_embd

        def random_vector(*, cols: int) -> Vector:
            """Return ``cols`` leaf nodes, one draw each."""
            return [
                Node(forward_val=rng.gauss(mu=0.0, sigma=INIT_STD)) for _ in range(cols)
            ]

        def random_matrix(*, rows: int, cols: int) -> Matrix:
            """Return ``rows`` rows of ``cols`` leaf nodes, one draw each."""
            return [random_vector(cols=cols) for _ in range(rows)]

        self.wte: Matrix = random_matrix(rows=config.vocab_size, cols=config.n_embd)
        self.wpe: Matrix = random_matrix(rows=config.context_length, cols=config.n_embd)

        self.blocks: list[Block] = [
            Block(
                wq=random_matrix(rows=config.n_embd, cols=config.n_embd),
                wk=random_matrix(rows=config.n_embd, cols=config.n_embd),
                wv=random_matrix(rows=config.n_embd, cols=config.n_embd),
                wo=random_matrix(rows=config.n_embd, cols=config.n_embd),
                fc1=random_matrix(rows=n_hidden, cols=config.n_embd),
                fc2=random_matrix(rows=config.n_embd, cols=n_hidden),
            )
            for _ in range(config.n_blocks)
        ]

        self.lm_head: Matrix = random_matrix(rows=config.vocab_size, cols=config.n_embd)

        tables: list[Matrix] = [self.wte, self.wpe]
        for block in self.blocks:
            tables.extend(
                [block.wq, block.wk, block.wv, block.wo, block.fc1, block.fc2]
            )
        tables.append(self.lm_head)

        self.params: list[Node] = [
            node for table in tables for row in table for node in row
        ]

    def forward(self, *, tokens: list[int]) -> Matrix:
        """Return one logits vector per position of ``tokens``.

        The function embeds each token as ``wte[token] + wpe[i]``, norms
        it, and then runs each block: ``x = x + attention(rmsnorm(x))``
        over the whole sequence, then ``x = x + mlp(rmsnorm(x))`` per
        position. A final norm and ``linear`` with ``lm_head`` make the
        logits. The embedding lookup is plain list indexing and makes no
        node. The function makes a new graph on every call and does not
        reset gradients.

        Args:
            tokens: The token ids of one sequence, in order. It must have
                at least one id and at most ``context_length`` ids, and
                each id must be in ``0 .. vocab_size - 1``.

        Returns:
            One vector per position, in order, each with ``vocab_size``
            nodes. The logits of position ``i`` depend on tokens ``0`` to
            ``i`` only.

        Raises:
            ValueError: If ``tokens`` is empty, has more than
                ``context_length`` ids, or has an id outside
                ``0 .. vocab_size - 1``.
        """
        if not tokens:
            raise ValueError("tokens is empty")
        if len(tokens) > self.config.context_length:
            raise ValueError(
                f"num_tokens={len(tokens)} exceeds context_length={self.config.context_length}"
            )
        for i, token in enumerate(tokens):
            if token not in range(self.config.vocab_size):
                raise ValueError(
                    f"token={token} at position={i} is not in 0...{self.config.vocab_size - 1}"
                )

        residual_stream: Matrix = [
            [te + pe for te, pe in zip(self.wte[token], self.wpe[i], strict=True)]
            for i, token in enumerate(tokens)
        ]
        residual_stream = [rmsnorm(x=row) for row in residual_stream]

        for block in self.blocks:
            normed: Matrix = [rmsnorm(x=row) for row in residual_stream]
            attention_outputs: Matrix = attention(
                x=normed,
                wq=block.wq,
                wk=block.wk,
                wv=block.wv,
                wo=block.wo,
                n_head=self.config.n_head,
            )
            residual_stream = [
                [a + b for a, b in zip(xi, yi, strict=True)]
                for xi, yi in zip(residual_stream, attention_outputs, strict=True)
            ]

            normed = [rmsnorm(x=row) for row in residual_stream]
            mlp_outputs: Matrix = [
                mlp(x=row, fc1=block.fc1, fc2=block.fc2) for row in normed
            ]
            residual_stream = [
                [a + b for a, b in zip(xi, yi, strict=True)]
                for xi, yi in zip(residual_stream, mlp_outputs, strict=True)
            ]

        normed = [rmsnorm(x=row) for row in residual_stream]
        logits: Matrix = [linear(x=row, w=self.lm_head) for row in normed]

        return logits
