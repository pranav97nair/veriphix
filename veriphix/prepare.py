from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphix.simulator import PrepareMethod
from typing_extensions import override

if TYPE_CHECKING:
    from typing import TypeVar

    from graphix.command import BaseN
    from graphix.sim.base_backend import Backend
    from graphix.sim.statevec import Statevector
    from numpy.random import Generator

    _StateT = TypeVar("_StateT")
    type Node = int


@dataclass
class ClientPrepareMethod(PrepareMethod):
    preparation_bank: dict[Node, Statevector]  # Should map every node in the graph, including inputs

    def prepare_node(self, backend: Backend[_StateT], node: Node) -> None:
        """Prepare a node."""
        backend.add_nodes(nodes=[node], data=self.preparation_bank[node])

    @override
    def prepare(self, backend: Backend[_StateT], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""
        self.prepare_node(backend, cmd.node)
