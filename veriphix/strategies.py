from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from graphix.fundamentals import IXYZ_VALUES
from graphix.pauli import Pauli
from graphix.rng import ensure_rng
from graphix.simulator import PatternSimulator
import networkx as nx
from stim import PauliString
from typing_extensions import override

from veriphix.protocols import OddPairGeneratorFn, odd_pair_generators_bfs

type Trap = frozenset[int]
type Node = int
type Data = nx.Graph | set[Node]


@dataclass
class Test():
    graph: nx.Graph
    traps: frozenset[Trap]
    stabilizer: PauliString
