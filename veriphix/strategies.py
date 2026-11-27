from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import networkx as nx
import stim
from graphix.fundamentals import Axis
from graphix.rng import ensure_rng
from stim import PauliString
from typing_extensions import override

from veriphix.protocols import GraphStabilizer, OddPairGeneratorFn, odd_pair_generators_bfs

type Trap = frozenset[int]
type Node = int
type Graph = nx.Graph[int]

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet
    from typing import Literal, TypeVar

    from numpy.random import Generator

    _T = TypeVar("_T")


@dataclass
class Test:
    graph: Graph
    traps: frozenset[Trap]
    stabilizer: PauliString


def get_graph_clifford_structure(graph: Graph) -> stim.Tableau:
    circuit = stim.Circuit()
    for edge in graph.edges:
        i, j = edge
        circuit.append_from_stim_program_text(f"CZ {i} {j}")
    return circuit.to_tableau()


def merge(strings: list[PauliString]) -> PauliString:
    n = len(strings)
    l = len(strings[0])
    common_string = strings[0]
    for i in range(1, n):
        common_string.sign *= strings[i].sign
        for j in range(l):
            if (common_string[j] != 0 and strings[i][j] != 0) and (common_string[j] != strings[i][j]):
                raise ValueError("Traps are not compatible.")
            if common_string[j] == 0 and strings[i][j] != 0:
                common_string[j] = strings[i][j]
    return common_string


def disjoint_sets(sets: Iterable[AbstractSet[_T]]) -> bool:
    """Check whether given sets are pairwise disjoint."""
    seen: set[_T] = set()
    for s in sets:
        if seen & s:
            return False
        seen |= s
    return True


# [MUN] TODO: Check if we can use Graphix PauliString to drop stim dep.
def build_stabilizer(traps: frozenset[Trap], graph: Graph, measurement_basis: Axis = Axis.X) -> PauliString:
    nqubits = len(graph.nodes)
    clifford_structure = get_graph_clifford_structure(graph)

    # Registry of Axis objects and their string labels, for stim compliance
    axis_labels: dict[Axis, Literal["X", "Y", "Z"]] = {
        Axis.X: "X",
        Axis.Y: "Y",
        Axis.Z: "Z",
    }

    # Build the PauliStrings representing the individual measurement of each trap qubit
    measurement_strings = [
        PauliString([axis_labels[measurement_basis] if i in trap else "I" for i in range(nqubits)]) for trap in traps
    ]

    # Conjugate each measurement
    conjugated_measurements = [clifford_structure.inverse()(meas) for meas in measurement_strings]
    return merge(conjugated_measurements)


class TestSampler(ABC):
    """Base class for test samplers."""

    @abstractmethod
    def sample_test(self, rng: Generator | None = None, *, stacklevel: int = 1) -> Test:
        pass


@dataclass
class EnumeratedTestSampler(TestSampler):
    tests: Sequence[Test]

    @override
    def sample_test(self, rng: Generator | None = None, *, stacklevel: int = 1) -> Test:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        return self.tests[int(rng.integers(len(self.tests)))]


@dataclass
class RandomTestSampler(TestSampler):
    graph: Graph

    @override
    def sample_test(self, rng: Generator | None = None, *, stacklevel: int = 1) -> Test:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        nodes = self.graph.nodes
        n = len(nodes)
        # The probability of needing more than k iterations is 2**(-n * k),
        # so the probability of iterating multiple times decreases exponentially.
        while True:
            bits = rng.integers(0, 2, size=n, dtype=bool)
            if bits.any():
                break
        random_nodes: frozenset[int] = frozenset(node for node, b in zip(nodes, bits, strict=True) if b)
        traps = frozenset({random_nodes})

        return Test(graph=self.graph, traps=traps, stabilizer=build_stabilizer(traps, self.graph))


@dataclass
class VerificationStrategy[Graph](ABC):
    """Base class for verification strategies."""

    graph: Graph

    @abstractmethod
    def build_sampler(self) -> TestSampler:
        pass


@dataclass
class Dummyless(VerificationStrategy[Graph]):
    odd_pair_generator: OddPairGeneratorFn = odd_pair_generators_bfs

    @override
    def build_sampler(self) -> EnumeratedTestSampler:
        # Step 1: build per-node GraphStabilizers (dict keyed by node id)
        stabdict: dict[int, GraphStabilizer] = {
            node: GraphStabilizer(
                node_indices={node},
                string=build_stabilizer(traps=frozenset({frozenset({node})}), graph=self.graph),
            )
            for node in self.graph.nodes
        }

        # Step 2: Rfull = product of all S_v
        n_qubits = len(self.graph.nodes)
        identity = GraphStabilizer(node_indices=set(), string=stim.PauliString(n_qubits))
        rfull: GraphStabilizer = reduce(mul, stabdict.values(), identity)

        # (removing S_v leaves I at v and flips Z count on its neighbours — stays in I/X/Y)
        # Step 3a: for each even-degree node v, R\v = Rfull * S_v
        generators: list[GraphStabilizer] = [rfull * stabdict[v] for v, degree in self.graph.degree if degree % 2 == 0]

        # Step 3b: one R\(u,w) generator per odd-degree node pair
        generators.extend(self.odd_pair_generator(self.graph, stabdict, rfull))

        # Step 4: create tests — each generator's node_indices form one multi-qubit trap
        tests: list[Test] = []
        for gs in generators:
            traps = frozenset({frozenset(gs.node_indices)})
            stabilizer = build_stabilizer(traps, self.graph)
            tests.append(Test(graph=self.graph, traps=traps, stabilizer=stabilizer))

        return EnumeratedTestSampler(tests)


@dataclass
class FK12(VerificationStrategy[Graph]):
    manual_coloring: Sequence[set[Node]] | None = None

    @override
    def build_sampler(self) -> EnumeratedTestSampler:
        if self.manual_coloring is None:
            coloring = nx.coloring.greedy_color(self.graph, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(self.graph.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_coloring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*self.manual_coloring)
            if not color_union == set(self.graph.nodes):
                raise ValueError("The provided colouring does not include all the nodes of the graph.")
            # check that colors are two by two disjoint
            if not disjoint_sets(self.manual_coloring):
                raise ValueError(
                    "The provided colouring is not a proper colouring i.e the same node belongs to at least two colours."
                )

            nodes_by_color = {i: list(c) for i, c in enumerate(self.manual_coloring)}
            colors = set(nodes_by_color.keys())

        # Create the tests : one per color
        tests: list[Test] = []
        for color in colors:
            # 1 color = 1 test = 1 collection of single-qubit traps
            traps = frozenset(frozenset([colored_node]) for colored_node in nodes_by_color[color])
            stabilizer = build_stabilizer(traps, self.graph)
            tests.append(Test(graph=self.graph, traps=traps, stabilizer=stabilizer))

        return EnumeratedTestSampler(tests)


@dataclass
class RandomTraps(VerificationStrategy[Graph]):
    @override
    def build_sampler(self) -> RandomTestSampler:
        return RandomTestSampler(self.graph)
