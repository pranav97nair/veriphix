from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx
import stim
from graphix.fundamentals import Axis
from graphix.rng import ensure_rng
from stim import PauliString
from typing_extensions import override

from veriphix.protocols import OddPairGeneratorFn, odd_pair_generators_bfs, GraphStabilizer

type Trap = frozenset[int]
type Node = int
type Data = nx.Graph | set[Node]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.random import Generator


@dataclass
class Test:
    graph: nx.Graph[int]
    traps: frozenset[Trap]
    stabilizer: PauliString


def get_graph_clifford_structure(graph: nx.Graph[int]) -> stim.Tableau:
    circuit = stim.Circuit()
    for edge in graph.edges:
        i, j = edge
        circuit.append_from_stim_program_text(f"CZ {i} {j}")
    return circuit.to_tableau()


# [MUN] TODO: Check if we can use Graphix PauliString to drop stim dep.
# [PGN] TODO: Fix issue with passing str(Axis) to PauliString
def build_stabilizer(traps: frozenset[Trap], graph: nx.Graph[int], measurement_basis: Axis = Axis.X) -> PauliString:
    nqubits = len(test.graph)
    clifford_structure = get_graph_clifford_structure(graph)

    # Build the PauliStrings representing the individual measurement of each trap qubit
    measurement_strings = [
        PauliString([str(measurement_basis) if i in trap else "I" for i in range(nqubits)]) for trap in traps
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
    tests: list[Test]

    @override
    def sample_test(self, rng: Generator | None = None, *, stacklevel: int = 1) -> Test:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        return self.tests[rng.integers(len(self.tests))]


class RandomTestSampler(TestSampler):
    @override
    def sample_test_run(self) -> Test:
        pass


@dataclass
class VerificationStrategy[Data](ABC):
    """Base class for verification strategies."""

    data: Data

    @abstractmethod
    def build_sampler(self) -> TestSampler:
        pass


@dataclass
class Dummyless(VerificationStrategy[nx.Graph]):
    odd_pair_generator: OddPairGeneratorFn = odd_pair_generators_bfs

    @override
    def build_sampler(self) -> EnumeratedTestSampler:
        # Step 1: build per-node GraphStabilizers (dict keyed by node id)
        stabdict: dict[int, GraphStabilizer] = {
            node: GraphStabilizer(
                node_indices={node},
                string=build_stabilizer(traps=frozenset(frozenset({node})), graph=self.data),
            )
            for node in self.data.nodes
        }

        # Step 2: Rfull = product of all S_v
        n_qubits = len(client.clifford_structure)
        identity = GraphStabilizer(node_indices=set(), string=stim.PauliString(n_qubits))
        rfull: GraphStabilizer = reduce(mul, stabdict.values(), identity)

        # (removing S_v leaves I at v and flips Z count on its neighbours — stays in I/X/Y)
        # Step 3a: for each even-degree node v, R\v = Rfull * S_v
        generators: list[GraphStabilizer] = [
            rfull * stabdict[v] for v, degree in client.graph.degree if degree % 2 == 0
        ]

        # Step 3b: one R\(u,w) generator per odd-degree node pair
        generators.extend(self.odd_pair_generator(client.graph, stabdict, rfull))

        # Step 4: build TestRuns — each generator's node_indices form one multi-qubit trap
        return [TestRun(client=client, traps=frozenset({frozenset(gs.node_indices)})) for gs in generators]


@dataclass
class FK12(VerificationStrategy[nx.Graph]):
    manual_coloring: Sequence[set[Node]] | None = None

    @override
    def build_sampler(self) -> EnumeratedTestSampler:
        if self.manual_colouring is None:
            coloring = nx.coloring.greedy_color(self.data, strategy="largest_first")
            colors = set(coloring.values())
            nodes_by_color: dict[int, list[int]] = {c: [] for c in colors}
            for node in sorted(self.data.nodes):
                color = coloring[node]
                nodes_by_color[color].append(node)
        else:
            # checks that manual_colouring is a proper colouring
            ## first check uniion is the whole graph
            color_union = set().union(*self.manual_colouring)
            if not color_union == set(self.data.nodes):
                raise ValueError("The provided colouring does not include all the nodes of the graph.")
            # check that colors are two by two disjoint
            if not disjoint_sets(self.manual_colouring):
                raise ValueError(
                    "The provided colouring is not a proper colouring i.e the same node belongs to at least two colours."
                )

            nodes_by_color = {i: list(c) for i, c in enumerate(self.manual_colouring)}
            colors = set(nodes_by_color.keys())

        # Create the tests : one per color
        tests: list[Test] = []
        for color in colors:
            # 1 color = 1 test = 1 collection of single-qubit traps
            traps = frozenset(frozenset([colored_node]) for colored_node in nodes_by_color[color])
            stabilizer = build_stabilizer(traps, self.data)
            tests.append(Test(graph=self.data, traps=traps, stabilizer=stabilizer))

        return EnumeratedTestSampler(tests)


@dataclass
class RandomTraps(VerificationStrategy[set[Node]]):
    rng: Generator | None = None

    @override
    def build_sampler(self) -> RandomTestSampler:
        pass
