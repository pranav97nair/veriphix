from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, assert_never, cast

from graphix import Plane
from graphix.clifford import Clifford
from graphix.command import BaseCommand, BaseM, BaseN, Command, CommandKind
from graphix.fundamentals import IXYZ_VALUES
from graphix.measurements import BlochMeasurement, Measurement, toggle_outcome
from graphix.pattern import Pattern
from graphix.pauli import Pauli
from graphix.rng import ensure_rng
from graphix.sim.statevec import Statevector
from graphix.simulator import MeasureMethod, PrepareMethod
from graphix.states import BasicStates, State
from typing_extensions import override

from veriphix.blind import SecretFlags
from veriphix.blind import Secrets as SecretsV2
from veriphix.blinding import SecretDatas, Secrets
from veriphix.malicious_noise_model import MaliciousNoiseModel
from veriphix.measure import DelegateMeasureMethod
from veriphix.prepare import ClientPrepareMethod as ClientPrepareMethodV2
from veriphix.protocols import FK12, Round, VerificationProtocol
from veriphix.strategies import get_graph_clifford_structure
from veriphix.verifying import ComputationRun, ResultAnalysis, Run, RunResult, TrappifiedSchemeParameters

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from collections.abc import Set as AbstractSet
    from typing import TypeVar

    from graphix.command import CommandType
    from graphix.measurements import Outcome
    from graphix.noise_models import NoiseModel
    from graphix.sim.base_backend import Backend
    from graphix.states import PlanarState
    from numpy.random import Generator
    from stim import PauliString

    _StateT = TypeVar("_StateT")
    type Node = int
    type InputNode = int
    type MeasuredNode = int
    type OutputNode = int
    type Domain = frozenset[MeasuredNode]


@dataclass
class ByProduct:
    z_domain: set[int]
    x_domain: set[int]


def get_byproduct_db(pattern: Pattern) -> dict[int, ByProduct]:
    byproduct_db = {}
    for node in pattern.output_nodes:
        byproduct_db[node] = ByProduct(z_domain=set(), x_domain=set())

    for cmd in pattern.correction_commands():
        if cmd.node in pattern.output_nodes:
            if cmd.kind == CommandKind.Z:
                byproduct_db[cmd.node].z_domain = cmd.domain
            if cmd.kind == CommandKind.X:
                byproduct_db[cmd.node].x_domain = cmd.domain
    return byproduct_db


def remove_flow(pattern: Pattern) -> Pattern:
    clean_pattern = Pattern(pattern.input_nodes)
    for cmd in pattern:
        if cmd.kind in (CommandKind.X, CommandKind.Z):
            # If byproduct, remove it so it's not done by the server
            continue
        if cmd.kind == CommandKind.M:
            # If measure, remove measure parameters
            new_cmd: BaseCommand = BaseM(node=cmd.node)
        else:
            new_cmd = cmd
        # The type annotations require the pattern to contain only
        # elements of type `CommandType`, which excludes `BaseM`. We hope
        # pattern types will become more precise in the near future.
        # See https://github.com/TeamGraphix/graphix/issues/266
        clean_pattern.add(cast("CommandType", new_cmd))
    # See test_reorder_output_nodes.
    clean_pattern.reorder_output_nodes(pattern.output_nodes)
    return clean_pattern


def qCircuit_predicate(output_string: str) -> bool:
    return int(output_string[0]) == 1


def _check_all_measurements_in_xy(pattern: Pattern) -> None:
    for cmd_m in pattern.measurement_commands().values():
        plane = cmd_m.measurement.to_bloch().plane
        if plane != Plane.XY:
            raise ValueError(
                f"UBQC only works for measurements in plane XY: {cmd_m.node} is measured in plane {plane.name}."
            )


class Client:
    input_state: list[State]
    results: dict[int, Outcome]

    def __init__(
        self,
        pattern: Pattern,
        input_state: Iterable[State] | None = None,
        classical_output: bool = True,
        output_predicate: Callable[[str], bool] = qCircuit_predicate,
        measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        test_measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        secrets: Secrets | None = None,
        parameters: TrappifiedSchemeParameters | None = None,
        protocol: VerificationProtocol | None = None,
        autogen: bool = True,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> None:
        # See test_reject_yz_measurement.
        _check_all_measurements_in_xy(pattern)
        self.initial_pattern: Pattern = pattern
        self.classical_output = classical_output
        self.output_predicate = output_predicate

        self.input_nodes = pattern.input_nodes.copy()
        self.output_nodes = pattern.output_nodes.copy()
        self.unmeasured_nodes = set() if classical_output else self.output_nodes
        self.input_state = [BasicStates.PLUS for _ in self.input_nodes] if input_state is None else list(input_state)
        self.protocol = protocol or FK12()
        self.parameters = parameters or TrappifiedSchemeParameters(20, 20, 5)
        self.results = {}
        if autogen:
            self.preprocess_pattern(classical_output=classical_output)
            self.create_blind_patterns(
                measure_method_cls=measure_method_cls,
                test_measure_method_cls=test_measure_method_cls,
                secrets=secrets,
                rng=rng,
                stacklevel=stacklevel + 1,
            )
            self.computationRun = ComputationRun(self)

    def preprocess_pattern(self, classical_output: bool = True) -> None:
        if classical_output:
            self._add_measurement_commands(self.initial_pattern)

        og = self.initial_pattern.to_opengraph()
        self.graph = og.graph
        self.clifford_structure = get_graph_clifford_structure(self.graph)

    def create_blind_patterns(
        self,
        measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        test_measure_method_cls: Callable[[Client], MeasureMethod] | None = None,
        secrets: Secrets | None = None,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> None:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        self.measure_method = (measure_method_cls or ClientMeasureMethod)(self)
        self.test_measure_method = (test_measure_method_cls or TestMeasureMethod)(self)

        self.measurement_db = self._get_measurement_db()
        self.byproduct_db = get_byproduct_db(self._copy_pattern())

        self.secrets = secrets or Secrets()
        self.secret_datas = SecretDatas.from_secrets(
            self.secrets,
            self.graph,
            set(self.input_nodes),
            set(self.unmeasured_nodes),
            rng=rng,
        )

        self.clean_pattern = remove_flow(self.initial_pattern)
        if not self.classical_output:
            self.test_pattern = remove_flow(self.initial_pattern)
            self._add_measurement_commands(self.test_pattern)
        else:
            self.test_pattern = self.clean_pattern
        self.computation_states = self.get_computation_states()

        self.preparation_bank: dict[int, Statevector] = {}
        self.prepare_method = ClientPrepareMethod(self.preparation_bank)

    @property
    def nodes(self) -> list[int]:
        return list(self.graph.nodes)

    def _add_measurement_commands(self, pattern: Pattern) -> None:
        for onode in pattern.output_nodes:
            pattern.add(Command.M(node=onode))

    def _copy_pattern(self) -> Pattern:
        pattern_copy = Pattern(self.initial_pattern.input_nodes)
        for cmd in self.initial_pattern:
            pattern_copy.add(cmd)
        pattern_copy.standardize()
        return pattern_copy

    def _get_measurement_db(self) -> dict[int, Command.M]:
        copied_pattern = self._copy_pattern()
        return copied_pattern.measurement_commands()

    def refresh_randomness(self, rng: Generator | None = None, *, stacklevel: int = 1) -> None:
        "method to refresh random randomness using parameters from Client instantiation."

        # refresh only if secrets bool is True; False is no randomness at all.
        if self.secrets is not None:
            self.secret_datas = SecretDatas.from_secrets(
                self.secrets,
                self.graph,
                set(self.input_nodes),
                set(self.unmeasured_nodes),
                rng=rng,
                stacklevel=stacklevel + 1,
            )

    def get_computation_states(self) -> dict[int, State]:
        """Return the (unblinded) state in which each node is prepared.

        These states do not depend on the secrets: input nodes are prepared in
        the specific input state desired by the Client, and every other node is
        prepared in ``|+>`` by default. Because they are secret-independent, the
        computation states never need to be refreshed; the secret-dependent
        blinding is applied separately at the blindness level (see
        :meth:`SecretDatas.blind_qubit`).
        """
        states = {}
        for node in self.graph.nodes:
            if node in self.input_nodes:
                state = self.input_state[node]

            else:
                state = BasicStates.PLUS
            states[node] = state
        return states

    def prepare_states_virtual(self, states_dict: Mapping[int, State]) -> None:
        """
        The Client creates the qubits and blind them in its preparation_bank
        """
        for node in states_dict:
            blinded_qubit_state = self.secret_datas.blind_qubit(node=node, state=states_dict[node])
            self.preparation_bank[node] = Statevector(blinded_qubit_state)

    def prepare_states(self, backend: Backend[_StateT], states_dict: Mapping[int, State]) -> None:
        # Initializes the bank (all the nodes)
        self.prepare_states_virtual(states_dict=states_dict)
        # Server asks the backend to create them
        ## Except for the input! The Client creates them itself
        for node in self.input_nodes:
            self.prepare_method.prepare_node(backend, node)

    def sample_canvas(self, rng: Generator | None = None, *, stacklevel: int = 1) -> dict[int, Run]:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)
        N = self.parameters.comp_rounds + self.parameters.test_rounds
        computation_rounds = set(rng.choice(N, size=self.parameters.comp_rounds, replace=False))

        return {
            r: self.computationRun
            if r in computation_rounds
            else self.protocol.sample_test_run(self, rng, stacklevel=stacklevel + 1)
            for r in range(N)
        }

    def delegate_canvas(
        self,
        canvas: dict[int, Run],
        backend_cls: type[Backend[_StateT]],
        noise_model: NoiseModel | None = None,
        rng: Generator | None = None,
    ) -> dict[int, RunResult[_StateT]]:
        outcomes = {}
        for r, run in canvas.items():
            backend = backend_cls()
            if isinstance(noise_model, MaliciousNoiseModel):
                noise_model.refresh_randomness(rng=rng)
            outcomes[r] = run.delegate(backend=backend, noise_model=noise_model, rng=rng)
        return outcomes

    def analyze_outcomes(
        self, canvas: dict[int, Run], outcomes: dict[int, RunResult[_StateT]]
    ) -> tuple[bool, bool, ResultAnalysis[_StateT]]:
        result_analysis: ResultAnalysis[_StateT] = ResultAnalysis()
        for r in canvas:
            outcomes[r].analyze(result_analysis=result_analysis, client=self)

        # True if Accept, False if Reject
        traps_decision = result_analysis.nr_failed_test_rounds <= self.parameters.threshold
        # The Client decides that the instance passes the predicate if more than half of the test rounds did pass
        computation_decision = result_analysis.computation_count >= ceil(self.parameters.comp_rounds / 2)

        return traps_decision, computation_decision, result_analysis

    def decode_output_state(self, backend: Backend[_StateT]) -> None:
        """Undo the blinding on a quantum output state held by the backend.

        When the output is a classical bitstring, the ``r``/``a_N`` decryption is
        applied automatically when storing each measurement outcome (see
        :meth:`BlindMeasureMethod.store_measurement_outcome`). When the output is
        a quantum state, no measurement happens on the output nodes, so the
        one-time-pad has to be undone here by applying the byproduct corrections
        computed in :meth:`decode_output`.
        """
        for node in self.output_nodes:
            z_decoding, x_decoding = self.decode_output(node)
            if z_decoding:
                backend.correct_byproduct(Command.Z(node, set()))
            if x_decoding:
                backend.correct_byproduct(Command.X(node, set()))

    def decode_output(self, node: int) -> tuple[int, int]:
        """Compute the ``(Z, X)`` byproduct corrections to apply on an output node.

        On top of the flow-induced byproduct (the sum over the ``z_domain`` /
        ``x_domain`` dependencies), the blinding has to be undone:

        - the ``Z`` correction absorbs the ``r`` one-time-pad (applied to every
          node) and the ``a_N`` encryption (the ``X`` Pauli propagated from the
          neighbours' ``a`` secrets through the entangling ``CZ``\\ s);
        - the ``X`` correction absorbs the input node's own ``a`` encryption.
        """
        z_decoding = sum(self.results[z_dep] for z_dep in self.byproduct_db[node].z_domain) % 2
        z_decoding ^= self.secret_datas.r.get(node, 0) ^ self.secret_datas.a.a_N.get(node, 0)
        x_decoding = sum(self.results[x_dep] for x_dep in self.byproduct_db[node].x_domain) % 2
        x_decoding ^= self.secret_datas.a.a.get(node, 0)
        return z_decoding, x_decoding


class ClientPrepareMethod(PrepareMethod):
    def __init__(self, preparation_bank: dict[int, Statevector]) -> None:
        self.__preparation_bank = preparation_bank

    def prepare_node(self, backend: Backend[_StateT], node: int) -> None:
        """Prepare a node."""
        backend.add_nodes(nodes=[node], data=self.__preparation_bank[node])

    @override
    def prepare(self, backend: Backend[_StateT], cmd: BaseN, rng: Generator | None = None) -> None:
        """Prepare a node."""
        self.prepare_node(backend, cmd.node)


class BlindMeasureMethod(MeasureMethod):
    def __init__(self, client: Client):
        self._client = client

    @override
    def measurement_outcome(self, node: int) -> Outcome:
        raise ValueError("Server cannot have access to measurement results")

    @override
    def store_measurement_outcome(self, node: int, result: Outcome) -> None:
        # The Server returns the outcome of the *blinded* measurement; the Client
        # decrypts it with `xor r xor a_N` before storing it. `r` is the
        # one-time-pad applied to every node and `a_N` is the X encryption
        # propagated from the neighbours' `a` secrets.
        flip_value = self._client.secret_datas.r.get(node, 0) ^ self._client.secret_datas.a.a_N.get(node, 0)
        if flip_value:
            result = toggle_outcome(result)
        self._client.results[node] = result


class ClientMeasureMethod(BlindMeasureMethod):
    @override
    def describe_measurement(self, cmd: BaseM) -> BlochMeasurement:
        parameters = self._client.measurement_db[cmd.node]

        # Extract secrets from Client
        a_value = self._client.secret_datas.a.a.get(cmd.node, 0)

        # Extract signals and compute the angle for the computation
        s_signal = sum(self._client.results[j] for j in parameters.s_domain) % 2
        t_signal = sum(self._client.results[j] for j in parameters.t_domain) % 2
        measurement = parameters.measurement
        if s_signal:
            measurement = measurement.clifford(Clifford.X)
        if t_signal:
            measurement = measurement.clifford(Clifford.Z)
        bloch = measurement.to_bloch()
        # Compensate the blinding inside the measurement angle: the `a` encryption
        # flips the angle sign (it commutes the X Pauli through the measurement)
        # and the `theta` encryption is added back via `blind_angle`.
        angle = (-1) ** a_value * bloch.angle + self._client.secret_datas.blind_angle(cmd.node)
        return BlochMeasurement(angle, bloch.plane)


class TestMeasureMethod(BlindMeasureMethod):
    @override
    def describe_measurement(self, cmd: BaseM) -> Measurement:
        # Blind the angle using the Client's secrets
        angle = self._client.secret_datas.blind_angle(cmd.node)

        return Measurement.XY(angle)


@dataclass
class DelegationRounds:
    n_compute: int
    n_test: int


@dataclass
class DelegationParameters[Data]:
    rounds: DelegationRounds
    strategy: ...  # VerificationStrategy[Data] # or DelegationStrategy?


@dataclass
class Runnable:  # tobe renamed Run
    public: Pattern
    measure_method: MeasureMethod
    prepare_method: PrepareMethod


type Test = frozenset[frozenset[int]]

type RunOrTestRun = Runnable | tuple[Runnable, Test]


@dataclass(frozen=True)
class ClientV2:
    secret_flags: SecretFlags = dataclasses.field(default_factory=SecretFlags)

    def build_run(
        self,
        clean_pattern: CleanPattern,
        states_mapping: Mapping[Node, State],
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> Runnable:
        # [MUN] We are standardizing every time we generate a run, we should store the open graph in clean_pattern
        secrets = SecretsV2(clean_pattern.public.to_opengraph(), self.secret_flags, rng=rng, stacklevel=stacklevel + 1)

        # [MUN] The two commented lines below are pseudocode from meeting with Sami:
        # #decoding_keys = Blindness.propagate(secrets.a, secrets.r, public.graph) # -> r'(i) = r(i) ^ a_N(i)
        # #measure_method = MeasureMethod(private, secrets.a, secrets.theta, decoding_keys)
        # I believe that "Blindness.propagate" is currently done by DelegateMeasureMethod.store_measurement_outcome.
        # Eventually, we'll refactor that, so that all the r' keys are precomputed and passed to DelegateMeasureMethod

        # [MUN] Since we pass the "measure_method" to the backend, won't the server have access to the private data then?
        measure_method = DelegateMeasureMethod(secrets, clean_pattern.private)

        # states_mapping maps a blinded state to ALL nodes, inputs and ancillas
        preparation_bank = self.blind_states(secrets, states_mapping)
        prepare_method = ClientPrepareMethodV2(preparation_bank)

        # At this stage, input nodes have not been addded the backend, but the prepare method has the capability to do so.
        # This is the responsability of the method who runs the protocol

        return Runnable(clean_pattern.public, measure_method, prepare_method)

    def stream_runs(
        self,
        clean_pattern: CleanPattern,
        input_state: Sequence[PlanarState],
        protocol: VerificationProtocol,
        rng: Generator | None = None,
        *,
        stacklevel: int = 1,
    ) -> Iterator[RunOrTestRun]:
        # Add support for input_state = None

        computation_states_mapping = ClientV2.generate_computation_states(clean_pattern.public, input_state)
        nodes = clean_pattern.public.nodes()

        # [MUN]
        # This was in Sami's notes. What is it for??
        #
        # test_private_data={
        # i: (0, [], [])
        # for i in public.nodes
        # }
        sampler = protocol.strategy.build_sampler()

        for r in protocol.rounds(
            rng, stacklevel=stacklevel + 1
        ):  # shuffle(['c']*protocol.n_comp_rounds + ['t']*protocol.n_test_rounds)
            match r:
                case Round.Computation:
                    yield self.build_run(clean_pattern, computation_states_mapping, rng, stacklevel=stacklevel + 1)
                case Round.Test:
                    test = sampler.sample_test(rng, stacklevel=stacklevel + 1)
                    test_states_mapping = ClientV2.generate_test_states(nodes, test.stabilizer)  # Unblinded states
                    yield test, self.build_run(clean_pattern, test_states_mapping, rng, stacklevel=stacklevel + 1)
                case _:
                    assert_never(r)

    @staticmethod
    def generate_computation_states(public: Pattern, input_state: Sequence[PlanarState]) -> dict[Node, PlanarState]:
        """Return the (unblinded) state in which each node is prepared.

        These states do not depend on the secrets: input nodes are prepared in
        the specific input state desired by the Client, and every other node is
        prepared in ``|+>`` by default. Because they are secret-independent, the
        computation states never need to be refreshed; the secret-dependent
        blinding is applied separately at the blindness level (see
        :meth:`SecretDatas.blind_qubit`).
        """
        states_mapping = dict(zip(public.input_nodes, input_state, strict=True))
        states_mapping.update((node, BasicStates.PLUS) for node in public.nodes() if node not in states_mapping)

        return states_mapping

    # [MUN]
    # This implementation doesn't require graph nodes to be indexed from 0 to n-1 qubits
    # Code in TestRun.delegate assumed so
    @staticmethod
    def generate_test_states(nodes: AbstractSet[int], stabilizer: PauliString) -> dict[Node, PlanarState]:
        return dict(zip(nodes, (Pauli(IXYZ_VALUES[pauli]).eigenstate() for pauli in stabilizer), strict=True))  # type: ignore[attr-defined]

    def blind_states(self, secrets: SecretsV2, states_mapping: Mapping[Node, State]) -> dict[Node, Statevector]:
        """
        The Client creates the qubits and blind them in its preparation_bank
        """
        # Should we have "blind_state" instead of "blind_qubit" ?
        return {
            node: Statevector(secrets.blind_qubit(node=node, state=state)) for node, state in states_mapping.items()
        }


@dataclass
class PrivateData:
    """The extracted private information from an MBQC pattern - a mapping of measured nodes to the measurement,
    and two mappings of output nodes to domains representing the pattern's X and Z correction functions."""

    measurements: dict[MeasuredNode, Command.M]
    x_byproducts: dict[OutputNode, Domain]
    z_byproducts: dict[OutputNode, Domain]


@dataclass
class CleanPattern:
    """An MBQC pattern without any measurement angles and by-products. The "cleaned" pattern is represented by the `public` attribute, and the extracted information is stored inside the `private` parameter."""

    public: Pattern
    private: PrivateData

    @staticmethod
    def from_pattern(pattern: Pattern) -> CleanPattern:
        """
        Extract the measurement angles and by-products from an MBQC pattern.

        Parameters
        ----------
        pattern: Pattern
            An MBQC pattern to be delegated blindly to a Server.

        Returns
        -------
        CleanPattern
            A data structure containing the extracted information as an attribute called `private` and the
            "cleaned" pattern as `public`.
        """
        public = Pattern(pattern.input_nodes)
        measurements = {}
        x_byproducts = {}
        z_byproducts = {}

        for cmd in pattern:
            match cmd.kind:
                # The following replicates the combined action of the methods `get_byproduct_db`
                # and `remove_flow`, wherein all X and Z commands are excluded from the new pattern,
                # but the by-products are extracted only if they apply to output nodes.
                # To be discussed why we do not extract corrections on non-output nodes.
                case CommandKind.Z:
                    if cmd.node in pattern.output_nodes:
                        z_byproducts[cmd.node] = frozenset(cmd.domain)
                case CommandKind.X:
                    if cmd.node in pattern.output_nodes:
                        x_byproducts[cmd.node] = frozenset(cmd.domain)
                case CommandKind.M:
                    measurements[cmd.node] = cmd
                    new_cmd: BaseCommand = BaseM(node=cmd.node)
                case _:
                    new_cmd = cmd

            public.add(cast("CommandType", new_cmd))

        public.reorder_output_nodes(pattern.output_nodes)
        private = PrivateData(measurements, x_byproducts, z_byproducts)

        return CleanPattern(public, private)
