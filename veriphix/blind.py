from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from graphix import OpenGraph, angle_to_rad
from graphix.fundamentals import ANGLE_PI
from graphix.measurements import outcome
from graphix.pauli import Pauli
from graphix.rng import ensure_rng
from graphix.sim.statevec import Statevector

if TYPE_CHECKING:
    import numpy.typing as npt
    from graphix.fundamentals import Angle
    from graphix.measurements import Outcome
    from graphix.opengraph import _AM_co
    from graphix.states import State
    from numpy.random import Generator

    from veriphix.client import InputNode, MeasuredNode, Node

    type ArrayOrNone = npt.NDArray[np.complex128] | None


@dataclass(frozen=True)
class SecretFlags:
    """Toggles for which blinding secrets are generated.

    - ``r``: the one-time-pad applied to every node, so that every result
      (including a quantum output) is masked.
    - ``a``: the ``X``-Pauli encryption, only meaningful on input nodes.
    - ``theta``: the ``Z``-rotation encryption applied to every measured node.
    """

    r: bool = True
    a: bool = True
    theta: bool = True


class Secrets:
    _r: dict[Node, Outcome]
    _a: dict[InputNode, Outcome]
    _a_neigh: dict[Node, Outcome]
    _theta: dict[MeasuredNode, Angle]

    def __init__(
        self, og: OpenGraph[_AM_co], flags: SecretFlags, rng: Generator | None = None, *, stacklevel: int = 1
    ) -> None:
        rng = ensure_rng(rng, stacklevel=stacklevel + 1)

        self._r = {node: outcome(rng.integers(2) == 1) for node in og.graph.nodes} if flags.r else {}
        self._a = {input_node: outcome(rng.integers(2) == 1) for input_node in og.input_nodes} if flags.a else {}
        self._a_neigh = (
            {node: outcome(sum([self.a(j) for j in og.neighbors(node)]) % 2 == 1) for node in og.graph.nodes}
            if flags.a
            else {}
        )
        self._theta = (
            {measured_node: rng.integers(8) * ANGLE_PI / 4 for measured_node in og.measurements} if flags.theta else {}
        )

    def r(self, node: Node) -> Outcome:
        return self._r.get(node, 0)

    def a(self, node: InputNode) -> Outcome:
        return self._a.get(node, 0)

    def a_neigh(self, node: Node) -> Outcome:
        return self._a_neigh.get(node, 0)

    def blind_angle(self, node: MeasuredNode) -> Angle:
        return self._theta.get(node, 0)

    def blind_qubit(self, node: int, state: State) -> Statevector:
        """Apply the secret-dependent blinding to a single prepared qubit.

        The unblinded ``state`` is the secret-independent computation state (the
        input state for input nodes, ``|+>`` otherwise). The blinding applies, in
        order, ``Z(theta) Z(r) X(a)`` on top of it: the ``X`` encryption ``a``
        (input nodes only), the ``Z`` one-time-pad ``r``, then the ``theta``
        Z-rotation.
        """

        def zrot_blind(node: Node) -> ArrayOrNone:
            if th := self._theta.get(node):
                return np.array([[1, 0], [0, np.exp(1j * angle_to_rad(th))]], dtype=np.complex128)
            return None

        def x_blind(node: Node) -> ArrayOrNone:
            return Pauli.X.matrix if self.a(node) else None

        def z_blind(node: Node) -> ArrayOrNone:
            return Pauli.Z.matrix if self.r(node) else None

        single_qubit_sv = Statevector(data=state, nqubit=1)
        for op in (zrot_blind(node), x_blind(node), z_blind(node)):
            if op is not None:
                single_qubit_sv.evolve_single(op=op, qubit=0)
        return single_qubit_sv
