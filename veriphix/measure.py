from graphix import MeasureMethod
from veriphix.blinding import SecretDatas
from dataclasses import dataclass
import dataclasses
from graphix.measurements import BlochMeasurement, toggle_outcome


from graphix import St

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphix.measurements import Outcome
    from graphix.command import Command, BaseM

@dataclass
class DelegateMeasureMethod(MeasureMethod):

    secret_data: SecretDatas
    private: Private
    results: dict[int, Outcome] = dataclasses.field(default_factory=dict)

    @override
    def describe_measurement(self, cmd: BaseM) -> BlochMeasurement:
        parameters = self.private.measurements[cmd.node]

        # Extract secrets
        a_value = self.secret_datas.a.a.get(cmd.node, 0)

        # Extract signals and compute the angle for the computation
        s_signal = sum(self.results[j] for j in parameters.s_domain) % 2
        t_signal = sum(self.results[j] for j in parameters.t_domain) % 2
        measurement = parameters.measurement
        if s_signal:
            measurement = measurement.clifford(Clifford.X)
        if t_signal:
            measurement = measurement.clifford(Clifford.Z)

        bloch = measurement.to_bloch()
        # Compensate the blinding inside the measurement angle: the `a` encryption
        # flips the angle sign (it commutes the X Pauli through the measurement)
        # and the `theta` encryption is added back via `blind_angle`.
        angle = (-1) ** a_value * bloch.angle + self.secret_datas.blind_angle(cmd.node)
        return BlochMeasurement(angle, bloch.plane)

    @override
    def measurement_outcome(self, node: int) -> Outcome:
        raise NotImplementedError("Server cannot have access to measurement results.")

    @override
    def store_measurement_outcome(self, node: int, result: Outcome) -> None:
        # The Server returns the outcome of the *blinded* measurement; the Client
        # decrypts it with `xor r xor a_N` before storing it. `r` is the
        # one-time-pad applied to every node and `a_N` is the X encryption
        # propagated from the neighbours' `a` secrets.
        flip_value = self.secret_data.r.get(node, 0) ^ self.secret_data.a.a_N.get(node, 0)
        if flip_value:
            result = toggle_outcome(result)
        self.results[node] = result
