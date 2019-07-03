import pytest
import os
import re
from dataclasses import dataclass
from openfermion import MolecularData
from qucochemistry.circuits import uccsd_ansatz_circuit_parametric


# WARNING: this feature requires Python 3.7
@dataclass
class GateRead:
    name: str
    qubits: list


@pytest.fixture
def sample_molecule():
    cwd = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(cwd, "resources", "H2.hdf5")
    molecule = MolecularData(filename=fname)
    return molecule


@pytest.fixture
def h2_gates():
    cwd = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(cwd, "resources", "H2.gates")
    gates = []
    with open(fname, "r") as f:
        lines = f.readlines()
        for l in lines:
            instruction = l.strip("\n").split(" ")
            gate_name = re.match(r'\w+', instruction[0]).group(0)
            qubits = [int(x) for x in instruction[1:]]
            gates.append(GateRead(gate_name, qubits))
    return gates


def test_uccsd_ansatz_circuit_parametric(sample_molecule, h2_gates):
    ansatz = uccsd_ansatz_circuit_parametric(sample_molecule.n_orbitals, sample_molecule.n_electrons)
    for i, ins in enumerate(ansatz.instructions):
        if i == 0:
            continue
        gate = GateRead(ins.name, [int(x.out()) for x in ins.qubits])
        assert h2_gates[i-1] == gate


def test_exponentiate_commuting_pauli_sum_parametric():
    pass

