import pytest

from openfermion.ops import QubitOperator
from qucochemistry.utils import qubitop_to_pyquilpauli
from pyquil.paulis import PauliSum

# qubit operator with the following format:
# k = qubit number
# v = (coefficient, qubit direction)
QUBIT_OPS = {
    '0': (0.5, 'X'),
    '1': (0.3, 'Z'),
    '2': (0.8, 'Y')
}


@pytest.fixture
def qops():
    qubit_ops = QubitOperator()
    for k, v in QUBIT_OPS.items():
        qubit_ops += v[0] * QubitOperator(f"{v[1]}{k}")
    return qubit_ops


def test_quops_to_pauli(qops):

    pauli = qubitop_to_pyquilpauli(qops)
    assert isinstance(pauli, PauliSum)

    true_qubits = [int(x) for x in QUBIT_OPS.keys()]
    true_values = list(QUBIT_OPS.values())

    for ind, term in enumerate(pauli.terms):

        # check that terms in the PauliSum actually corresponds
        # to the chosen operator in the QubitOperator
        coeff = term.coefficient.real
        qubit = list(term.operations_as_set())[0][0]
        ops = list(term.operations_as_set())[0][1]

        assert coeff == true_values[ind][0]
        assert qubit == true_qubits[ind]
        assert ops == true_values[ind][1]


def test_pauli_to_quops():
    pass


def test_pauli_meas():
    pass

