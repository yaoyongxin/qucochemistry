import pytest

from openfermion.ops import QubitOperator
from qucochemistry.utils import qubitop_to_pyquilpauli, pyquilpauli_to_qubitop
from pyquil.paulis import PauliSum, PauliTerm

# qubit operator with the following format:
# k = qubit number
# v = (coefficient, qubit direction)
QUBIT_OPS = {
    '0': (0.5, 'X'),
    '1': (0.3, 'Z'),
    '2': (0.8, 'Y')
}

true_qubits = [int(x) for x in QUBIT_OPS.keys()]
true_values = list(QUBIT_OPS.values())


@pytest.fixture
def qops():
    qubit_ops = QubitOperator()
    for k, v in QUBIT_OPS.items():
        qubit_ops += v[0] * QubitOperator(f"{v[1]}{k}")
    return qubit_ops


@pytest.fixture
def paulis():
    terms = []
    for k, v in QUBIT_OPS.items():
        term = PauliTerm(v[1], int(k), v[0])
        terms.append(term)
    return PauliSum(terms)


def test_quops_to_pauli(qops):

    pauli = qubitop_to_pyquilpauli(qops)
    assert isinstance(pauli, PauliSum)

    for ind, term in enumerate(pauli.terms):

        # check that terms in the PauliSum actually corresponds
        # to the chosen operator in the QubitOperator instance
        coeff = term.coefficient.real
        qubit = list(term.operations_as_set())[0][0]
        ops = list(term.operations_as_set())[0][1]

        assert coeff == true_values[ind][0]
        assert qubit == true_qubits[ind]
        assert ops == true_values[ind][1]


def test_pauli_to_quops(paulis):

    qubitops = pyquilpauli_to_qubitop(paulis)
    assert isinstance(qubitops, QubitOperator)

    for ind, term in enumerate(qubitops.terms.items()):

        # check that the qubits composing the genenerated QubitOperator
        # object corresponds to the chosen PauliSum instance
        qubit = term[0][0][0]
        ops = term[0][0][1]
        coeff = term[1].real

        assert coeff == true_values[ind][0]
        assert qubit == true_qubits[ind]
        assert ops == true_values[ind][1]


@pytest.mark.xfail
def test_pauli_to_quops_failure():
    _ = qubitop_to_pyquilpauli(QUBIT_OPS)
    _ = pyquilpauli_to_qubitop(QUBIT_OPS)
