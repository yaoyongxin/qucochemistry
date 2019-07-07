import pytest
from pyquil import Program
from pyquil.gates import RY
from pyquil.paulis import PauliSum, PauliTerm

from qucochemistry.vqe import VQEexperiment
from . utils import start_qvm, HAMILTONIAN, NSHOTS


@pytest.fixture
def vqe_strategy():
    # default value for the strategy
    return "custom_program"


@pytest.fixture
def vqe(vqe_strategy, vqe_tomography):

    def custom_ansatz_program():
        prog = Program()
        theta = prog.declare('theta', memory_type='REAL', memory_size=1)
        prog.inst(RY(theta[0], 0))
        return prog

    _vqe = None
    if vqe_strategy == "custom_program":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method='WFS',
                             strategy=vqe_strategy,
                             parametric=False,
                             tomography=vqe_tomography,
                             shotN=NSHOTS)
        _vqe.set_custom_ansatz(custom_ansatz_program())

    return _vqe


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
def test_variational_custom_program_strategy(vqe):
    pass

