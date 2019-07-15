import pytest
from pyquil import Program
from pyquil.gates import RY
from pyquil.paulis import PauliSum, PauliTerm

from qucochemistry.vqe import VQEexperiment
from . utils import start_qvm, HAMILTONIAN, NSHOTS


@pytest.fixture
def vqe(vqe_strategy, vqe_tomography):

    def parametric_ansatz_program():
        prog = Program()
        theta = prog.declare('theta', memory_type='REAL', memory_size=1)
        prog.inst(RY(theta[0], 0))
        return prog

    _vqe = None
    if vqe_strategy == "custom_program_parametric":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method='WFS',
                             strategy="custom_program",
                             parametric=True,
                             tomography=vqe_tomography,
                             shotN=NSHOTS)
        _vqe.set_custom_ansatz(parametric_ansatz_program())

    return _vqe


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ["custom_program_parametric"])
def test_variational_custom_program_strategy(vqe):
    # vqe.start_vqe(theta=[1], maxiter=2)
    pass
