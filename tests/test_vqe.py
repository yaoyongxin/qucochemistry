import pytest
import numpy as np
import decorator
from pyquil.api import local_qvm
from pyquil.paulis import PauliTerm, PauliSum

from qucochemistry.vqe import VQEexperiment

# constants used by the tests


HAMILTONIAN = [
    ('Z', 0, -3.2),
    ('Z', 1, 2.3),
    ('X', 0, 1.4)
]
APPROX_GS = -5.792
APPROX_EC = -0.9
NSHOTS = 10000
NQUBITS = 2

# utilities


def start_qvm(fn):
    """
    This decorator ensures that in the the context where the decorated function is executed
    the following processes are running:
    >> qvm -S
    >> quilc -S

    They needed to compile and execute a program on the quantum virtual machine. When
    the context is terminated also the processes are.
    """
    def wrapper(fn, *args):
        with local_qvm():
            return fn(*args)
    return decorator.decorator(wrapper, fn)

# tests


@pytest.fixture
def hamiltonian():
    """
    Define an hamiltonian as a sum of Pauli matrices given as
    constant input for the algorithm
    """
    hamilt = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
    return hamilt


@pytest.fixture
def custom_vqe(hamiltonian):
    """
    Initialize a VQE experiment with a custom hamiltonian
    """
    vqe = VQEexperiment(hamiltonian=hamiltonian,
                        method='WFS',
                        strategy='custom_program',
                        parametric=True,
                        tomography=True,
                        shotN=NSHOTS)
    return vqe


# TODO: test also other objective function computation methods

@start_qvm
def test_hamiltonian_gs(custom_vqe):
    gs = custom_vqe.get_exact_gs()
    ec = custom_vqe.objective_function()
    assert np.isclose(gs, APPROX_GS, atol=1e-3)
    assert np.isclose(ec, APPROX_EC, atol=1e-1)


def test_get_qubit_req(custom_vqe):
    nq = custom_vqe.get_qubit_req()
    assert nq == NQUBITS
