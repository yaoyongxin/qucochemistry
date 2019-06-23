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


@pytest.fixture
def hamiltonian():
    """
    Define an hamiltonian as a sum of Pauli matrices
    """
    hamilt = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
    return hamilt


@start_qvm
def test_hamiltonian_gs(hamiltonian):
    vqe = VQEexperiment(hamiltonian=hamiltonian,
                        method='WFS',
                        strategy='custom_program',
                        parametric=True,
                        tomography=True,
                        shotN=NSHOTS)

    gs = vqe.get_exact_gs()
    ec = vqe.objective_function()
    assert np.isclose(gs, APPROX_GS, atol=1e-3)
    assert np.isclose(ec, APPROX_EC, atol=1e-1)
