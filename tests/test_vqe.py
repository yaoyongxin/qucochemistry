import os
import pytest
import numpy as np
from openfermion import MolecularData
from pyquil.paulis import PauliTerm, PauliSum

from qucochemistry.vqe import VQEexperiment
from . utils import start_qvm, HAMILTONIAN, NSHOTS, NQUBITS

# constants used by the tests

CUSTOM_APPROX_GS = -5.792
CUSTOM_APPROX_EC = -0.9
HF_GS = -0.4665818495572751


# tests


@pytest.fixture
def vqe_strategy():
    # default value for the strategy
    return "custom_program"


@pytest.fixture
def static_vqe(vqe_strategy, vqe_tomography):
    """
    Initialize a VQE experiment with a custom hamiltonian
    given as constant input
    """
    _vqe = None
    if vqe_strategy == "custom_program":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method='WFS',
                             strategy=vqe_strategy,
                             parametric=True,
                             tomography=vqe_tomography,
                             shotN=NSHOTS)
    elif vqe_strategy == "HF":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             method='WFS',
                             strategy=vqe_strategy,
                             parametric=False,
                             tomography=vqe_tomography,
                             shotN=NSHOTS)
    return _vqe


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
def test_static_custom_program_strategy(static_vqe):
    gs = static_vqe.get_exact_gs()
    ec = static_vqe.objective_function()
    assert np.isclose(gs, CUSTOM_APPROX_GS, atol=1e-3)
    assert np.isclose(ec, CUSTOM_APPROX_EC, atol=1e-1)


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF'])
def test_static_hf_strategy(static_vqe):
    ec = static_vqe.objective_function()
    gs = static_vqe.get_exact_gs()
    assert np.isclose(ec, HF_GS, 1e-9)
    assert np.isclose(ec, gs, 1e-9)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['custom_program', 'HF'])
def test_get_qubit_req(static_vqe):
    nq = static_vqe.get_qubit_req()
    assert nq == NQUBITS

