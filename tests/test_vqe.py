import os
import pytest
import numpy as np
from openfermion import MolecularData
from pyquil.paulis import PauliTerm, PauliSum

from qucochemistry.vqe import VQEexperiment
from . utils import *

# constants used by the tests

CUSTOM_APPROX_GS = -5.792
CUSTOM_APPROX_EC = -0.9
HF_GS = -0.4665818495572751
UCCSD_GS = -1.1372701746609015
UCCSD_EC = -1.1372701746609015

ground_states = {
    "HF": (-0.4665818495572751, -0.4665818495572751),
    "UCCSD": (-1.1372701746609015, -1.1397672933805079),
}


# tests

@pytest.fixture
def vqe_tomography():
    # default value for the tomography flag
    return False


@pytest.fixture
def vqe_strategy():
    # default value for the strategy
    return "custom_program"


# TODO: this fixture should become a general utility functio since it is used also for other tests
@pytest.fixture
def vqe(vqe_strategy, vqe_tomography):
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
    elif vqe_strategy == "UCCSD":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H2.hdf5")
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
def test_static_custom_program_strategy(vqe):
    gs = vqe.get_exact_gs()
    ec = vqe.objective_function()
    assert np.isclose(gs, CUSTOM_APPROX_GS, atol=1e-3)
    assert np.isclose(ec, CUSTOM_APPROX_EC, atol=1e-1)

    custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN2])
    gs_with_ham = vqe.get_exact_gs(hamiltonian=custom_ham)
    assert gs_with_ham != gs


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF', 'UCCSD'])
def test_static_hf_strategy(vqe):
    ec = vqe.objective_function()
    gs = vqe.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe.strategy][0], atol=1e-2)
    assert np.isclose(ec, ground_states[vqe.strategy][1], atol=1e-2)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['custom_program', 'HF', 'UCCSD'])
def test_get_qubit_req(vqe):
    nq = vqe.get_qubit_req()
    assert nq == NQUBITS_H if vqe.strategy != 'UCCSD' else NQUBITS_H2

