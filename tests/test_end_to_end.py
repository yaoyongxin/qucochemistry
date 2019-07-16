import os
import pytest
import numpy as np
from openfermion import MolecularData
from pyquil.paulis import PauliSum, PauliTerm
from scipy.optimize import OptimizeResult

from qucochemistry.vqe import VQEexperiment
from . utils import start_qvm, HAMILTONIAN, parametric_ansatz_program


@pytest.fixture
def vqe(vqe_strategy, vqe_tomography):

    _vqe = None
    if vqe_strategy == "custom_program":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method='WFS',
                             strategy="custom_program",
                             parametric=True,
                             tomography=vqe_tomography)
        _vqe.set_custom_ansatz(parametric_ansatz_program())
    elif vqe_strategy == "UCCSD":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H2.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             method='linalg',
                             strategy=vqe_strategy,
                             parametric=False,
                             tomography=vqe_tomography)

    return _vqe


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ["custom_program"])
def test_variational_custom_program_strategy(vqe):
    gs_initial = vqe.get_exact_gs()
    obj_fn_initial = vqe.objective_function([1])
    # vqe.start_vqe(theta=[0.5], maxiter=1)


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [False])
@pytest.mark.parametrize('vqe_strategy', ["UCCSD"])
def test_variational_molecular_ansatz_strategy(vqe):
    gs_initial = vqe.get_exact_gs()
    obj_fn_initial = vqe.objective_function()
    vqe.start_vqe(maxiter=1)
    res = vqe.get_results()
    assert isinstance(res, OptimizeResult)
    assert res.fun < gs_initial
    assert np.isclose(res.fun, obj_fn_initial, atol=1e-5)
