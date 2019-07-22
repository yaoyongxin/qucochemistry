import pytest
from pyquil import Program
from pyquil.gates import RY
from scipy.optimize import OptimizeResult
import numpy as np
import scipy.sparse as sp_sparse
import scipy.stats as sp_stats


def parametric_ansatz_program():
    prog = Program()
    theta = prog.declare('theta', memory_type='REAL', memory_size=1)
    prog.inst(RY(theta[0], 0))
    return prog


@pytest.fixture
def circuit(is_sparse):
    return sp_stats.unitary_group.rvs(4) if not is_sparse else sp_sparse.coo_matrix(sp_stats.unitary_group.rvs(4))


@pytest.mark.parametrize('vqe_strategy', ["custom_program", "UCCSD"])
@pytest.mark.parametrize('vqe_method', ["WFS"])
def test_variational_parametric_end_to_end(vqe_parametric, local_qvm_quilc):

    # FIXME: the local_qvm_quilc fixture does not work with this option
    if vqe_parametric.strategy == "UCCSD":
        pytest.skip("local_qvm function currently not working with UCCSD strategy")

    if vqe_parametric.strategy == "custom_program":
        vqe_parametric.set_custom_ansatz(parametric_ansatz_program())
    obj_fn_initial = vqe_parametric.objective_function()

    theta = [0.5] if vqe_parametric.strategy == "custom_program" else None
    vqe_parametric.start_vqe(theta=theta, maxiter=10)
    res = vqe_parametric.get_results()
    assert isinstance(res, OptimizeResult)
    assert res.fun <= obj_fn_initial


@pytest.mark.parametrize('vqe_strategy', ["UCCSD"])
@pytest.mark.parametrize('vqe_method', ["linalg"])
def test_variational_parametric_end_to_end_linalg(vqe_parametric, local_qvm_quilc):

    gs_initial = vqe_parametric.get_exact_gs()
    obj_fn_initial = vqe_parametric.objective_function()

    vqe_parametric.start_vqe(maxiter=10)
    res = vqe_parametric.get_results()
    assert isinstance(res, OptimizeResult)
    assert np.isclose(res.fun, gs_initial, 1e-9)
    assert res.fun <= obj_fn_initial


@pytest.mark.parametrize('vqe_strategy', ["UCCSD"])
@pytest.mark.parametrize('vqe_method', ["linalg"])
@pytest.mark.parametrize('is_sparse', [True, False])
def test_unitary_matrix_linalg_sparse(vqe_parametric, circuit, local_qvm_quilc):

    gs_initial = vqe_parametric.get_exact_gs()
    obj_fn_initial = vqe_parametric.objective_function()
    vqe_parametric.set_circuit_unitary(circuit)

    vqe_parametric.start_vqe()
    res = vqe_parametric.get_results()
    assert np.isclose(res.fun, gs_initial, 1e-9)
    assert res.fun <= obj_fn_initial

