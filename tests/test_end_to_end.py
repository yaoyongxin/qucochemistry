from scipy.optimize import OptimizeResult
import numpy as np

from . utils import *


@start_qvm
@pytest.mark.parametrize('vqe_strategy', ["custom_program", "UCCSD"])
def test_variational_parametric_end_to_end(vqe_parametric):

    if vqe_parametric.strategy == "custom_program":
        vqe_parametric.set_custom_ansatz(parametric_ansatz_program())
    obj_fn_initial = vqe_parametric.objective_function()

    theta = [0.5] if vqe_parametric.strategy == "custom_program" else None
    vqe_parametric.start_vqe(theta=theta, maxiter=10)
    res = vqe_parametric.get_results()
    assert isinstance(res, OptimizeResult)
    assert res.fun <= obj_fn_initial


@start_qvm
@pytest.mark.parametrize('vqe_strategy', ["UCCSD"])
@pytest.mark.parametrize('vqe_method', ["linalg"])
def test_variational_parametric_end_to_end_linalg(vqe_parametric):

    gs_initial = vqe_parametric.get_exact_gs()
    obj_fn_initial = vqe_parametric.objective_function()

    vqe_parametric.start_vqe(maxiter=10)
    res = vqe_parametric.get_results()
    assert isinstance(res, OptimizeResult)
    assert np.isclose(res.fun, gs_initial, 1e-9)
    assert res.fun <= obj_fn_initial
