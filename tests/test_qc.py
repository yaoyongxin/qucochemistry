import numpy as np

from . utils import *


ground_states = {
    "HF": (-0.4665818495572751, -0.4665818495572751),
    "UCCSD": (-1.1372701746609015, -1.137269793918515),
    "custom_program": (-5.792849839314596, -0.9)
}

@start_qvm
@pytest.mark.parametrize('vqe_strategy', ['UCCSD', 'HF'])
@pytest.mark.parametrize('vqe_qc_backend', ['Aspen-qvm', 'Nq-qvm'])
@pytest.mark.parametrize('vqe_parametricflag', [True, False])
def test_qc_parametric_flag(vqe_qc):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)

@start_qvm
@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
@pytest.mark.parametrize('vqe_qc_backend', ['Aspen-qvm', 'Nq-qvm'])
@pytest.mark.parametrize('vqe_parametricflag', [True])
def test_qc_custom_program(vqe_qc):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)

@start_qvm
@pytest.mark.parametrize('vqe_strategy', ['UCCSD', 'HF'])
@pytest.mark.parametrize('vqe_qc_backend', ['Nq-pyqvm'])
@pytest.mark.parametrize('vqe_parametricflag', [False])
def test_qc_custom_program(vqe_qc):
    ec = vqe_qc.objective_function()
    gs = vqe_qc.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_qc.strategy][0], atol=1e-5)
    assert np.isclose(ec, ground_states[vqe_qc.strategy][1], atol=2e-1)