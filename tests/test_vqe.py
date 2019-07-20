import numpy as np

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
    "custom_program": (-5.792, -0.9)
}


# tests

@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
def test_static_custom_program_strategy(vqe_parametric):
    gs = vqe_parametric.get_exact_gs()
    ec = vqe_parametric.objective_function()
    assert np.isclose(gs, CUSTOM_APPROX_GS, atol=1e-3)
    assert np.isclose(ec, CUSTOM_APPROX_EC, atol=1e-1)

    custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN2])
    gs_with_ham = vqe_parametric.get_exact_gs(hamiltonian=custom_ham)
    assert gs_with_ham != gs


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF', 'UCCSD'])
def test_strategy_parametric(vqe_parametric):
    ec = vqe_parametric.objective_function()
    gs = vqe_parametric.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_parametric.strategy][0], atol=1e-2)
    assert np.isclose(ec, ground_states[vqe_parametric.strategy][1], atol=1e-2)


@start_qvm
@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['UCCSD'])
@pytest.mark.parametrize('vqe_method', ["WFS", "Numpy"])
def test_strategy_fixed(vqe_fixed):
    ec = vqe_fixed.objective_function()
    gs = vqe_fixed.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_fixed.strategy][0], atol=1e-2)
    assert np.isclose(ec, ground_states[vqe_fixed.strategy][1], atol=1e-2)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['custom_program', 'HF', 'UCCSD'])
def test_get_qubit_req(vqe_parametric):
    nq = vqe_parametric.get_qubit_req()
    assert nq == NQUBITS_H if vqe_parametric.strategy != 'UCCSD' else NQUBITS_H2


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF'])
def test_save_program(vqe_parametric):
    cwd = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(cwd, "resources", "test.prog")
    vqe_parametric.save_program(fname)
    with open(fname, "r") as f:
        prog_str = f.read()
        expected_prog = Program(X(0))
        assert Program(prog_str) == expected_prog
    os.remove(fname)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF'])
def test_get_circuit(vqe_parametric):
    expected_prog = Program(X(0))
    assert vqe_parametric.get_circuit() == expected_prog


@pytest.mark.parametrize('vqe_tomography', [True])
@pytest.mark.parametrize('vqe_strategy', ['HF', 'UCCSD', 'custom_program'])
def test_set_tomo_nshots(vqe_parametric):
    nshots = 100
    vqe_parametric.set_tomo_shots(nshots)
    assert vqe_parametric.shotN == nshots
