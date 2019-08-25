import os
import pytest
import numpy as np
from pyquil import Program
from pyquil.gates import X, Y
from pyquil.paulis import PauliSum, PauliTerm

from . conftest import ground_states, HAMILTONIAN2, NQUBITS_H, NQUBITS_H2, HAMILTONIAN_LARGE


@pytest.mark.parametrize('vqe_tomography', [True, False])
def test_static_custom_program_strategy(vqe_parametric, vqe_tomography, local_qvm_quilc):
    gs = vqe_parametric.get_exact_gs()
    ec = vqe_parametric.objective_function()
    assert np.isclose(gs, ground_states[vqe_parametric.strategy][0], atol=1e-3)
    assert np.isclose(ec, ground_states[vqe_parametric.strategy][1], atol=1e-1)

    custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN2])
    gs_with_ham = vqe_parametric.get_exact_gs(hamiltonian=custom_ham)
    assert gs_with_ham != gs


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF', 'UCCSD'])
def test_strategy_parametric(vqe_parametric, vqe_tomography, vqe_strategy, local_qvm_quilc):
    ec = vqe_parametric.objective_function()
    gs = vqe_parametric.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_parametric.strategy][0], atol=1e-2)
    assert np.isclose(ec, ground_states[vqe_parametric.strategy][1], atol=1e-2)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['UCCSD'])
@pytest.mark.parametrize('vqe_method', ["WFS", "Numpy"])
def test_strategy_fixed(vqe_fixed, vqe_tomography, vqe_strategy, vqe_method, local_qvm_quilc):
    ec = vqe_fixed.objective_function()
    gs = vqe_fixed.get_exact_gs()
    assert np.isclose(gs, ground_states[vqe_fixed.strategy][0], atol=1e-2)
    assert np.isclose(ec, ground_states[vqe_fixed.strategy][1], atol=1e-2)


def test_large_diagonalization(vqe_parametric, vqe_tomography, vqe_strategy, vqe_method, local_qvm_quilc):
    custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN_LARGE])
    gs_with_ham = vqe_parametric.get_exact_gs(hamiltonian=custom_ham)
    assert np.isclose(gs_with_ham, -37.6, atol=1e-4)


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['custom_program', 'HF', 'UCCSD'])
def test_get_qubit_req(vqe_parametric, vqe_tomography, vqe_strategy):
    nq = vqe_parametric.get_qubit_req()
    assert nq == NQUBITS_H if vqe_parametric.strategy != 'UCCSD' else NQUBITS_H2


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['HF'])
def test_save_program(vqe_parametric, vqe_tomography, vqe_strategy):
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
def test_get_circuit(vqe_parametric, vqe_tomography, vqe_strategy):
    expected_prog = Program(X(0))
    assert vqe_parametric.get_circuit() == expected_prog


@pytest.mark.parametrize('vqe_tomography', [True, False])
@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
def test_set_circuit(vqe_parametric, vqe_tomography, vqe_strategy):
    cust_ref = Program(Y(0))
    vqe_parametric.set_custom_ref_preparation(Y(0))
    assert vqe_parametric.get_circuit() == cust_ref

    cust_ansatz = Program(X(0))
    vqe_parametric.set_custom_ansatz(cust_ansatz)
    assert vqe_parametric.get_circuit() == cust_ref + cust_ansatz


@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
@pytest.mark.parametrize('vqe_method', ["linalg"])
def test_linalg_set_psi(vqe_parametric, vqe_tomography, vqe_strategy, vqe_method):

    assert vqe_parametric.initial_psi is None

    psi = [1, 2, 3, 4]
    vqe_parametric.set_initial_state(psi)

    assert vqe_parametric.initial_psi == psi


@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
@pytest.mark.parametrize('vqe_method', ["WFS"])
def test_set_initial_angles(vqe_parametric, vqe_tomography, vqe_strategy, vqe_method):
    angles = [1, 2, 3, 4]
    vqe_parametric.set_initial_angles(angles)
    assert vqe_parametric.initial_packed_amps == angles


@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
@pytest.mark.parametrize('vqe_method', ["WFS"])
def test_verbose_output(vqe_parametric, vqe_tomography, vqe_strategy, vqe_method):
    vqe_parametric.verbose_output()
    assert vqe_parametric.verbose
    vqe_parametric.verbose_output(False)
    assert not vqe_parametric.verbose
    vqe_parametric.verbose_output(True)
    assert vqe_parametric.verbose


@pytest.mark.parametrize('vqe_strategy', ['custom_program'])
@pytest.mark.parametrize('vqe_method', ["WFS"])
def test_maxiter(vqe_parametric, vqe_tomography, vqe_strategy, vqe_method):
    vqe_parametric.set_maxiter(123)
    assert vqe_parametric.maxiter == 123


@pytest.mark.parametrize('vqe_tomography', [True])
@pytest.mark.parametrize('vqe_strategy', ['HF', 'UCCSD', 'custom_program'])
def test_set_tomo_nshots(vqe_parametric, vqe_tomography, vqe_strategy):
    nshots = 100
    vqe_parametric.set_tomo_shots(nshots)
    assert vqe_parametric.shotN == nshots
