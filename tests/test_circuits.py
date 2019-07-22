import pytest
from numpy import ceil as np_ceil
from pyquil import Program
from scipy.special import comb

from qucochemistry.circuits import uccsd_ansatz_circuit_parametric, uccsd_ansatz_circuit


@pytest.mark.parametrize('is_parametric', [True])
def test_uccsd_ansatz_circuit_parametric(sample_molecule, h2_programs):

    ansatz = uccsd_ansatz_circuit_parametric(sample_molecule.n_orbitals, sample_molecule.n_electrons)
    assert isinstance(ansatz, Program)
    assert ansatz.out() == h2_programs.out()


def test_uccsd_ansatz_circuit(sample_molecule, h2_programs):

    n_spatial_orbitals = sample_molecule.n_orbitals
    n_occupied = int(np_ceil(sample_molecule.n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied
    n_single_amplitudes = n_occupied * n_virtual
    # make a mock-packed_amplitudes of length (2 * n_single_amplitudes)
    packed_amplitudes = [1]*(2 * n_single_amplitudes + int(comb(n_single_amplitudes, 2)))

    ansatz = uccsd_ansatz_circuit(packed_amplitudes, sample_molecule.n_orbitals,
                                  sample_molecule.n_electrons)

    assert isinstance(ansatz, Program)
    assert ansatz.out() == h2_programs.out()
