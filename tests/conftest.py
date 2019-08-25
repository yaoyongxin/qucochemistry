import os
import shutil
import pytest
from openfermion import MolecularData
from pyquil import Program
from pyquil.api import get_qc, local_qvm
from pyquil.paulis import PauliSum, PauliTerm

from qucochemistry.vqe import VQEexperiment


# constants used throughout the tests

HAMILTONIAN = [
    ('Z', 0, -3.2),
    ('Z', 1, 2.3),
    ('X', 0, 1.4)
]

HAMILTONIAN2 = [
    ('Z', 0, -5.2),
    ('Z', 1, 2.3),
    ('X', 0, 1.5)
]

HAMILTONIAN_LARGE = [
    ('Z', 0, -5.2),
    ('Z', 1, 2.3),
    ('X', 2, 3.5),
    ('Y', 3, -2.2),
    ('X', 4, 2.3),
    ('Z', 5, 7.5),
    ('X', 6, 1.3),
    ('Y', 7, -2.2),
    ('X', 8, 2.3),
    ('Z', 9, 7.5),
    ('X', 10, 1.3)
]

ground_states = {
    "HF": (-0.4665818495572751, -0.4665818495572751),
    "UCCSD": (-1.1372701746609015, -1.1397672933805079),
    "custom_program": (-5.792, -0.9)
}

NSHOTS_INT = 10000
NSHOTS_SMALL = 1000
NSHOTS_FLOAT = 10000.25
NQUBITS_H = 2
NQUBITS_H2 = 4


# utilities

# FIXME: this fixture should be removed in the future
@pytest.fixture(scope="module")
def local_qvm_quilc():
    """
    Execute test with local qvm and quilc running
    """
    if shutil.which('qvm') is None or shutil.which('quilc') is None:
        yield
        # pytest.exit("The unit tests requires 'qvm' and 'quilc' "
        #             "executables to be installed locally.")
    with local_qvm() as context:
        yield context


# default fixture values

@pytest.fixture
def vqe_method():
    return "WFS"


@pytest.fixture
def vqe_tomography():
    return False


@pytest.fixture
def vqe_strategy():
    return "custom_program"


@pytest.fixture
def vqe_parametricflag():
    return True


@pytest.fixture
def is_parametric():
    return False


@pytest.fixture
def vqe_qc_backend():
    return "Nq-pyqvm"

# fixtures for generating test molecule/programs input


@pytest.fixture
def h2_programs(is_parametric):
    cwd = os.path.abspath(os.path.dirname(__file__))
    name = "H2.gates_parametric" if is_parametric else "H2.gates"
    fname = os.path.join(cwd, "resources", name)
    with open(fname, "r") as f:
        program_str = f.read()
    return Program(program_str)


@pytest.fixture
def sample_molecule():
    cwd = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(cwd, "resources", "H2.hdf5")
    molecule = MolecularData(filename=fname)
    return molecule


# fixtures for generating different flavors of VQE experiments


@pytest.fixture
def vqe_parametric(vqe_strategy, vqe_tomography, vqe_method):
    """
    Initialize a VQE experiment with a custom hamiltonian
    given as constant input
    """

    _vqe = None

    if vqe_strategy == "custom_program":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method=vqe_method,
                             strategy=vqe_strategy,
                             parametric=True,
                             tomography=vqe_tomography,
                             shotN=NSHOTS_FLOAT)
    elif vqe_strategy == "HF":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             method=vqe_method,
                             strategy=vqe_strategy,
                             parametric=True,
                             tomography=vqe_tomography,
                             shotN=NSHOTS_FLOAT)
    elif vqe_strategy == "UCCSD":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H2.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             method=vqe_method,
                             strategy=vqe_strategy,
                             parametric=True,
                             tomography=vqe_tomography,
                             shotN=NSHOTS_FLOAT)
    return _vqe


@pytest.fixture
def vqe_fixed(vqe_strategy, vqe_tomography, vqe_method):
    """
    Initialize a VQE experiment with a custom hamiltonian
    given as constant input
    """

    _vqe = None

    if vqe_strategy == "custom_program":
        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             method=vqe_method,
                             strategy=vqe_strategy,
                             parametric=False,
                             tomography=vqe_tomography,
                             shotN=NSHOTS_INT)
    elif vqe_strategy == "UCCSD":
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H2.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             method=vqe_method,
                             strategy=vqe_strategy,
                             parametric=False,
                             tomography=vqe_tomography,
                             shotN=NSHOTS_INT)
    return _vqe


@pytest.fixture
def vqe_qc(vqe_strategy, vqe_qc_backend, vqe_parametricflag, sample_molecule):
    """
    Initialize a VQE experiment with a custom hamiltonian
    given as constant input, given a QC-type backend (tomography is always set to True then)
    """

    _vqe = None
    qc = None
    vqe_cq = None

    if vqe_strategy == "custom_program":
        if vqe_qc_backend == 'Aspen-qvm':
            qc = get_qc('Aspen-4-2Q-A-qvm')
            vqe_cq = [1, 2]
        elif vqe_qc_backend == 'Aspen-pyqvm':
            qc = get_qc('Aspen-4-2Q-A-pyqvm')
            vqe_cq = [1, 2]
        elif vqe_qc_backend == 'Nq-qvm':
            qc = get_qc('2q-qvm')
        elif vqe_qc_backend == 'Nq-pyqvm':
            qc = get_qc('2q-pyqvm')

        custom_ham = PauliSum([PauliTerm(*x) for x in HAMILTONIAN])
        _vqe = VQEexperiment(hamiltonian=custom_ham,
                             qc=qc,
                             custom_qubits=vqe_cq,
                             method='QC',
                             strategy=vqe_strategy,
                             parametric=True,
                             tomography=True,
                             shotN=NSHOTS_SMALL)
    elif vqe_strategy == "HF":
        if vqe_qc_backend == 'Aspen-qvm':
            qc = get_qc('Aspen-4-2Q-A-qvm')
            vqe_cq = [1, 2]
        elif vqe_qc_backend == 'Aspen-pyqvm':
            qc = get_qc('Aspen-4-2Q-A-pyqvm')
            vqe_cq = [1, 2]
        elif vqe_qc_backend == 'Nq-qvm':
            qc = get_qc('2q-qvm')
        elif vqe_qc_backend == 'Nq-pyqvm':
            qc = get_qc('2q-pyqvm')
        cwd = os.path.abspath(os.path.dirname(__file__))
        fname = os.path.join(cwd, "resources", "H.hdf5")
        molecule = MolecularData(filename=fname)
        _vqe = VQEexperiment(molecule=molecule,
                             qc=qc,
                             custom_qubits=vqe_cq,
                             method='QC',
                             strategy=vqe_strategy,
                             parametric=vqe_parametricflag,
                             tomography=True,
                             shotN=NSHOTS_SMALL)
    elif vqe_strategy == "UCCSD":
        if vqe_qc_backend == 'Aspen-qvm':
            qc = get_qc('Aspen-4-4Q-A-qvm')
            vqe_cq = [7, 0, 1, 2]
        elif vqe_qc_backend == 'Aspen-pyqvm':
            qc = get_qc('Aspen-4-4Q-A-pyqvm')
            vqe_cq = [7, 0, 1, 2]
        elif vqe_qc_backend == 'Nq-qvm':
            qc = get_qc('4q-qvm')
        elif vqe_qc_backend == 'Nq-pyqvm':
            qc = get_qc('4q-pyqvm')

        _vqe = VQEexperiment(molecule=sample_molecule,
                             qc=qc,
                             custom_qubits=vqe_cq,
                             method='QC',
                             strategy=vqe_strategy,
                             parametric=vqe_parametricflag,
                             tomography=True,
                             shotN=NSHOTS_SMALL)
    return _vqe
