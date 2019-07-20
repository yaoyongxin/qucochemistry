from typing import List, Dict

from pyquil.quilatom import MemoryReference

from pyquil.quil import Program, percolate_declares
from pyquil.gates import X, RX, RY, MOVE
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, commuting_sets, simplify_pauli_sum,  \
    exponentiate_commuting_pauli_sum

from openfermion.hamiltonians import MolecularData
from openfermion.utils import uccsd_singlet_generator, normal_ordered
from openfermion.transforms import jordan_wigner

import numpy as np

from qucochemistry.utils import qubitop_to_pyquilpauli, uccsd_singlet_generator_with_indices


def pauli_meas(idx, op):
    r"""
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis. The available operations are the
    following:

    .. math::

        'X' = \begin{bmatrix}
                0 & \frac{-\pi}{2} \\
                \frac{-\pi}{2} & 0 
            \end{bmatrix}

        'Y' = \begin{bmatrix}
                0 & \frac{-i\pi}{2} \\
                \frac{i\pi}{2} & 0 
            \end{bmatrix}


    and :math:`'Z' = 'I' = \mathbb{I}`.

    :param int idx: the qubit index on which the measurement basis rotation is to be performed.
    :param str op: enumeration ('X', 'Y', 'Z', 'I') representing the axis of the given Pauli matrix

    :return: a `pyquil` Program representing the Pauli matrix projected onto the Z eigenbasis.
    :rtype: Program
    """
    if op == 'X':
        return Program(RY(-np.pi / 2, idx))
    elif op == 'Y':
        return Program(RX(np.pi / 2, idx))
    elif op == 'Z':
        return Program()
    elif op == 'I':
        return Program()


def augment_program_with_memory_values(quil_program, memory_map):
    """
    This function allocates the classical memory values (gate angles) to a parametric quil program in order to use it
    on a Numpy-based simulator

    :param Program quil_program: parametric quil program which would require classical memory allocation
    :param Dict memory_map: dictionary with as keys the MemoryReference or String descrbing the classical memory, and
        with items() an array of values for that classical memory

    :return: quil program with gate angles from memory_map allocated to the (originally parametric) program
    :rtype: Program
    """
    p = Program()

    # this function allocates the memory values for a parametric program correctly...

    if len(memory_map.keys()) == 0:
        return quil_program
    elif isinstance(list(memory_map.keys())[0], MemoryReference):
        for k, v in memory_map.items():
            p += MOVE(k, v)
    elif isinstance(list(memory_map.keys())[0], str):
        for name, arr in memory_map.items():
            for index, value in enumerate(arr):
                p += MOVE(MemoryReference(name, offset=index), value)
    else:
        raise TypeError("Bad memory_map type; expected Dict[str, List[Union[int, float]]].")

    p += quil_program

    return percolate_declares(p)


def exponentiate_commuting_pauli_sum_parametric(pauli_sum, term_dict, memref):
    """
    Returns a Program() (NOT A function) that maps all substituent PauliTerms and sums them into a program. NOTE: Use
    this function with care. Substituent PauliTerms in pauli_sum should commute for this to work correctly!

    :param List pauli_sum: list of Pauli terms to exponentiate.
    :param Dict term_dict: Dictionary containing as keys the Pauliterm frozensets, and as values the indices in
        packed_amplitudes (and in Memoreference pointer, same index!), corresponding to the same index in pauli_sum list
    :param MemoryReference memref: memory reference which should be inserted to generate the program

    :returns: A program that parametrizes the exponential.
    :rtype: Program()
    """

    prog = Program()
    for i, term in enumerate(pauli_sum):
        memrefindex = term_dict[term.operations_as_set()]
        prog += exponential_map(1j*term)(memref[memrefindex])

    return prog


def ref_state_preparation_circuit(molecule, ref_type='HF', cq=None):

    """
    This function returns a program which prepares a reference state to begin from with a Variational ansatz.

    :param MolecularData molecule: molecule data object containing information on HF state.
    :param str ref_type: type of reference state desired
    :param list() cq: (optional) list of qubit labels if different from standard 0 to N-1 convention

    :return: pyquil program which prepares the reference state
    :rtype: Program
    """

    if ref_type != 'HF':
        raise NotImplementedError('Currently only supports ref_type = `HF`. More to be added in future releases.')

    if not isinstance(molecule, MolecularData):
        raise TypeError('molecule should be of type MolecularData, data class from the OpenFermion.hamiltonians module')

    # initialize pyQuil program
    prog = Program()

    # flip n_electrons qubits representing spinorbitals with the lowest energies, hereby preparing the HF state
    n_excited = molecule.n_electrons
    for i in range(n_excited):
        if cq is None:
            prog += X(i)
        else:
            # the qubit ordering does not go from 0 to N-1, but has a custom labeling supplied by user via list cq
            prog += X(cq[i])

    return prog


def uccsd_ansatz_circuit(packed_amplitudes, n_orbitals, n_electrons, cq=None):
    # TODO apply logical re-ordering to Fermionic non-parametric way too!
    """
    This function returns a UCCSD variational ansatz with hard-coded gate angles. The number of orbitals specifies the
    number of qubits, the number of electrons specifies the initial HF reference state which is assumed was prepared.
    The packed_amplitudes input defines which gate angles to apply for each CC operation. The list cq is an optional
    list which specifies the qubit label ordering which is to be assumed.

    :param list() packed_amplitudes: amplitudes t_ij and t_ijkl of the T_1 and T_2 operators of the UCCSD ansatz
    :param int n_orbitals: number of *spatial* orbitals
    :param int n_electrons: number of electrons considered
    :param list() cq: list of qubit label order

    :return: circuit which prepares the UCCSD variational ansatz
    :rtype: Program
    """

    # Fermionic UCCSD
    uccsd_propagator = normal_ordered(uccsd_singlet_generator(packed_amplitudes, 2*n_orbitals, n_electrons))
    qubit_operator = jordan_wigner(uccsd_propagator)

    # convert the fermionic propagator to a Pauli spin basis via JW, then convert to a Pyquil compatible PauliSum
    pyquilpauli = qubitop_to_pyquilpauli(qubit_operator)

    # re-order logical stuff if necessary
    if cq is not None:
        pauli_list=[PauliTerm("I", 0, 0.0)]
        for term in pyquilpauli.terms:
            new_term = term
            # if the QPU has a custom lattice labeling, supplied by user through a list cq, reorder the Pauli labels.
            if cq is not None:
                new_term = term.coefficient
                for pauli in term:
                    new_index = cq[pauli[0]]
                    op = pauli[1]
                    new_term = new_term * PauliTerm(op=op, index=new_index)

            pauli_list.append(new_term)
        pyquilpauli = PauliSum(pauli_list)

    # create a pyquil program which performs the ansatz state preparation with angles unpacked from packed_amplitudes
    ansatz_prog = Program()

    # add each term as successive exponentials (1 single Trotter step, not taking into account commutation relations!)
    for commuting_set in commuting_sets(simplify_pauli_sum(pyquilpauli)):
        ansatz_prog += exponentiate_commuting_pauli_sum(-1j*PauliSum(commuting_set))(-1.0)

    return ansatz_prog


def uccsd_ansatz_circuit_parametric(n_orbitals, n_electrons, cq=None):
    """
    This function returns a UCCSD variational ansatz with hard-coded gate angles. The number of orbitals specifies the
    number of qubits, the number of electrons specifies the initial HF reference state which is assumed was prepared.
    The list cq is an optional list which specifies the qubit label ordering which is to be assumed.


    :param int n_orbitals: number of spatial orbitals in the molecule (for building UCCSD singlet generator)
    :param int n_electrons: number of electrons in the molecule
    :param list() cq: custom qubits
    
    :return: program which prepares the UCCSD :math:`T_1 + T_2` propagator with a spin-singlet assumption.
    :rtype: Program
    """
    # determine number of required parameters
    trotter_steps = 1
    m = (2*n_orbitals-n_electrons)*n_electrons/4
    n_parameters = int(trotter_steps * (m*(m+3))/2)

    prog = Program()
    memref = prog.declare('theta', memory_type='REAL', memory_size=n_parameters)

    fermionic_list, indices = uccsd_singlet_generator_with_indices(2 * n_orbitals, n_electrons)

    pauli_list = []
    index_dict = {}
    for i, fermionic_term in enumerate(fermionic_list):
        pauli_sum = qubitop_to_pyquilpauli(jordan_wigner(normal_ordered(fermionic_term)))
        for term in pauli_sum:
            new_term = term
            # if the QPU has a custom lattice labeling, supplied by user through a list cq, reorder the Pauli labels.
            if cq is not None:
                new_term = term.coefficient
                for pauli in term:
                    new_index = cq[pauli[0]]
                    op = pauli[1]
                    new_term = new_term * PauliTerm(op=op, index=new_index)

            pauli_list.append(new_term)
            # keep track of the indices in a dictionary
            index_dict[new_term.operations_as_set()] = indices[i]

    # add each term as successive exponentials (1 trotter step, and not taking into account commutation relations!)
    term_sets = commuting_sets(simplify_pauli_sum(PauliSum(pauli_list)))

    for j, terms in enumerate(term_sets):
        prog += exponentiate_commuting_pauli_sum_parametric(terms, index_dict, memref)

    return prog
