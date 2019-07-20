from pyquil.paulis import PauliSum, PauliTerm
from openfermion.ops import FermionOperator, down_index, up_index, QubitOperator
from scipy.special import comb
import numpy as np
from scipy.optimize import minimize
import itertools

"""
The utility functions qubitop_to_pyquilpauli() and pyquilpauli_to_qubitop() are from forestopenfermion, removing the 
need to install that package as a dependency.

The utility function uccsd_singlet_generator_with_indices() is used to keep track of indices when preparing a parametric
 UCCSD preparation circuit

"""


def qubitop_to_pyquilpauli(qubit_operator):
    """
    Convert an OpenFermion QubitOperator to a PauliSum

    :param QubitOperator qubit_operator: OpenFermion QubitOperator to convert to a pyquil.PauliSum
    
    :return: PauliSum representing the qubit operator
    :rtype: PauliSum
    """
    if not isinstance(qubit_operator, QubitOperator):
        raise TypeError("qubit_operator must be a OpenFermion "
                        "QubitOperator object")

    transformed_term = PauliSum([PauliTerm("I", 0, 0.0)])
    for qubit_terms, coefficient in qubit_operator.terms.items():
        base_term = PauliTerm('I', 0)
        for tensor_term in qubit_terms:
            base_term *= PauliTerm(tensor_term[1], tensor_term[0])

        transformed_term += base_term * coefficient

    return transformed_term


def pyquilpauli_to_qubitop(pyquil_pauli):
    """
    Convert a pyQuil PauliSum to an OpenFermion QubitOperator
    
    :param pyquil_pauli: pyQuil PauliTerm or PauliSum to convert to an OpenFermion QubitOperator
    :type pyquil_pauli: [PauliTerm, PauliSum]
    
    :returns: a QubitOperator representing the PauliSum or PauliTerm 
    :rtype: QubitOperator
    """
    if not isinstance(pyquil_pauli, (PauliSum, PauliTerm)):
        raise TypeError("pyquil_pauli must be a pyquil PauliSum or "
                        "PauliTerm object")

    if isinstance(pyquil_pauli, PauliTerm):
        pyquil_pauli = PauliSum([pyquil_pauli])

    transformed_term = QubitOperator()
    # iterate through the PauliTerms of PauliSum
    for pauli_term in pyquil_pauli.terms:
        transformed_term += QubitOperator(
            term=tuple(zip(pauli_term._ops.keys(), pauli_term._ops.values())),
            coefficient=pauli_term.coefficient)

    return transformed_term


def uccsd_singlet_generator_with_indices(n_qubits, n_electrons):
    """Create a singlet UCCSD generator for a system with n_electrons, but return a `list of Fermion operators`, for
    each term in packed_amplitudes, name it a list called generator, instead of a `sum` called generator. It also
    returns a list of indices matching each term. This function generates a FermionOperator for a UCCSD generator
    designed to act on a single reference state consisting of n_qubits spin orbitals and n_electrons electrons,
    that is a spin singlet operator, meaning it conserves spin.

    :param n_qubits: Number of spin-orbitals used to represent the system, which also corresponds to number of qubits in a non-compact map.
    :type n_qubits: int
    :param n_electrons: Number of electrons in the physical system.
    :type n_electrons: int

    :return: Generator of the UCCSD operator that builds the UCCSD wavefunction.
    :rtype: list(FermionOperator)
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(np.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual

    # make a mock-packed_amplitudes of length (2 * n_single_amplitudes)
    packed_amplitudes = [1]*(2 * n_single_amplitudes + int(comb(n_single_amplitudes, 2)))

    # Single amplitudes
    t1 = packed_amplitudes[:n_single_amplitudes]
    # Double amplitudes associated with one spatial occupied-virtual pair
    t2_1 = packed_amplitudes[n_single_amplitudes:2 * n_single_amplitudes]
    # Double amplitudes associated with two spatial occupied-virtual pairs
    t2_2 = packed_amplitudes[2 * n_single_amplitudes:]

    # Initialize list of FermionOperator()'s
    generator = []
    # Initialize list of corresponding term indices inside packed_amplitudes
    generator_indices = []

    # Generate excitations
    spin_index_functions = [up_index, down_index]
    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(
            itertools.product(range(n_virtual), range(n_occupied))):

        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q

        for spin in range(2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            this_index = spin_index_functions[spin]
            other_index = spin_index_functions[1 - spin]

            # Get indices of spin orbitals
            virtual_this = this_index(virtual_spatial)
            virtual_other = other_index(virtual_spatial)
            occupied_this = this_index(occupied_spatial)
            occupied_other = other_index(occupied_spatial)

            # Generate single excitations
            coeff = t1[i]
            coeff_i = i
            generator_indices.append(coeff_i)
            generator += [FermionOperator((
                (virtual_this, 1),
                (occupied_this, 0)),
                coeff) + FermionOperator((
                    (occupied_this, 1),
                    (virtual_this, 0)),
                    -coeff)]

            # Generate double excitation
            coeff = t2_1[i]
            coeff_i = i + n_single_amplitudes
            generator_indices.append(coeff_i)
            generator += [FermionOperator((
                (virtual_this, 1),
                (occupied_this, 0),
                (virtual_other, 1),
                (occupied_other, 0)),
                coeff) + FermionOperator((
                    (occupied_other, 1),
                    (virtual_other, 0),
                    (occupied_this, 1),
                    (virtual_this, 0)),
                    -coeff)]

    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)),
                2)):

        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s

        # Generate double excitations
        coeff = t2_2[i]
        coeff_i = i + (2 * n_single_amplitudes)
        for (spin_a, spin_b) in itertools.product(range(2), repeat=2):
            # Get the functions which map a spatial orbital index to a
            # spin orbital index
            index_a = spin_index_functions[spin_a]
            index_b = spin_index_functions[spin_b]

            # Get indices of spin orbitals
            virtual_1_a = index_a(virtual_spatial_1)
            occupied_1_a = index_a(occupied_spatial_1)
            virtual_2_b = index_b(virtual_spatial_2)
            occupied_2_b = index_b(occupied_spatial_2)

            generator_indices.append(coeff_i)
            generator += [FermionOperator((
                (virtual_1_a, 1),
                (occupied_1_a, 0),
                (virtual_2_b, 1),
                (occupied_2_b, 0)),
                coeff) + FermionOperator((
                    (occupied_2_b, 1),
                    (virtual_2_b, 0),
                    (occupied_1_a, 1),
                    (virtual_1_a, 0)),
                    -coeff)]

    return generator, generator_indices


# class AnnotatedMinimizer:
#
#     niter = 0
#
#     @staticmethod
#     def callback(x):
#         print(f"{niter}: {x}")
#         niter += 1
#
#     def __call__(self, *args, **kwargs):
#         try:
#             res = minimize(*args, **kwargs, callback=minimizer_iter_callback)
#         except Exception
#             print("Error minimizing the variational ansatz!")
#             raise
#         self.niter = 0
#         return res


# TODO: replace the following code with the class above

niter = 1


def minimizer_iter_callback(x):
    global niter
    print(f"{niter}: {x}")
    niter += 1


def minimizer(fn, starting_angles, method, options={}):
    try:
        res = minimize(fn, starting_angles, method=method, options=options,
                       callback=minimizer_iter_callback)
    except Exception:
        # TODO: replace this print with a more robust logging system
        print("Error minimizing the variational ansatz!")
        raise
    global niter
    niter = 0
    return res
