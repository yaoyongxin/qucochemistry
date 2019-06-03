from typing import List, Union

from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.pyqvm import PyQVM
from pyquil.quil import Program, percolate_declares
from pyquil.gates import I, RESET, MEASURE
from pyquil.paulis import PauliSum, PauliTerm, ID
from pyquil.operator_estimation import TomographyExperiment, group_experiments, ExperimentSetting, TensorProductState

from openfermion.hamiltonians import MolecularData
from openfermion.utils import uccsd_singlet_generator, normal_ordered, uccsd_singlet_get_packed_amplitudes,\
    expectation, jw_hartree_fock_state
from openfermion.transforms import jordan_wigner, get_sparse_operator, get_fermion_operator

from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

import numpy as np
import scipy as sp

import time

from qucochemistry.utils import qubitop_to_pyquilpauli, pyquilpauli_to_qubitop
from qucochemistry.circuits import augment_program_with_memory_values, pauli_meas, ref_state_preparation_circuit, \
    uccsd_ansatz_circuit, uccsd_ansatz_circuit_parametric


class VQEexperiment:

    def __init__(self, qc: Union[QuantumComputer, None] = None, hamiltonian: Union[PauliSum, List[PauliTerm], None] =
                 None, molecule: MolecularData = None, method: str = 'Numpy', strategy: str = 'UCCSD',
                 optimizer: str = 'BFGS', maxiter: int = 0, shotN: int = 10000, active_reset: bool = True,
                 tomography: bool = False, verbose: bool = False, parametric: bool = False, custom_qubits=None):
        """

        :param qc: QuantumComputer() object
        :param hamiltonian: PauliSum or list of PauliTerm
        :param molecule: MolecularData
        :param method: string describing the Backend solver method. current options: {Numpy, WFS, linalg, QC}
        :param strategy: string describing circuit VQE strategy. current options: {UCCSD, HF, custom_program}
        :param optimizer: string: classical optimization algorithm, choose from scipy.optimize.minimize options
        :param maxiter: int: max number of iterations
        :param shotN: int: number of shots in the Tomography experiments
        :param active_reset: boolean: whether or not to actively reset the qubits (see )
        :param tomography: boolean: set to False for access to full wavefunction, set to True for just sampling from it
        :param verbose: boolean: set to True for verbose output to the console, for all methods in this class
        :param parametric: boolean: set to True to use parametric gate compilation, False to compile a new circuit for
        every iteration
        :param custom_qubits: list: list of qubits, i.e. [7,0,1,2] ordering the qubit IDs as they appear on the QPU
        lattice of the QuantumComputer() object. In the future this may be automatically assigned.
        """

        if isinstance(hamiltonian, PauliSum):
            # Hamiltonian as a PauliSum, extracted to give a list instead
            self.pauli_list = hamiltonian.terms
            self.n_qubits = self.get_qubit_req()  # assumes 0-(N-1) ordering and every pauli index is in use
        elif isinstance(hamiltonian, List):
            if len(hamiltonian) > 0:
                if all([isinstance(term, PauliTerm) for term in hamiltonian]):
                    self.pauli_list = hamiltonian
                    self.n_qubits = self.get_qubit_req()
                else:
                    raise TypeError('Hamiltonian as a list must contain only PauliTerm objects')
            else:
                print('Warning, empty hamiltonian passed, assuming identity Hamiltonian = 1')
                self.pauli_list = [ID()]  # this is allowed in principle, but won't make a lot of sense to use.
        elif hamiltonian is None:
            if molecule is None:
                raise TypeError('either feed a MolecularData object or a PyQuil Hamiltonian to this class')
            else:
                self.H = normal_ordered(get_fermion_operator(molecule.get_molecular_hamiltonian()))  # store Fermionic
                # Hamiltonian in FermionOperator() instance
                self.qubitop = jordan_wigner(self.H)  # Apply jordan_wigner transformation and store
                self.n_qubits = 2 * molecule.n_orbitals
                self.pauli_list = qubitop_to_pyquilpauli(self.qubitop).terms
        else:
            raise TypeError('hamiltonian must be a PauliSum or list of PauliTerms')

        # abstract QC. can refer to a qvm or qpu. QC architecture and available gates decide the compilation of the
        # programs!
        if isinstance(qc, QuantumComputer):
            self.qc = qc
        elif qc is None:
            self.qc = None
        else:
            raise TypeError('qc must be a QuantumComputer object. If you do not use a QC backend, omit, or supply '
                            'qc=None')

        # number of shots in a tomography experiment
        if isinstance(shotN, int):
            self.shotN = shotN
        elif isinstance(shotN, float):
            self.shotN = int(shotN)
        else:
            raise TypeError('shotN must be an integer or float')

        # simulation method. Choose from
        methodoptions = ['WFS', 'linalg', 'QC', 'Numpy']
        if method in methodoptions:
            self.method = method
        else:
            raise ValueError('choose a method from the following list: ' + str(methodoptions) +
                             '. If a QPU, QVM or PyQVM is passed to qc, select QC.')

        # circuit strategy. choose from UCCSD, HF, custom_program
        strategyoptions = ['UCCSD', 'HF', 'custom_program']
        if strategy in strategyoptions:
            self.strategy = strategy
        else:
            raise ValueError('choose a circuit strategy from the following list: ' + str(strategyoptions))

        # classical optimizer
        classical_options = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B ', 'TNC', 'COBYLA',
                             'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
        if optimizer not in classical_options:
            raise ValueError('choose a classical optimizer from the following list: ' + str(classical_options))
        else:
            self.optimizer = optimizer

        # store the optimizer historical values
        self.history = []

        # chemistry files. must be properly formatted in order to use a UCCSD ansatz (see MolecularData)
        self.molecule = molecule

        # whether or not the qubits should be actively reset. False will make the hardware wait for 3 coherence lengths
        # to go back to |0>
        self.active_reset = active_reset

        # max number of iterations for the classical optimizer
        self.maxiter = maxiter

        # vqe results, stores output of scipy.optimize.minimize, a OptimizeResult object. initialize to None
        self.res = None

        # list of grouped experiments (only relevant to tomography)
        self.experiment_list = []

        # whether to print debugging data to console
        self.verbose = verbose

        # real QPU has a custom qubit labeling
        self.custom_qubits = custom_qubits

        # i'th function call
        self.it_num = 0

        # whether to perform parametric method
        self.parametric_way = parametric

        # whether to do tomography or just calculate the wavefunction
        self.tomography = tomography

        # initialize offset for pauli terms with identity. this serves to avoid doing simulation for the identity
        # operator, which always yields unity times the coefficient due to wavefunction normalization.
        self.offset = 0

        # set empty circuit unitary. This is used for the direct linear algebraic methods.
        self.circuit_unitary = None

        if strategy not in ['UCCSD', 'HF', 'custom_program']:
            raise ValueError('please select a strategy from UCCSD, HF, custom_program or modify this class with your '
                             'own options')

        if strategy == 'UCCSD':
            # load UCCSD initial amps from the CCSD amps in the MolecularData() object
            amps = uccsd_singlet_get_packed_amplitudes(self.molecule.ccsd_single_amps, self.molecule.ccsd_double_amps,
                                                       n_qubits=self.molecule.n_orbitals * 2,
                                                       n_electrons=self.molecule.n_electrons)
            self.initial_packed_amps = amps
        else:
            if strategy == 'HF' and not self.parametric_way:
                print('warning: HF strategy only supports parametric way; for now setting self.parametric_way = True')
                self.parametric_way = True
            # allocate empty initial angles for the circuit. modify later.
            self.initial_packed_amps = []

        if (strategy is 'UCCSD') and (method is not 'linalg'):  # UCCSD circuit strategy preparations
            self.ref_state = ref_state_preparation_circuit(molecule, ref_type='HF', cq=self.custom_qubits)

            if self.parametric_way:
                # in the parametric_way, the circuit is built with free parameters
                self.ansatz = uccsd_ansatz_circuit_parametric(self.molecule.n_orbitals, self.molecule.n_electrons,
                                                              cq=self.custom_qubits)
            else:
                if self.custom_qubits is not None:  # TODO also implement the UCCSD non-parametric way for custom cq
                    raise NotImplementedError('parametric circuits not yet implemented for UCCSD on custom lattice cq')
                # in the non-parametric_way, the circuit has hard-coded angles for the gates.
                self.ansatz = uccsd_ansatz_circuit(self.initial_packed_amps, self.molecule.n_orbitals,
                                                   self.molecule.n_electrons, cq=self.custom_qubits)
        elif strategy is 'HF':
            self.ref_state = ref_state_preparation_circuit(self.molecule, ref_type='HF', cq=self.custom_qubits)
            self.ansatz = Program()
        elif strategy is 'custom_program':
            self.ref_state = Program()
            self.ansatz = Program()

        # prepare tomography experiment if necessary
        if self.tomography:
            self.compile_tomo_expts()
        else:
            # avoid having to re-calculate the PauliSum object each time, store it.
            self.pauli_sum = PauliSum(self.pauli_list)

        # perform miscellaneous method-specific preparations
        if self.method == 'QC':
            if qc is None:
                raise ValueError('Method is QC, please supply a valid QuantumComputer() object to the qc variable.')
        elif self.method == 'WFS':
            pass
        elif self.method == 'Numpy':
            if self.parametric_way:
                raise ValueError('NumpyWavefunctionSimulator() backend does not yet support parametric programs.')
        elif self.method == 'linalg':
            if self.tomography:
                raise NotImplementedError('Tomography is not yet implemented for the linalg method.')

            if molecule is not None:
                # sparse initial state vector from the MolecularData() object
                self.initial_psi = jw_hartree_fock_state(self.molecule.n_electrons, 2*self.molecule.n_orbitals)
                # sparse operator from the MolecularData() object
                self.hamiltonian_matrix = get_sparse_operator(self.H, trunc=2)
            else:
                self.hamiltonian_matrix = get_sparse_operator(pyquilpauli_to_qubitop(PauliSum(self.pauli_list)))
                self.initial_psi = None
                print('Please supply VQE initial state with method VQEexperiment().set_initial_state()')
        else:
            raise ValueError('unknown method: please choose from method = {linalg, WFS, tomography} for direct linear '
                             'algebra, pyquil WavefunctionSimulator, or doing Tomography, respectively')

    def compile_tomo_expts(self):
        self.offset = 0
        # use Forest's sorting algo from the Tomography suite to group Pauli measurements together
        experiments = []
        for term in self.pauli_list:
            # if the Pauli term is an identity operator, add the term's coefficient directly to the VQE class' offset
            if len(term.operations_as_set()) == 0:
                self.offset += term.coefficient.real
            else:
                experiments.append(ExperimentSetting(TensorProductState(), term))

        suite = TomographyExperiment(experiments, program=Program())

        gsuite = group_experiments(suite)

        grouped_list = []
        for setting in gsuite:
            group = []
            for term in setting:
                group.append(term.out_operator)
            grouped_list.append(group)

        if self.verbose:
            print('Number of tomography experiments: ', len(grouped_list))

        self.experiment_list = []
        for group in grouped_list:

            self.experiment_list.append(GroupedPauliSetting(group, qc=self.qc, ref_state=self.ref_state,
                                                            ansatz=self.ansatz, shotN=self.shotN,
                                                            parametric_way=self.parametric_way, n_qubits=self.n_qubits,
                                                            method=self.method, verbose=self.verbose,
                                                            cq=self.custom_qubits))

    def objective_function(self, amps=None):
        """
        This function returns the Hamiltonian expectation value over the final circuit output state. If argument
        packed_amps is given, the circuit will run with those parameters. Otherwise, the initial angles will be used.
        :param amps: list or np.ndarray, list of circuit angles.
        :return: float: energy estimation
        """

        E = 0
        t = time.time()

        if amps is None:
            packed_amps = self.initial_packed_amps
        elif isinstance(amps, np.ndarray):
            packed_amps = amps.tolist()[:]
        elif isinstance(amps, list):
            packed_amps = amps[:]
        else:
            raise TypeError('Please supply the circuit parameters as a list or np.ndarray')

        if self.tomography:
            if not self.parametric_way:  # modify hard-coded type ansatz circuit based on packed_amps angles
                self.ansatz = uccsd_ansatz_circuit(packed_amps, self.molecule.n_orbitals,
                                                   self.molecule.n_electrons, cq=self.custom_qubits)
                self.compile_tomo_expts()
            for experiment in self.experiment_list:
                E += experiment.run_experiment(self.qc, packed_amps)  # Run tomography experiments
            E += self.offset  # add the offset energy to avoid doing superfluous tomography over the identity operator.
        elif self.method == 'WFS':
            # In the direct WFS method without tomography, direct access to wavefunction is allowed and expectation
            # value is exact each run.
            if self.parametric_way:
                E += WavefunctionSimulator().expectation(self.ref_state+self.ansatz,
                                                         self.pauli_sum,
                                                         {'theta': packed_amps}).real  # attach parametric angles here
            else:
                if packed_amps is not None:  # modify hard-coded type ansatz circuit based on packed_amps angles
                    self.ansatz = uccsd_ansatz_circuit(packed_amps, self.molecule.n_orbitals,
                                                       self.molecule.n_electrons, cq=self.custom_qubits)
                E += WavefunctionSimulator().expectation(self.ref_state+self.ansatz, self.pauli_sum).real
        elif self.method == 'Numpy':
            if self.parametric_way:
                raise ValueError('NumpyWavefunctionSimulator() backend does not yet support parametric programs.')
            else:
                if packed_amps is not None:
                    self.ansatz = uccsd_ansatz_circuit(packed_amps,
                                                       self.molecule.n_orbitals,
                                                       self.molecule.n_electrons,
                                                       cq=self.custom_qubits)
                E += NumpyWavefunctionSimulator(n_qubits=self.n_qubits).\
                    do_program(self.ref_state+self.ansatz).expectation(self.pauli_sum).real
        elif self.method == 'linalg':
            # check if molecule has data sufficient to construct UCCSD ansatz and propagate starting from HF state
            if self.molecule is not None:

                propagator = normal_ordered(uccsd_singlet_generator(packed_amps, 2 * self.molecule.n_orbitals,
                                                                    self.molecule.n_electrons,
                                                                    anti_hermitian=True))
                qubit_propagator_matrix = get_sparse_operator(propagator, trunc=2)
                uccsd_state = expm_multiply(qubit_propagator_matrix, self.initial_psi)
                expected_uccsd_energy = expectation(self.hamiltonian_matrix, uccsd_state).real
                E += expected_uccsd_energy
            else:   # apparently no molecule was supplied; attempt to just propagate the ansatz from user-specified
                # initial state, using a circuit unitary if supplied by the user, otherwise the initial state itself,
                # and then estimate over <H>
                if self.initial_psi is None:
                    raise ValueError('Warning: no initial wavefunction set. Please set using '
                                     'VQEexperiment().set_initial_state()')
                # attempt to propagate with a circuit unitary
                if self.circuit_unitary is None:
                    psi = self.initial_psi
                else:
                    psi = expm_multiply(self.circuit_unitary, self.initial_psi)
                E += expectation(self.hamiltonian_matrix, psi).real
        else:
            raise ValueError('Impossible method: please choose from method = {WFS, Numpy, linalg} if Tomography is set'
                             ' to False, or choose from method = {QC, WFS, Numpy, linalg} if tomography is set to True')

        if self.verbose:
            self.it_num += 1
            print('black-box function call #' + str(self.it_num))
            print('Energy estimate is now:  ' + str(E))
            print('at angles:               ', packed_amps)
            print('and this took ' + '{0:.3f}'.format(time.time()-t) + ' seconds to evaluate')

        self.history.append(E)

        return E

    def start_vqe(self, theta=None, maxiter: int = 0, options: dict = {}):
        """
        This method starts the VQE algorithm. User can supply an initial circuit setting, otherwise the stored initial
        settings are used. the maxiter refers to the scipy optimizer number of iterations. (which may well be much less
         than the number of function calls)

        :param theta: list or np.ndarray initial angles for the circuit to start the optimizer in.
        :param maxiter: int:  maximum number of iterations
        :param options: dictionary: options for the
        :return: scipy.optimize.minimize result object
        """
        t0 = time.time()
        if self.strategy == 'HF':
            raise ValueError('Warning: vqe object set to a static circuit, no variational algorithm possible. '
                             'Consider running the method objective_function() instead.')

        # allows user to initialize the VQE with custom circuit angles. len(angles) must be equal to the size of
        # memory register Theta in the parametric program
        if theta is None:
            if self.verbose:
                print('Setting starting circuit parameters to intial amps: ', self.initial_packed_amps)
            starting_angles = np.array(self.initial_packed_amps)
        elif isinstance(theta, np.ndarray):
            starting_angles = theta
        elif isinstance(theta, list):
            starting_angles = np.array(theta)
        else:
            raise TypeError('Please supply the circuit parameters as a list or np.ndarray')

        # store historical values of the optimizer
        self.history = []

        if maxiter > 0:
            self.maxiter = maxiter

        # define a base_options which can be extended with another dictionary supplied in the start_vqe() call.
        base_options = {'disp': self.verbose, 'maxiter': self.maxiter}

        self.it_num = 0
        # run the classical optimizer with the quantum circuit evaluation as an objective function.
        self.res = minimize(self.objective_function, starting_angles, method=self.optimizer,
                            options={**base_options, **options})

        if self.verbose:
            print('VQE optimization took ' + '{0:.3f}'.format(time.time()-t0) + ' seconds to evaluate')
        return self.res

    def get_exact_gs(self, hamiltonian=None):
        """
        Calculate the exact groundstate energy for the loaded Hamiltonian
        :return:
        """
        if hamiltonian is None:
            h = get_sparse_operator(pyquilpauli_to_qubitop(PauliSum(self.pauli_list)))
        else:
            if isinstance(hamiltonian, PauliSum):
                h = get_sparse_operator(pyquilpauli_to_qubitop(hamiltonian))
            else:
                raise TypeError('please give a PauliSum() object as the Hamiltonian.')

        if self.get_qubit_req() > 5:  # sparse matrix eigenvalue decomposition is less accurate, but is necessary for
            # larger matrices
            [w, _] = sp.sparse.linalg.eigsh(h, k=1)
        else:
            [w, _] = np.linalg.eigh(h.todense())

        Egs = min(w).real

        return Egs

    def get_qubit_req(self):
        """
        - assumes all Pauli term indices up to the largest one are in use!
        - assumes pauli_list has been loaded properly
        :return: int: number of qubits required in the circuit, as set by the Hamiltonian terms' indices.
        """

        if self.pauli_list is None:
            raise ValueError('Error: pauli_list not loaded properly. re-initialize VQE object')

        max_i = 0
        for term in self.pauli_list:
            for (index, st) in term.operations_as_set():
                if index > max_i:
                    max_i = index
        return max_i+1

    def get_results(self):
        """
        get results from the VQE experiment
        :return: scipy.optimize.minimize result object
        """
        if self.res is not None:
            return self.res
        else:
            raise Exception('No VQE results yet. Please run VQE using VQEexperiment.start_vqe()')

    def get_history(self):
        """
        Get historical values of objective_function from the optimizer
        :return:
        """
        return self.history

    def set_maxiter(self, maxiter: int):
        """
        set the maximum iteration number for the classical optimizer
        :param maxiter: maximum iteration number for the classical optimizer
        """
        self.maxiter = maxiter

    def verbose_output(self, verbose: bool = True):
        """
        :param verbose: boolean: set verbose output to console.
        """
        self.verbose = verbose

    def set_custom_ansatz(self, prog: Program = Program()):
        """
        :param prog: Program(): set a custom ansatz circuit as a program. All variational angles must be parametric !
        """
        if self.method == 'linalg':
            raise TypeError('method is linalg. Please set custom unitary instead of custom circuit.')
        self.ansatz = Program(prog)
        if self.tomography:
            self.compile_tomo_expts()

    def set_custom_ref_preparation(self, prog: Program = Program()):
        """
        :param prog: Program(): set a custom reference state preparation circuit as a program. All variational angles
        must be parametric !
        """
        if self.method == 'linalg':
            raise TypeError('method is linalg. Please set custom unitary instead of custom circuit.')
        self.ref_state = Program(prog)
        if self.tomography:
            self.compile_tomo_expts()

    def set_initial_angles(self, angles: List):
        """
        User-specify the initial angles for the experiment.
        :param angles: list: list of angles for the circuit
        """
        self.initial_packed_amps = angles

    def set_tomo_shots(self, shotN):
        """
        Set the number of shots for the tomography experiment. Warning: requires recompilation of the circuits.
        :param shotN: int: number of tomography repetitions.
        """
        if self.tomography:
            self.shotN = shotN
            self.compile_tomo_expts()
        else:
            print("WARNING: the VQE is not set to tomography mode, changing shot number won't affect anything!")

    def set_initial_state(self, psi):
        """
        Manually set the complex-valued initial state of the qubits
        :param psi: np.ndarray or scipy.sparse array: the complex-valued initial state of the qubits
        """
        self.initial_psi = psi

    def get_circuit(self):
        """
        :return: Program() : return the PyQuil program which defines the circuit. Excludes tomography rotations.
        """
        return Program(self.ref_state+self.ansatz)

    def set_circuit_unitary(self, unitary):
        """
        Sets the circuit unitary for use in the 'linalg' method.
        :param unitary: np.ndarray of size [2^N x 2^N]
        """
        if sp.sparse.issparse(unitary):
            self.circuit_unitary = unitary
        elif isinstance(unitary, np.ndarray):
            self.circuit_unitary = sp.sparse.csc_matrix(unitary)
        else:
            raise ValueError('Please supply either a sparse matrix or numpy ndarray')

    def save_program(self, filename):
        """this saves the preparation circuit as a quil program which can be parsed with pyquil.parser.parse
        :param filename: string: saves the quil program to this filename.
        """
        prog = self.ref_state + self.ansatz
        text = prog.out()
        with open(filename, "w") as text_file:
            text_file.write(text)


class GroupedPauliSetting:

    def __init__(self, list_gsuit_paulis: List[PauliTerm], qc: QuantumComputer, ref_state: Program, ansatz: Program,
                 shotN: int, parametric_way: bool, n_qubits: int, active_reset: bool = True, cq=None, method='QC',
                 verbose: bool = False):
        """
        A tomography experiment class for use in VQE. instantiate using a the following parameters:
        :param list_gsuit_paulis: list of Pauli terms which can be measured at the same time (they share a TPB!)
        :param qc: QuantumComputer() object which will simulate the terms
        :param ref_state: Program() object which produces the initial reference state (f.ex. Hartree-Fock)
        :param ansatz: Program() object which produces the ansatz (f.ex. UCCSD)
        :param shotN: number of shots to run this Setting for
        :param parametric_way: boolean, whether to use parametric gates or hard-coded gates
        :param n_qubits: int, total number of qubits used for the program
        :param active_reset: boolean, whether or not to actively reset the qubits
        :param cq: list of qubit labels instead of the default [0,1,2,3,...,N-1]
        :param method: string, describing the computational method from {QC, linalg, WFS, Numpy}
        :param verbose: boolean, whether or not to give verbose output during execution
        """
        self.parametric_way = parametric_way
        self.pauli_list = list_gsuit_paulis
        self.shotN = shotN
        self.method = method
        self.parity_matrix = self.construct_parity_matrix(list_gsuit_paulis, n_qubits)
        self.verbose = verbose
        self.n_qubits = n_qubits
        self.cq = cq

        if qc is not None:
            if qc.name[-4:] == 'yqvm' and self.cq is not None:
                raise NotImplementedError('manual qubit lattice feed not yet implemented. please set cq=None')

        # instantiate a new program and construct it for compilation
        prog = Program()

        if self.method == 'QC':
            ro = prog.declare('ro', memory_type='BIT', memory_size=self.n_qubits)

        if active_reset and self.method == 'QC':
            if not qc.name[-4:] == 'yqvm':  # in case of PyQVM, can not contain reset statement
                prog += RESET()

        # circuit which produces reference state (f.ex. Hartree-Fock)
        prog += ref_state

        # produce which prepares an ansatz state starting from a reference state (f.ex. UCCSD or swap network UCCSD)
        prog += ansatz

        self.coefficients = []
        already_done = []
        for pauli in list_gsuit_paulis:
            # let's store the pauli term coefficients for later use
            self.coefficients.append(pauli.coefficient)

            # also, we perform the necessary rotations going from X or Y to Z basis
            for (i, st) in pauli.operations_as_set():
                if st is not 'I' and i not in already_done:
                    # note that 'i' is the *logical* index corresponding to the pauli.
                    if cq is not None:  # if the logical qubit should be remapped to physical qubits, access this cq
                        prog += pauli_meas(cq[i], st)
                    else:
                        prog += pauli_meas(i, st)
                    # if we already have rotated the basis due to another term, don't do it again!
                    already_done.append(i)

        self.pure_pyquil_program = Program(prog)

        if self.method == 'QC':
            # measure the qubits and assign the result to classical register ro
            for i in range(self.n_qubits):
                if cq is not None:
                    prog += MEASURE(cq[i], ro[i])
                else:
                    prog += MEASURE(i, ro[i])

            prog2 = percolate_declares(prog)

            # wrap in shotN number of executions on the qc, to get operator measurement by sampling
            prog2.wrap_in_numshots_loop(shots=self.shotN)

            self.pyqvm_program = prog2

            # compile to native quil if it's not a PYQVM
            if not qc.name[-4:] == 'yqvm':
                nq_program = qc.compiler.quil_to_native_quil(prog2)
                # if self.verbose:  # debugging purposes
                #    print('')
                #    print(nq_program.native_quil_metadata)
                self.pyquil_executable = qc.compiler.native_quil_to_executable(nq_program)

    def run_experiment(self, qc: Union[QuantumComputer, None], angles=None):
        """
        method to run the Tomography experiment for this instance's setting, repeating for shotN shots.
        :param qc: QuantumComputer() object or None, depending on self.method
        :param angles: circuit parameters to feed
        :return: float: returns sum of all commuting pauli terms estimations for this experiment
        """

        if self.method == 'WFS':
            if self.cq is None:
                qubits = range(self.n_qubits)
            else:
                qubits = self.cq[:]

            if len(self.pure_pyquil_program) == 0:  # makes sure there is always a circuit to run
                self.pure_pyquil_program += I(qubits[0])

            if self.parametric_way:
                bitstrings = WavefunctionSimulator().run_and_measure(self.pure_pyquil_program, trials=self.shotN,
                                                                     qubits=qubits,
                                                                     memory_map={'theta': angles})
            else:
                bitstrings = WavefunctionSimulator().run_and_measure(self.pure_pyquil_program, qubits=qubits,
                                                                     trials=self.shotN)
        elif self.method == 'Numpy':
            prog = self.pure_pyquil_program
            if self.parametric_way:
                prog = augment_program_with_memory_values(self.pure_pyquil_program,
                                                          memory_map={'theta': angles})
            npwfs = PyQVM(n_qubits=self.n_qubits).wf_simulator.do_program(prog)
            bitstrings = npwfs.sample_bitstrings(n_samples=self.shotN)
        elif qc.name[-4:] == '-qvm':
            # run the pre-compiled QUIL executable and use parametric compilation at run-time for a performance boost
            if self.parametric_way:
                bitstrings = qc.run(self.pyquil_executable, memory_map={'theta': angles})
            else:  # when no angles are supplied it is assumed that the pyquil binary already has them: no memory_map
                bitstrings = qc.run(self.pyquil_executable)
        elif qc.name[-4:] == 'yqvm':
            # the QVM is actually a numpy wavefunction simulator, so let's run the pyquil (not-compiled) program instead
            if not self.parametric_way:
                bitstrings = qc.run(self.pyqvm_program)
            else:
                # TODO: parametric gates on PyQVM
                raise NotImplementedError('Parametric gates do not work properly on PyQVM yet')
                # first, we populate the parametric gates correctly:
                # prog = augment_program_with_memory_values(self.pyqvm_program, memory_map={'theta': angles})
                # bitstrings = qc.run(prog)
        else:  # assumes the only other possibility is an actual QPU
            # TODO: actually this should also work if a real QPU is supplied! enter additional wishes here
            # run the pre-compiled QUIL executable and use parametric compilation at run-time for a performance boost
            if self.parametric_way:
                bitstrings = qc.run(self.pyquil_executable, memory_map={'theta': angles})
            else:  # when no angles are supplied it is assumed that the pyquil binary already has them: no memory_map
                bitstrings = qc.run(self.pyquil_executable)

        #t = time.time() # dev only
        # start data processing
        # this matrix computes the pauli string parity, and stores that for each bitstring
        is_odd = np.mod(bitstrings.dot(self.parity_matrix), 2)

        # if the parity is odd, the bitstring gives a -1 eigenvalue, and +1 vice versa.
        # sum over all bitstrings, average over shotN shots, and weigh each pauli string by its coefficient
        E = (1 - 2 * np.sum(is_odd, axis=0) / self.shotN).dot(np.array(self.coefficients)).real
        # if self.verbose:  # dev only
        #    print('evaluating bitstrings took ' + str(time.time() - t) + ' seconds')
        # end data processing

        return E

    @staticmethod
    def construct_parity_matrix(pauli_list, n_qubits):
        """
        This method constructs a matrix which is used to evaluate PauliTerm expectation values from an array of
        bitstrings returned by qc.run(). See run_experiment() how this parity_matrix is used.
        :param pauli_list: list of PauliTerm() objects which should be measured.
        :param n_qubits: number of qubits assumed for the parity matrix
        :return: returns parity_matrix
        """

        def pauli_list_to_indices(trial_pauli_s):
            listi = []
            for term in trial_pauli_s:
                ll = []
                for (i, st) in term.operations_as_set():
                    if st is not 'I':
                        ll.append(i)
                listi.append(ll)
            return listi

        a = pauli_list_to_indices(pauli_list)

        parity_matrix = np.zeros([n_qubits, len(a)], dtype=int)

        for j, term in enumerate(pauli_list):
            for (i, st) in term.operations_as_set():
                if st is not 'I':
                    parity_matrix[i, j] = 1
        return parity_matrix
