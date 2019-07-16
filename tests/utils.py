import decorator
from pyquil import Program
from pyquil.api import local_qvm
from pyquil.gates import X, RY

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

NSHOTS = 10000
NQUBITS_H = 2
NQUBITS_H2 = 4


def start_qvm(fn):
    """
    This decorator ensures that in the the context where the decorated function is executed
    the following processes are running:
    >> qvm -S
    >> quilc -S

    They needed to compile and execute a program on the quantum virtual machine. When
    the context is terminated also the processes are.
    """
    def wrapper(fn, *args):
        with local_qvm():
            return fn(*args)
    return decorator.decorator(wrapper, fn)


def parametric_ansatz_program():
    prog = Program()
    theta = prog.declare('theta', memory_type='REAL', memory_size=1)
    prog.inst(RY(theta[0], 0))
    return prog


def static_ansatz_program():
    return Program(X(1))
