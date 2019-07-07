import decorator
from pyquil.api import local_qvm

HAMILTONIAN = [
    ('Z', 0, -3.2),
    ('Z', 1, 2.3),
    ('X', 0, 1.4)
]

NSHOTS = 10000
NQUBITS = 2


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
