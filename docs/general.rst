.. _general:

================================
General Information
================================

The `Qu & Co Chemistry <http://www.quandco.com>`__ package is an open source library (licensed under Apache 2) for compiling and running quantum chemistry algorithms on Rigetti's Forest quantum computing platform.

Installation
------------

To start using Qu & Co Chemistry library, you need to first install the Rigetti's `Forest SDK <https://www.rigetti.com/forest>`__ which contains both the Quantum Virtual Machine and the Rigetti's quantum compiler.

You can install the library in two different ways.

**From PyPi**

Using pip install the latest version from PyPi in user mode:

.. code-block:: bash

    python -m pip install --user qucochemistry

Alternatively, the library can be install within a Conda virtual environment:

.. code-block:: bash

    conda env create -n <env_name>
    conda activate <env_name>
    conda install pip
    pip install qucochemistry

**From source**

Using pip, install the library in user mode:

.. code-block:: bash

    python -m pip install --user -r deploy/requirements.txt
    python -m pip install --user -e .

Alternatively, install within a Conda environment using the provided environment:

.. code-block:: bash

    conda env -n <env_name> -f deploy/conda_env.yml
    conda activate <env_name>
    pip install -e .

Usage
------------

In order to use this library within your program, Rigetti's quantum virtual machine and quantum compilers must be running in the background. Provided that the Rigetti's `Forest SDK <https://www.rigetti.com/forest>`__ is correctly installed, you can do so with the following commands:

.. code-block:: bash

    screen -dm -S qvm qvm -S
    screen -dm -S quilc quilc -S

For more details on how to use the library, several tutorials on Jupyter notebook are available `here <https://github.com/qu-co/qucochemistry/tree/master/examples/Tutorial_Single_molecule_end_to_end_VQE.ipynb>`__.
To be able run end-to-end programs, you should install PySCF and OpenFermion-PySCF as additional dependencies with pip:

.. code-block:: bash

  python -m pip install --user openfermionpyscf pyscf

If you created the Conda environment as described in the previous section, you should be able to install these dependencies within the environment with the same command (without the :code:`--user` flag).

How to contribute
-----------------

We'd love to accept your contributions and patches to Qu & Co Chemistry.
There are a few guidelines you need to follow.
Contributions to Qu & Co Chemistry must be accompanied by a Contributor License Agreement.
You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as part of the project.

All submissions, including submissions by project members, require review.
We use GitHub pull requests for this purpose. Consult
`GitHub Help <https://help.github.com/articles/about-pull-requests/>`__ for
more information on using pull requests.
Furthermore, please make sure your new code comes with extensive tests!
We use automatic testing to make sure all pull requests pass tests and do not
decrease overall test coverage by too much. Make sure you adhere to our style
guide. Just have a look at our code for clues. We mostly follow
`PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__ and use
the corresponding `linter <https://pypi.python.org/pypi/pep8>`__ to check for it.
Code should always come with documentation.

Authors
----------

`Vincent Elfving <https://github.com/vincentelfving>`__ (Qu & Co B.V.)

We are happy to include future contributors as authors on later Qu & Co Chemistry releases.