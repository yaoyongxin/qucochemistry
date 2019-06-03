================================
Qu & Co Quantum Code - Chemistry
================================

The `Qu & Co Chemistry <http://www.quandco.com>`__ package is an open source library (licensed under Apache 2) for compiling and running quantum chemistry algorithms on Rigetti's Forest quantum computing platform.

Installation
------------

To start using Qu & Co Chemistry, first install Rigetti's `Forest SDK
<https://www.rigetti.com/forest>`__, then install `pyQuil
<https://github.com/rigetti/pyquil>`__ and `rpcq
<https://github.com/rigetti/rpcq>`__
Then, to install the latest versions of Qu & Co Chemistry (in development mode):

.. code-block:: bash

  git clone https://github.com/qu-co/qucochemistry
  cd qucochemistry
  python -m pip install -e .

Alternatively, to install the latest PyPI releases as libraries (in user mode):

.. code-block:: bash

  python -m pip install --user qucochemistry

or in your Anaconda environment using

.. code-block:: bash

  conda install qucochemistry

Also be sure to take a look at the `ipython notebook demo <https://github.com/qu-co/qucochemistry/tree/master/examples/Tutorial_Single_molecule_end_to_end_VQE.ipynb>`__.
To be able run end-to-end programs, please consider installing PySCF and OpenFermion-PySCF using pip via

.. code-block:: bash

  python -m pip install --user openfermionpyscf pyscf

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

Disclaimer
----------
Copyright 2019
