================================
Qu & Co Quantum Code - Chemistry
================================

.. image:: https://readthedocs.org/projects/qucochemistry/badge/?version=latest
 :target: https://qucochemistry.readthedocs.io/en/latest/documentation.html
 :alt: Documentation Status
 
.. image:: https://badge.fury.io/py/qucochemistry.svg
 :target: https://badge.fury.io/py/qucochemistry
 
.. image:: https://anaconda.org/quco/qucochemistry/badges/version.svg   
 :target: https://anaconda.org/quco/qucochemistry

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/qu-co/qucochemistry/master?filepath=examples
 
.. image:: https://img.shields.io/badge/python-3.7-brightgreen.svg

.. image:: https://travis-ci.org/qu-co/qucochemistry.svg?branch=master 
 :target: https://travis-ci.org/qu-co/qucochemistry 


The `Qu & Co Chemistry <http://www.quandco.com>`__ package is an open source library (licensed under Apache 2) for compiling and running quantum chemistry algorithms on Rigetti's Forest quantum computing platform.

Installation
------------

To start using Qu & Co Chemistry library, you need to first install the Rigetti's `Forest SDK <https://www.rigetti.com/forest>`__ which contains both the Quantum Virtual Machine and the Rigetti's quantum compiler.

You can install the library in two different ways.

**From PyPi or conda**

Using pip install the latest version from PyPi within a virtual environment:

.. code-block:: bash

    python -m pip install qucochemistry

Alternatively, the library can be installed within a conda environment:

.. code-block:: bash

    conda install -c quco qucochemistry

**From source**

Using pip, install the library within a virtual environment:

.. code-block:: bash

    python -m pip install -r deploy/requirements.txt
    python -m pip install -e .

Alternatively, install within a Conda environment using the provided environment:

.. code-block:: bash

    conda env create -n <env_name> -f deploy/environment.yml
    conda activate <env_name>
    python -m pip install -e .


Usage
------------

In order to use this library within your program, Rigetti's quantum virtual machine and quantum compilers must be running in the background. 
If you run on Linux or OSX and the Rigetti's `Forest SDK <https://www.rigetti.com/forest>`__ is correctly installed, you can start them in the 
background with the following commands:

.. code-block:: bash

    screen -dm -S qvm qvm -S
    screen -dm -S quilc quilc -S

On Windows just execute :code:`qvm -S` and :code:`quilc -S` commands in two separate cmd terminals. 

For more details on how to use the library, several tutorials on Jupyter notebook are available `here <https://github.com/qu-co/qucochemistry/tree/master/examples/>`__. 
To be able run end-to-end programs, you should install PySCF and OpenFermion-PySCF as additional dependencies with pip:

.. code-block:: bash

  python -m pip install openfermionpyscf pyscf

If you created the Conda environment as described in the previous section, you should be able to install these dependencies within 
the environment with the same command.


**With Docker container**

The library can also be used in Jupyter notebooks hosted within a Docker container. You should have both `docker` and `docker-compose` installed in your system. 

To setup the Docker environment in the project root directory run:

.. code-block:: bash

  docker-compose up -d

Now you can access a Jupyter notebook in your browser at :code:`http://127.0.0.1:8888` with Qu&Co Chemistry library available. Navigate to the `examples/` folder to run the tutorial notebooks.



Development
-----------------

The unit tests are built using the `pytest` framework. In order to run them, install the qucochemistry package using the previous instruction
and add the following dependencies:

.. code-block:: bash

  # for Conda environment
  conda install pytest pytest-cov 
  # for standard virtual environment
  python -m pip install pytest pytest-cov 

The tests can be executed in the root project directory as follows:

.. code-block:: bash

  pytest -v --cov=qucochemistry

An automatic code coverage report will be generated after running the above command. In order to visualize 
the details of the code coverage for each module, an HTML report can be generated and rendered with your favorite
browser


.. code-block:: bash

  pytest -v --cov=qucochemistry --cov-report html
  firefox htmlcov/index.html


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
