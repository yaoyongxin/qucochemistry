import io

from setuptools import setup, find_packages

# This reads the __version__ variable
exec(open('qucochemistry/_version.py').read())

# Readme file as long_description:
long_description = ('=================\n' +
                    'Qu & Co Chemistry\n' +
                    '=================\n')
stream = io.open('README.rst', encoding='utf-8')
stream.readline()
long_description += stream.read()


# Read in requirements.txt
#requirements = open('requirements.txt').readlines()
#requirements = [r.strip() for r in requirements]


setup(
    name="qucochemistry",
    version=__version__,
    author="Vincent Elfving",
    author_email="quantumcode@quandco.com",
    description="A VQE package which interfaces with Rigetti's QCS platform",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=[
        # The minimum spec for a working qucochemistry install.
        # note to developers: this should be a subset of requirements.txt
        'numpy==1.15.4',
        'scipy==1.2.0',
        'openfermion==0.9.0',
        'pyquil==2.7.2'
    ],
    packages=find_packages(),
    license="Apache 2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
