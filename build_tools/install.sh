#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to be in control of the numpy version
    # we are using for example through apt-get install
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install nose
}


create_new_conda_env() {
    # Skip Travis related code on circle ci.
    if [ -z $CIRCLECI ]; then
        # Deactivate the travis-provided virtual environment and setup a
        # conda-based environment instead
        deactivate
    fi

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh \
         -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    conda update --yes conda

    # Create a conda environment
    conda env create -f environment.yml
    source activate testenv
}

if [[ "$DISTRIB" == "neurodebian" ]]; then
    create_new_venv
    bash <(wget -q -O- http://neuro.debian.net/_files/neurodebian-travis.sh)
    sudo apt-get install -qq python-scipy python-nose python-nibabel python-sklearn

elif [[ "$DISTRIB" == "conda" ]]; then
    create_new_conda_env

else
    echo "Unrecognized distribution ($DISTRIB); cannot setup travis environment."
    exit 1
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

