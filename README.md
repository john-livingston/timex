# timex

Transit timing estimator in Jax — a Python package for robust, efficient, and flexible Bayesian analysis of individual exoplanet transit events.

Documentation: [john-livingston.github.io/timex](https://john-livingston.github.io/timex)

## Installation

    git clone https://github.com/john-livingston/timex.git
    cd timex
    conda install eigen
    pip install -e .
    pip install git+https://github.com/john-livingston/celerite2
    pip install git+https://github.com/john-livingston/limbdark

## Usage

    timex examples/hip67522b

The working directory must contain both `fit.yaml` and `sys.yaml` files. See the [documentation](https://john-livingston.github.io/timex) for details.
