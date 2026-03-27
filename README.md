# timex

A Python package for transit fitting analysis.

## Installation

    git clone https://github.com/john-livingston/timex.git
    cd timex
    conda create -n timex python=3.13
    conda activate timex
    pip install -e .
    pip install git+https://github.com/john-livingston/limbdark

## Usage

After installation, you can use the command-line interface:

    timex examples/hip67522b
    timex examples/hip67522c

The working directory must contain both `fit.yaml` and `sys.yaml` files. 

## Dependencies

The package automatically installs the following dependencies:
- pyyaml
- pymc
- astropy
- patsy
- exoplanet
- exoplanet-core
- dill
- corner
- numpy
- pandas
- matplotlib
- arviz

Note: You still need to manually install `limbdark` from the GitHub repository as shown in the installation instructions above.
