# Installation

## Requirements

- Python >= 3.8 (3.13 recommended)

## Install

```bash
git clone https://github.com/john-livingston/timex.git
cd timex
conda create -n timex python=3.13
conda activate timex
conda install eigen
pip install -e .
```

### celerite2

The GP support requires `celerite2` built from source (the PyPI release is outdated). The `eigen` headers (installed above) are needed for compilation:

```bash
pip install git+https://github.com/john-livingston/celerite2
```

### limbdark

The `limbdark` package must be installed manually:

```bash
pip install git+https://github.com/john-livingston/limbdark
```

## Dependencies

Installed automatically via pip:

- jax, jaxlib, numpyro
- jaxoplanet, celerite2
- astropy
- numpy, pandas, matplotlib
- arviz, corner
- pyyaml, patsy
