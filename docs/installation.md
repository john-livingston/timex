# Installation

## Requirements

- Python >= 3.8 (3.13 recommended)

## Install

```bash
git clone https://github.com/john-livingston/timex.git
cd timex
conda create -n timex python=3.13
conda activate timex
pip install -e .
```

### limbdark

The `limbdark` package must be installed manually:

```bash
pip install git+https://github.com/john-livingston/limbdark
```

## Dependencies

Installed automatically via pip:

- pymc
- exoplanet, exoplanet-core
- astropy
- numpy, pandas, matplotlib
- arviz, corner
- pyyaml, dill, patsy
