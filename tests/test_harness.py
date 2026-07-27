import pandas as pd

import timex


def test_package_imports():
    assert hasattr(timex, '__version__') or timex.__file__.endswith('__init__.py')


def test_synthetic_lc_fixture_is_three_columns(synthetic_lc):
    df = pd.read_csv(synthetic_lc)
    assert list(df.columns) == ['time', 'flux', 'fluxerr']
    assert len(df) == 120


def test_synthetic_lc_aux_fixture_has_covariates(synthetic_lc_aux):
    df = pd.read_csv(synthetic_lc_aux)
    assert list(df.columns) == ['time', 'flux', 'fluxerr', 'airmass', 'dx']
