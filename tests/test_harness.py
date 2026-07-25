import numpy as np
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


def test_map_soln_fixture_shapes(map_soln):
    assert np.size(map_soln['g_mean']) == 1
    assert map_soln['g_lm'].shape == (100,)
    assert map_soln['g_light_curves'].shape == (100, 1)
