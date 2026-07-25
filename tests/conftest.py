import sys
import numpy as np
import pandas as pd
import pytest
from unittest import mock

# Mock celerite2.jax to work around JAX v0.6.0 compatibility issue
sys.modules['celerite2.jax'] = mock.MagicMock()
sys.modules['celerite2.jax.terms'] = mock.MagicMock()
sys.modules['celerite2.jax.ops'] = mock.MagicMock()


@pytest.fixture
def synthetic_lc(tmp_path):
    """Path to a plain 3 column light curve CSV (time, flux, fluxerr)."""
    rng = np.random.default_rng(42)
    n = 120
    t = np.linspace(2460000.0, 2460000.1, n)
    flux = 1.0 + rng.normal(0, 1e-3, n)
    fluxerr = np.full(n, 1e-3)
    fp = tmp_path / 'plain.csv'
    pd.DataFrame({'time': t, 'flux': flux, 'fluxerr': fluxerr}).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def synthetic_lc_aux(tmp_path):
    """Path to a light curve CSV with two auxiliary covariate columns."""
    rng = np.random.default_rng(7)
    n = 120
    t = np.linspace(2460000.0, 2460000.1, n)
    flux = 1.0 + rng.normal(0, 1e-3, n)
    fluxerr = np.full(n, 1e-3)
    fp = tmp_path / 'aux.csv'
    pd.DataFrame({
        'time': t,
        'flux': flux,
        'fluxerr': fluxerr,
        # deliberately not linear or quadratic in time: a covariate that lies
        # in the span of the polynomial trend basis would make the design
        # matrix rank deficient and break the rank assertions in test_io.py
        'airmass': 1.3 - 0.3 * np.cos(np.linspace(0.0, 2.5, n)) + rng.normal(0, 1e-3, n),
        'dx': rng.normal(0, 1, n),
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def map_soln():
    """A minimal MAP solution dict: one dataset named 'g', 100 points, 1 planet."""
    n = 100
    return {
        't0': np.array(0.05),
        'period': np.array([3.0]),
        'ror': np.array([0.05]),
        'b': np.array([0.3]),
        'dur': np.array([0.1]),
        'u_star_g': np.array([0.4, 0.2]),
        'g_mean': np.array(0.1),
        'g_log_sigma_lc': np.array(-1.0),
        'g_lm': np.full(n, 0.2),
        'g_light_curves': np.full((n, 1), -1.0),
        'g_light_curves_hr': np.full((500, 1), -1.0),
        'g_lc_pred': np.full(n, -1.0),
    }
