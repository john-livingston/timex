import numpy as np
import pandas as pd
import pytest

from .pipeline_fixtures import session_fit, use_fixed_t0, use_gp


@pytest.fixture(scope='session')
def default_fit(tmp_path_factory):
    """One short default (spline) run of the shipped example.

    Shared by the end-to-end checks, the non-GP IC rows, and the default
    posterior classification test. Those only read the result.
    """
    return session_fit(tmp_path_factory, 'default_fit', lambda p: None)


@pytest.fixture(scope='session')
def gp_fit(tmp_path_factory):
    """One short GP run of the shipped example.

    Shared by the GP pipeline, IC, EDF, and posterior classification tests.
    """
    return session_fit(tmp_path_factory, 'gp_fit', use_gp)


@pytest.fixture(scope='session')
def fixed_t0_fit(tmp_path_factory):
    """One short run with t0 held fixed."""
    return session_fit(tmp_path_factory, 'fixed_t0_fit', use_fixed_t0)


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
def gapped_lc_aux(tmp_path):
    """Two blocks of four points, separated by a gap wider than chunk_thresh.

    read_generic subtracts int(x.min()), so this reads back as
    x = [0, .01, .02, .03, .10, .11, .12, .13]. The 0.01 d spacing inside each
    block is below the 0.02 d default threshold and the 0.07 d gap is above
    it, so chunk_offset appends exactly two indicator columns.
    """
    t = 2460000.0 + np.array([0.0, .01, .02, .03, .10, .11, .12, .13])
    n = len(t)
    fp = tmp_path / 'gapped.csv'
    pd.DataFrame({
        'time': t,
        'flux': 1.0 + np.arange(n) * 1e-4,
        'fluxerr': np.full(n, 1e-3),
        'airmass': 1.2 + np.arange(n) * 0.01,
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def gapped_lc(tmp_path):
    """The same two blocks as gapped_lc_aux, with no auxiliary columns.

    A design matrix built from this holds only the blocks the config names
    plus the chunk offsets, so the column count exceeds what the config
    accounts for while there is no covariate at all.
    """
    t = 2460000.0 + np.array([0.0, .01, .02, .03, .10, .11, .12, .13])
    n = len(t)
    fp = tmp_path / 'gapped.csv'
    pd.DataFrame({
        'time': t,
        'flux': 1.0 + np.arange(n) * 1e-4,
        'fluxerr': np.full(n, 1e-3),
    }).to_csv(fp, index=False)
    return str(fp)


@pytest.fixture
def map_soln():
    """A minimal MAP solution dict: one dataset named 'g', 100 points, 1 planet.

    Shapes match what util.get_map_soln actually produces: entries ending in a
    DERIVED_SUFFIXES entry are squeezed, so a single planet's light curves are
    (n,) and not (n, 1). Free parameters keep their site shape.
    """
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
        'g_light_curves': np.full(n, -1.0),
        'g_light_curves_hr': np.full(500, -1.0),
        'g_lc_pred': np.full(n, -1.0),
    }


@pytest.fixture
def map_soln_multiplanet():
    """The same dataset fitted with two planets, so the light curves stay 2-D.

    Squeezing cannot flatten a genuine planet axis, so this is the shape that
    reaches the `lcs.ndim > 1` branches in util.get_residuals,
    util.get_outlier_mask, util.get_corrected and model._add_gp_predictions.
    The two planets have different depths, so summing over the planet axis is
    distinguishable from picking either column.
    """
    n = 100
    return {
        't0': np.array([0.05, 0.06]),
        'period': np.array([3.0, 7.0]),
        'ror': np.array([0.05, 0.03]),
        'b': np.array([0.3, 0.1]),
        'dur': np.array([0.1, 0.15]),
        'u_star_g': np.array([0.4, 0.2]),
        'g_mean': np.array(0.1),
        'g_log_sigma_lc': np.array(-1.0),
        'g_lm': np.full(n, 0.2),
        'g_light_curves': np.column_stack([np.full(n, -1.0), np.full(n, -0.25)]),
        'g_light_curves_hr': np.column_stack([np.full(500, -1.0), np.full(500, -0.25)]),
        'g_lc_pred': np.full(n, -1.25),
    }
