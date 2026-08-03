"""Chromatic fits give each band its own radius ratio.

`chromatic: true` replaces the single `ror` site with one `ror_<band>` per
band, and the orbit is built from their mean. Nothing covered this, so a
chromatic fit could silently collapse back to a shared radius ratio, which is
the whole point of the option, and every existing test would still pass.
"""
import os

import numpy as np
import pytest
import yaml
from numpyro import handlers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

BANDS = ('g', 'i')


def _dataset(band, n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 0.2, n))
    return dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.5), X=None,
                texp=0.001, x_hr=x, band=band, ref_time=0.0)


def _traced_sites(chromatic, bands=BANDS):
    """Every sample site of a built model, without running a MAP."""
    from timex import model, util

    with open(os.path.join(EXAMPLE, 'sys.yaml')) as f:
        sys_params = yaml.safe_load(f)
    first = list(sys_params['planets'])[0]
    planets = [sys_params['planets'][first]]
    priors = util.get_priors('duration', sys_params['star'], planets, [],
                             list(bands), 2460423.03, 0.04)
    datasets = {b: _dataset(b) for b in bands}
    model_fn, _ = model.build(datasets, priors, 1,
                              masks={b: None for b in bands},
                              chromatic=chromatic, optimize=False, verbose=False)
    return handlers.trace(handlers.seed(model_fn, 0)).get_trace()


def test_chromatic_fits_one_radius_ratio_per_band():
    """Each band gets its own ror site and the shared one is gone.

    If the shared `ror` survived alongside the per band sites, or the per band
    sites were never created, the fit would report a chromatic result built
    from a single radius ratio.
    """
    sites = _traced_sites(chromatic=True)
    for band in BANDS:
        assert f'ror_{band}' in sites, f'no ror_{band} site in a chromatic model'
    assert 'ror' not in sites, (
        'the shared ror site is still sampled alongside the per band ones'
    )


def test_non_chromatic_fits_a_single_shared_radius_ratio():
    """The control. Without this, a model that always went per band would
    satisfy the test above and nothing would notice."""
    sites = _traced_sites(chromatic=False)
    assert 'ror' in sites
    for band in BANDS:
        assert f'ror_{band}' not in sites, (
            f'ror_{band} is sampled even though chromatic is off'
        )


def _var_names(chromatic):
    from timex import util
    data = {b: _dataset(b) for b in BANDS}
    return util.get_var_names(data, list(BANDS), 'duration', False, [],
                              chromatic=chromatic)


def test_chromatic_summary_reports_a_radius_ratio_per_band():
    """summary.csv is built from get_var_names, so the per band parameters have
    to be named there or the fit succeeds and reports nothing about them."""
    names = _var_names(chromatic=True)
    for band in BANDS:
        assert f'ror_{band}' in names, f'ror_{band} missing from the summary parameters'
    assert 'ror' not in names, 'the shared ror is still reported for a chromatic fit'


def test_non_chromatic_summary_reports_one_shared_radius_ratio():
    """Control for the test above."""
    names = _var_names(chromatic=False)
    assert 'ror' in names
    for band in BANDS:
        assert f'ror_{band}' not in names
