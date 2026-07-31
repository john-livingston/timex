import os
import shutil

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')


def _fit(tmp_path, name, **overrides):
    """Construct a TransitFit on a copy of the shipped example.

    __init__ runs set_priors, so this is all that is needed to inspect the
    limb darkening priors. clobber=True keeps the example's shipped out/
    from being adopted.
    """
    from timex import fit

    wd = tmp_path / name
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))
    with open(wd / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(wd / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True))
    fit_params.update(overrides)
    return fit.TransitFit(sys_params, fit_params, wd=str(wd))


def _u_star(tf):
    """The limb darkening priors as one flat array.

    tf.priors['u_star'] and ['u_star_unc'] are {band: [c1, c2]} dicts, not
    arrays. 'u_star_prior' is the string 'gaussian' and is deliberately not
    included. Both the coefficients and their uncertainties come out of the
    claret Monte Carlo, so both move with the seed.
    """
    out = []
    for key in ('u_star', 'u_star_unc'):
        per_band = tf.priors[key]
        for band in sorted(per_band):
            out.extend(np.asarray(per_band[band], dtype=float).ravel())
    assert out, 'expected u_star priors'
    return np.array(out)


def test_same_seed_gives_the_same_limb_darkening_priors(tmp_path):
    """claret marginalizes with numpy's global RNG, so without seeding the
    priors differ on every run and no fit is reproducible."""
    a = _u_star(_fit(tmp_path, 'a', random_seed=3))
    b = _u_star(_fit(tmp_path, 'b', random_seed=3))
    assert a == pytest.approx(b, rel=0, abs=0)


def test_different_seeds_give_different_limb_darkening_priors(tmp_path):
    """Not redundant with the test above. With the np.random.seed call removed
    but the get_state/set_state restore left in place, every call replays the
    same restored state, so both seeds return identical priors and only this
    control catches it."""
    a = _u_star(_fit(tmp_path, 'a', random_seed=3))
    b = _u_star(_fit(tmp_path, 'b', random_seed=4))
    assert not np.allclose(a, b)


def test_unseeded_is_still_unseeded(tmp_path):
    """The default must change nothing: with random_seed unset, claret stays
    unseeded and the limb darkening priors are redrawn per run. If this ever
    passes by returning equal priors, the None path has started seeding and the
    promise that the default is byte-identical to the old behavior is broken."""
    a = _u_star(_fit(tmp_path, 'a', random_seed=None))
    b = _u_star(_fit(tmp_path, 'b', random_seed=None))
    assert not np.allclose(a, b)


def test_building_a_fit_leaves_the_callers_numpy_state_untouched(tmp_path):
    """Seeding is a side effect on a process global. A caller drawing its own
    random numbers must not have its stream silently reset by constructing a
    fit."""
    np.random.seed(1234)
    expected = np.random.rand(3)

    np.random.seed(1234)
    _fit(tmp_path, 'a', random_seed=99)
    assert np.random.rand(3) == pytest.approx(expected, rel=0, abs=0)
