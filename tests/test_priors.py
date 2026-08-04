"""Uniform prior bounds must survive the trip from fit.yaml into the model.

`get_priors` does not hand the model a lower and an upper bound. It encodes a
uniform prior as a midpoint in `priors[key]` and a full width in
`priors[f'{key}_unc']`, and `model.get_rv` decodes that back as
`midpoint -+ width/2`. Nothing checks that the encode and the decode agree, so
these tests read the bounds off the numpyro distribution the model actually
builds rather than recomputing the formula, which would only prove the formula
equals itself.
"""
import numpy as np
import pytest
import yaml
import os

from numpyro import handlers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')


def _star_and_planets(nplanets=1):
    with open(os.path.join(EXAMPLE, 'sys.yaml')) as f:
        sys_params = yaml.safe_load(f)
    first = list(sys_params['planets'])[0]
    planets = [sys_params['planets'][first] for _ in range(nplanets)]
    return sys_params['star'], planets


def _priors(uniform, nplanets=1, bands=('g',), fixed=()):
    from timex import util
    star, planets = _star_and_planets(nplanets)
    return util.get_priors('duration', star, planets, list(fixed), list(bands),
                           2460423.03, 0.04, uniform=uniform)


def _site_bounds(priors, key):
    """The low and high of the Uniform the model actually samples `key` from.

    Traced out of numpyro rather than recomputed, so this exercises get_rv's
    decode as well as get_priors' encode.
    """
    from timex import model

    def m():
        model.get_rv(key=key, priors=priors)

    trace = handlers.trace(handlers.seed(m, 0)).get_trace()
    fn = trace[key]['fn']
    assert type(fn).__name__ == 'Uniform', (
        f'{key} is a {type(fn).__name__}, so no uniform bounds were applied'
    )
    return np.atleast_1d(np.asarray(fn.low)), np.atleast_1d(np.asarray(fn.high))


def _dataset(band='g', n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 0.2, n))
    return dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.5), X=None,
                texp=0.001, x_hr=x, band=band, ref_time=0.0)


def _u_star_site_bounds(uniform, bands=('g', 'i')):
    """Build a real model and read the u_star bounds off its sampled site.

    u_star is built inside model_fn rather than through get_rv, so it needs a
    model. optimize=False skips the MAP, which is what keeps this fast.
    """
    from timex import model

    priors = _priors(uniform, bands=bands)
    datasets = {b: _dataset(band=b) for b in bands}
    model_fn, _ = model.build(datasets, priors, 1, masks={b: None for b in bands},
                              optimize=False, verbose=False)
    trace = handlers.trace(handlers.seed(model_fn, 0)).get_trace()
    out = {}
    for b in bands:
        site = trace[f'u_star_{b}']
        fn = site['fn']
        out[b] = (np.atleast_1d(np.asarray(fn.low)),
                  np.atleast_1d(np.asarray(fn.high)))
    return out


def test_uniform_u_star_bounds_reach_the_model():
    """A configured u_star range must be the range the model samples from.

    get_priors already encodes it: with u_star: [0.2, 0.8] the priors dict
    decodes to lo 0.2, hi 0.8. If the model builds its own bounds instead, the
    user's constraint is computed, stored and then silently discarded, and the
    fit explores limb darkening the config ruled out. Nothing raises.
    """
    bounds = _u_star_site_bounds({'u_star': [0.2, 0.8]})
    for band, (lo, hi) in bounds.items():
        assert lo == pytest.approx(0.2), f'{band}: lower bound is not the configured 0.2'
        assert hi == pytest.approx(0.8), f'{band}: upper bound is not the configured 0.8'


def test_uniform_u_star_site_keeps_both_coefficients():
    """The site has to stay shape (2,), one entry per quadratic coefficient.

    The uniform branch stores a scalar midpoint and width per band, not one per
    coefficient. Passing those through without broadcasting yields a scalar
    site, and the light curve code that reads u_star[0] and u_star[1] then dies
    with an IndexError deep inside jax rather than anywhere informative.
    """
    bounds = _u_star_site_bounds({'u_star': [0.2, 0.8]})
    for band, (lo, hi) in bounds.items():
        assert lo.shape == (2,), f'{band}: lower bound is {lo.shape}, not (2,)'
        assert hi.shape == (2,), f'{band}: upper bound is {hi.shape}, not (2,)'


def test_uniform_u_star_zero_to_one_is_unchanged():
    """The common case must behave exactly as it did when 0 to 1 was hardcoded,
    so honoring configured bounds is not a silent change for existing configs."""
    bounds = _u_star_site_bounds({'u_star': [0.0, 1.0]})
    for band, (lo, hi) in bounds.items():
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.0)


def test_scalar_uniform_bounds_reach_the_model():
    """The round trip that nothing checked: what the config asks for is what
    the model samples from.

    get_priors stores a midpoint and a full width; get_rv reconstructs
    midpoint -+ width/2. Storing a half width instead, or reconstructing with
    the wrong factor, shifts every bound without raising, and the fit quietly
    explores a different parameter space than the one requested.
    """
    priors = _priors({'ror': [0.01, 0.15]})
    lo, hi = _site_bounds(priors, 'ror')
    assert lo == pytest.approx(0.01)
    assert hi == pytest.approx(0.15)


def test_per_planet_uniform_bounds_stay_with_their_planet():
    """Planet indexed bounds must not be broadcast from the first planet.

    With [[0.03, 0.06], [0.01, 0.04]] a broadcast would give planet 1 planet
    0's range, which contains its true value, so the fit still converges and
    nothing looks wrong. Only the second planet's bound is evidence.
    """
    priors = _priors({'ror': [[0.03, 0.06], [0.01, 0.04]]}, nplanets=2)
    lo, hi = _site_bounds(priors, 'ror')
    assert lo == pytest.approx([0.03, 0.01])
    assert hi == pytest.approx([0.06, 0.04])


def test_per_planet_bounds_reject_a_planet_count_mismatch():
    """Two bound pairs for three planets is a config error, not something to
    broadcast or truncate: silently fitting a planet under another's prior is
    worse than refusing to start."""
    from timex import util
    star, planets = _star_and_planets(3)
    with pytest.raises(ValueError, match='must match number of planets'):
        util.get_priors('duration', star, planets, [], ['g'], 2460423.03, 0.04,
                        uniform={'ror': [[0.03, 0.06], [0.01, 0.04]]})


def test_uniform_initval_is_clipped_inside_the_bounds():
    """A stored initval must lie strictly inside the prior it belongs to.

    sys.yaml's ror is the natural initval and it sits outside the range below,
    so without the clip get_priors would hand out a value at zero prior density.

    Nothing in timex consumes *_initval today: numpyro takes its starting point
    from map_soln through init_to_value. This guards the invariant rather than a
    live code path, which is worth the line because the invariant is load
    bearing next door. timer's model does read initval, and an unclipped one put
    a nan in its initial point.
    """
    from timex import util
    star, planets = _star_and_planets(1)
    sys_ror = float(np.atleast_1d(np.asarray(planets[0]['ror'][0]))[0])
    lo_b, hi_b = 0.001, 0.5 * sys_ror
    assert not (lo_b < sys_ror < hi_b), 'fixture must put the sys.yaml value outside the bounds'

    priors = util.get_priors('duration', star, planets, [], ['g'],
                             2460423.03, 0.04, uniform={'ror': [[lo_b, hi_b]]})
    initval = np.atleast_1d(np.asarray(priors['ror_initval'], dtype=float))
    assert np.all(initval > lo_b) and np.all(initval < hi_b), (
        f'initval {initval} is not strictly inside [{lo_b}, {hi_b}]'
    )
