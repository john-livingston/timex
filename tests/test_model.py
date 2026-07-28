import numpy as np
import pytest

from timex import model


def _gp_inputs(n=100):
    x = np.linspace(0.0, 0.1, n)
    datasets = {'g': dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.5))}
    masks = {'g': None}
    return datasets, masks


# the GP prediction is the conditional mean K (K + diag)^-1 r, and every input
# to it is fixed here so the answer can be built independently
AMP, RHO, JIT = 1.3, 0.018, 0.35
YERR, MEAN, LM, DEPTH = 0.2, 0.5, 0.25, -1.0


def _dense_conditional_mean(x, residuals):
    """K (K + diag)^-1 r with a dense Matern32 kernel, written out.

    Independent of celerite2's factorization: this is the textbook form of the
    same quantity, so agreement is a real cross check rather than a mirror.
    """
    d = x[:, None] - x[None, :]
    r = np.sqrt(3) * np.abs(d) / RHO
    K = AMP**2 * (1 + r) * np.exp(-r)
    diag = np.full(len(x), JIT**2 + YERR**2)
    return K @ np.linalg.solve(K + np.diag(diag), residuals)


@pytest.mark.parametrize('include_mean', [True, False])
def test_add_gp_predictions_matches_the_dense_conditional_mean(include_mean):
    """The value, not the shape. The GP prediction is subtracted from the data
    in every *-cor.csv and every residual report, so a prediction that is
    merely finite and correctly shaped is not evidence of anything: replacing
    gp.predict with zeros, or conditioning on the raw flux instead of on the
    residuals, both survive a shape assertion.

    Hand derived: the deterministic model is mean + light curve + lm, so the
    residuals are y - (0.5 - 1.0 + 0.25) = y + 0.25 with a mean site and
    y + 0.75 without one, and the conditional mean of those follows from the
    Matern32 kernel directly.
    """
    n = 12
    x = np.linspace(0.0, 0.11, n)
    y = np.arange(n, dtype=float)
    datasets = {'g': dict(x=x, y=y, yerr=np.full(n, YERR))}
    soln = {
        'g_lm': np.full(n, LM),
        'g_light_curves': np.full(n, DEPTH),
        'gp_log_amp': np.array(np.log10(AMP)),
        'gp_log_scale': np.array(np.log10(RHO)),
        'g_log_sigma_lc': np.array(np.log(JIT)),
    }
    if include_mean:
        soln['g_mean'] = np.array(MEAN)

    out = model._add_gp_predictions(soln, datasets, {'g': None}, gp_config=None)

    model_flux = (MEAN if include_mean else 0.0) + DEPTH + LM
    expected = _dense_conditional_mean(x, y - model_flux)
    assert out['g_gp_pred'] == pytest.approx(expected, abs=1e-8)


def test_add_gp_predictions_missing_mean_matches_explicit_zero(map_soln):
    # model_fn uses mean = 0.0 when include_mean is False, so dropping the
    # site must give the same residuals as setting it to zero explicitly
    datasets, masks = _gp_inputs()

    without = dict(map_soln)
    del without['g_mean']
    without['gp_log_amp'] = np.array(-1.0)
    without['gp_log_scale'] = np.array(-2.0)

    explicit = dict(map_soln)
    explicit['g_mean'] = np.array(0.0)
    explicit['gp_log_amp'] = np.array(-1.0)
    explicit['gp_log_scale'] = np.array(-2.0)

    a = model._add_gp_predictions(without, datasets, masks, gp_config=None)
    b = model._add_gp_predictions(explicit, datasets, masks, gp_config=None)

    assert np.allclose(a['g_gp_pred'], b['g_gp_pred'])
    # and the fixture's nonzero mean must give a different answer, so the
    # comparison above is not vacuous
    nonzero = dict(map_soln)
    nonzero['gp_log_amp'] = np.array(-1.0)
    nonzero['gp_log_scale'] = np.array(-2.0)
    c = model._add_gp_predictions(nonzero, datasets, masks, gp_config=None)
    assert not np.allclose(a['g_gp_pred'], c['g_gp_pred'])


def test_add_gp_predictions_sums_over_the_planet_axis(map_soln_multiplanet):
    """The GP is conditioned on residuals, so an unsubtracted second transit
    would be absorbed into the GP prediction instead of the transit model."""
    datasets, masks = _gp_inputs()
    soln = dict(map_soln_multiplanet)
    soln['gp_log_amp'] = np.array(-1.0)
    soln['gp_log_scale'] = np.array(-2.0)

    presummed = dict(soln)
    presummed['g_light_curves'] = soln['g_light_curves'].sum(axis=-1)
    first_only = dict(soln)
    first_only['g_light_curves'] = soln['g_light_curves'][:, 0]

    out = model._add_gp_predictions(dict(soln), datasets, masks, gp_config=None)
    reference = model._add_gp_predictions(presummed, datasets, masks, gp_config=None)
    wrong = model._add_gp_predictions(first_only, datasets, masks, gp_config=None)

    assert out['g_gp_pred'].shape == (100,)
    assert np.allclose(out['g_gp_pred'], reference['g_gp_pred'])
    # and the comparison is not vacuous: dropping a planet does change it
    assert not np.allclose(out['g_gp_pred'], wrong['g_gp_pred'])


def test_as_init_arrays_converts_python_scalars():
    """A map.pkl pickled before get_map_soln was changed to preserve free
    parameter shapes can still hold plain Python floats for scalar sites.
    numpyro's init_to_value substitutes them verbatim, and get_rv then calls
    .squeeze() on the result. Anything handed to init_to_value must therefore
    be an array.
    """
    from timex import model

    out = model._as_init_arrays({
        't0': 0.05,                       # plain float, as a legacy map.pkl might hold
        'ror': np.array([0.1]),           # already an array
        'u_star_g': np.array([0.4, 0.2]),
    })
    for key, value in out.items():
        assert hasattr(value, 'squeeze'), f'{key} is not array-like: {type(value)}'
    assert out['t0'].squeeze().shape == ()
    assert out['u_star_g'].shape == (2,)


def test_as_init_arrays_leaves_values_unchanged_numerically():
    from timex import model

    out = model._as_init_arrays({'t0': 0.05, 'ror': np.array([0.1])})
    assert float(out['t0']) == 0.05
    assert float(out['ror'][0]) == 0.1


def test_build_gp_uses_log10_convention_and_jitter_diagonal():
    """Amplitude and scale are stored as log10, and the diagonal is
    exp(2*log_sigma_lc) + yerr**2. Both are easy to get wrong, so this pins
    the values rather than just checking the call succeeds."""
    from celerite2 import GaussianProcess, terms
    from timex import model

    n = 20
    x = np.linspace(0.0, 0.2, n)
    yerr = np.full(n, 0.3)
    soln = {
        'gp_log_amp': np.array(1.0),      # 10**1.0 = 10.0, but exp(1.0) = 2.718
        'gp_log_scale': np.array(-1.5),   # 10**-1.5 = 0.0316, but exp(-1.5) = 0.223
        'g_log_sigma_lc': np.array(np.log(0.4)),
    }
    gp, diag = model._build_gp(soln, 'g', x, yerr, gp_config=None)

    assert np.allclose(diag, 0.4**2 + 0.3**2)

    v = np.ones(n)
    right = GaussianProcess(terms.Matern32Term(sigma=10.0, rho=10**-1.5))
    right.compute(x, diag=diag)
    assert np.allclose(gp.apply_inverse(v), right.apply_inverse(v))

    # and it must NOT match the natural log reading of the same numbers
    wrong = GaussianProcess(terms.Matern32Term(sigma=np.exp(1.0), rho=np.exp(-1.5)))
    wrong.compute(x, diag=diag)
    assert not np.allclose(gp.apply_inverse(v), wrong.apply_inverse(v))


def test_build_gp_honours_per_dataset_hyperparameters():
    from timex import model

    n = 10
    x = np.linspace(0.0, 0.1, n)
    yerr = np.full(n, 0.1)
    soln = {
        'gp_log_amp_g': np.array(0.0),
        'gp_log_amp_r': np.array(1.0),
        'gp_log_scale': np.array(-2.0),
        'g_log_sigma_lc': np.array(-1.0),
        'r_log_sigma_lc': np.array(-1.0),
    }
    cfg = {'per_dataset': ['log_amp']}
    gp_g, _ = model._build_gp(soln, 'g', x, yerr, cfg)
    gp_r, _ = model._build_gp(soln, 'r', x, yerr, cfg)
    # r has 10x the amplitude, so its kernel is not the same object's twin
    assert gp_g.apply_inverse(np.ones(n))[0] != gp_r.apply_inverse(np.ones(n))[0]


def test_build_gp_honours_per_dataset_log_scale():
    from timex import model

    n = 10
    x = np.linspace(0.0, 0.1, n)
    yerr = np.full(n, 0.1)
    soln = {
        'gp_log_amp': np.array(0.0),
        'gp_log_scale_g': np.array(-2.0),
        'gp_log_scale_r': np.array(-0.5),
        'g_log_sigma_lc': np.array(-1.0),
        'r_log_sigma_lc': np.array(-1.0),
    }
    cfg = {'per_dataset': ['log_scale']}
    gp_g, _ = model._build_gp(soln, 'g', x, yerr, cfg)
    gp_r, _ = model._build_gp(soln, 'r', x, yerr, cfg)
    # r has a much longer scale, so its kernel is not the same object's twin
    assert gp_g.apply_inverse(np.ones(n))[0] != gp_r.apply_inverse(np.ones(n))[0]
