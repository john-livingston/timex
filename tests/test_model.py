import numpy as np

from timex import model


def _gp_inputs(n=100):
    x = np.linspace(0.0, 0.1, n)
    datasets = {'g': dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.5))}
    masks = {'g': None}
    return datasets, masks


def test_add_gp_predictions_without_mean_site(map_soln):
    datasets, masks = _gp_inputs()
    soln = dict(map_soln)
    del soln['g_mean']            # include_mean=False leaves no mean site
    soln['gp_log_amp'] = np.array(-1.0)
    soln['gp_log_scale'] = np.array(-2.0)

    out = model._add_gp_predictions(soln, datasets, masks, gp_config=None)

    assert 'g_gp_pred' in out
    assert out['g_gp_pred'].shape == (100,)
    assert np.all(np.isfinite(out['g_gp_pred']))


def test_add_gp_predictions_with_mean_site(map_soln):
    datasets, masks = _gp_inputs()
    soln = dict(map_soln)
    soln['gp_log_amp'] = np.array(-1.0)
    soln['gp_log_scale'] = np.array(-2.0)

    out = model._add_gp_predictions(soln, datasets, masks, gp_config=None)

    assert out['g_gp_pred'].shape == (100,)
    assert np.all(np.isfinite(out['g_gp_pred']))


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
