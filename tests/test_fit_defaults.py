from timex import fit


def test_n_restarts_is_a_model_setting():
    """It changes the MAP solution, so it is not a sampler setting."""
    assert 'n_restarts' in fit.defaults['model']
    assert 'n_restarts' not in fit.defaults['sampler']


def test_sampler_defaults_are_only_run_tier_and_no_effect_keys():
    """Guards the tier split: anything else added to sampler would be
    silently classified as model tier by cache.compute_keys."""
    from timex import cache
    assert set(fit.defaults['sampler']) == set(cache.RUN_TIER) | set(cache.NO_EFFECT)


def _validated(fit_params):
    """Run the real validate() over fit_params and return it, merged in place.

    A full TransitFit needs data files, priors and an output directory, none
    of which validate() touches; same construction shortcut as
    test_get_ic_counts_only_unmasked_points below.
    """
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.fit_params = fit_params
    tf.validate()
    return tf.fit_params


def test_omitted_n_restarts_resolves_to_the_default():
    assert _validated({'data': {}, 'planets': 'c'})['n_restarts'] == 1


def test_explicit_n_restarts_survives_the_default_merge():
    """validate() merges defaults for keys the config omits. Without its
    'k not in fit_params' guard it overwrites every key instead, so a user
    asking for 5 restarts would silently get 1 and a worse MAP."""
    merged = _validated({'data': {}, 'planets': 'c', 'n_restarts': 5})
    assert merged['n_restarts'] == 5


def test_random_seed_is_a_model_setting():
    """It seeds the limb darkening Monte Carlo, so it changes the priors and
    the MAP, not only the chain. Same argument as n_restarts."""
    assert 'random_seed' in fit.defaults['model']
    assert 'random_seed' not in fit.defaults['sampler']


def test_random_seed_defaults_to_none():
    """None must mean today's behavior: fixed sampler key, unseeded claret."""
    assert _validated({'data': {}, 'planets': 'c'})['random_seed'] is None


def test_explicit_random_seed_survives_the_default_merge():
    assert _validated({'data': {}, 'planets': 'c', 'random_seed': 11})['random_seed'] == 11


def test_explicit_sampler_settings_survive_the_default_merge():
    """The same guard covers the sampler section, where being overwritten
    would silently replace a long production run with the 2000 draw default."""
    merged = _validated({'data': {}, 'planets': 'c', 'draws': 37, 'chains': 3})
    assert merged['draws'] == 37
    assert merged['chains'] == 3
    assert merged['tune'] == fit.defaults['sampler']['tune']


def test_get_ic_counts_only_unmasked_points(monkeypatch):
    """Clipped outliers never entered the likelihood, so they must not inflate
    ndata in the information criteria."""
    import numpy as np
    from timex import fit, util

    captured = {}

    def fake_compute_ic(max_loglike, nparams, ndata, method='BIC', verbose=True):
        captured['ndata'] = ndata
        return 0.0

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.trace = None
    tf.model_fn = None
    tf.data = {'g': dict(x=np.arange(10.0)), 'r': dict(x=np.arange(10.0))}
    mask = np.ones(10, dtype=bool)
    mask[:3] = False
    tf.masks = {'g': mask, 'r': None}
    tf.map_soln = {'t0': np.array(0.1)}

    monkeypatch.setattr(util, 'compute_ic', fake_compute_ic)
    monkeypatch.setattr(util, 'get_max_loglike',
                        lambda trace, model_fn=None: (-1.0, (0, 0)))
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: 1)

    tf.get_ic()
    assert captured['ndata'] == 17, 'expected 7 unmasked in g plus 10 in r'


def test_save_corrected_writes_jitter_inflated_errors_in_relative_flux(tmp_path, map_soln):
    """The corrected light curve is published in relative flux, while the fit
    and its jitter live in ppt. Adding the jitter after the 1e-3 conversion,
    or leaving it out, both give an error column that no longer matches the
    weights the fit used.
    """
    import numpy as np
    import pandas as pd
    from timex import fit

    n = 100
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.wd = str(tmp_path / 'target')
    tf.ref_time = 2460000.
    tf.nplanets = 1
    tf.map_soln = map_soln
    tf.data = {'g': dict(x=np.linspace(0., .1, n), y=np.zeros(n),
                         yerr=np.full(n, .5), x_hr=np.linspace(0., .1, 500))}
    tf.masks = {'g': np.ones(n, dtype=bool)}

    tf.save_corrected()

    df = pd.read_csv(tmp_path / 'target-g-cor.csv')
    # 0.5 ppt photometric error and a jitter of exp(-1) ppt combine to
    # sqrt(0.25 + 0.1353353) = 0.6207538 ppt, or 6.207538e-4 in relative flux
    assert np.allclose(df['yerr'].values, 6.207538e-4, atol=1e-10)
