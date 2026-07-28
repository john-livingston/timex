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

    def fake_compute_ic(soln, max_loglike, nparams, ndata, method='BIC', verbose=True):
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
    monkeypatch.setattr(util, 'get_map_soln', lambda trace: ({}, -1.0))
    monkeypatch.setattr(util, 'get_max_loglike', lambda trace, model_fn=None: -1.0)
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: 1)

    tf.get_ic()
    assert captured['ndata'] == 17, 'expected 7 unmasked in g plus 10 in r'
