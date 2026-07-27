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


def test_default_n_restarts_still_resolves():
    params = {'data': {}, 'planets': 'c'}
    for section in ('model', 'sampler'):
        for k, v in fit.defaults[section].items():
            params.setdefault(k, v)
    assert params['n_restarts'] == 1


def test_get_ic_counts_only_unmasked_points(monkeypatch):
    """Clipped outliers never entered the likelihood, so they must not inflate
    ndata in the information criteria."""
    import numpy as np
    from timex import fit, util

    captured = {}

    def fake_compute_ic(soln, max_logp, nparams, ndata, method='BIC', verbose=True):
        captured['ndata'] = ndata
        return 0.0

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.trace = None
    tf.data = {'g': dict(x=np.arange(10.0)), 'r': dict(x=np.arange(10.0))}
    mask = np.ones(10, dtype=bool)
    mask[:3] = False
    tf.masks = {'g': mask, 'r': None}
    tf.map_soln = {'t0': np.array(0.1)}

    monkeypatch.setattr(util, 'compute_ic', fake_compute_ic)
    monkeypatch.setattr(util, 'get_map_soln', lambda trace: ({}, -1.0))
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: 1)

    tf.get_ic()
    assert captured['ndata'] == 17, 'expected 7 unmasked in g plus 10 in r'
