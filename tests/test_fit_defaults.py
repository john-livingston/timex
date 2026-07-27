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
