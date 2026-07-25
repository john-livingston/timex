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
