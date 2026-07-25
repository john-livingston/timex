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
