"""The GP drawn on fit.png must be the GP the fit actually used.

fit.png is the figure the systematics model is judged by, so a GP overlay
computed from a different noise diagonal than model.py's makes the trend look
smoother than the one subtracted in the published corrected light curve.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest


N = 60
N_HR = 200
X = np.linspace(0., .1, N)
X_HR = np.linspace(0., .1, N_HR)
YERR = np.full(N, .5)
MEAN = .3
LM = .8 * np.sin(2 * np.pi * X / .07)
TRA = np.where(np.abs(X - .05) < .02, -1.5, 0.)
TRA_HR = np.where(np.abs(X_HR - .05) < .02, -1.5, 0.)
# correlated structure for the GP to latch onto, on top of the transit and
# the linear model
GP_TRUTH = 1.6 * np.sin(2 * np.pi * X / .022)
Y = MEAN + LM + TRA + GP_TRUTH

LOG_SIGMA_LC = -1.     # jitter exp(-1) = 0.368 ppt, so exp(2*log_sigma) = 0.135
LOG_AMP = .3           # amplitude 10**0.3 = 2.0 ppt
LOG_SCALE = -1.7       # length scale 10**-1.7 = 0.020 d


def _trace():
    """Two chains of three identical draws, so every posterior median is
    exactly the value written here."""
    import arviz as az

    def draws(value):
        return np.broadcast_to(np.asarray(value), (2, 3) + np.shape(value)).copy()

    return az.from_dict(posterior={
        'g_mean': draws(MEAN),
        'g_log_sigma_lc': draws(LOG_SIGMA_LC),
        'g_lm': draws(LM),
        'g_light_curves': draws(TRA[:, None]),
        'g_light_curves_hr': draws(TRA_HR[:, None]),
        'gp_log_amp': draws(LOG_AMP),
        'gp_log_scale': draws(LOG_SCALE),
    })


def _systematics_ydata(fig):
    lines = [ln for ln in fig.axes[0].lines if ln.get_label() == 'systematics']
    assert len(lines) == 1, 'expected exactly one systematics curve'
    return lines[0].get_ydata()


def _gp_mean(diag):
    """The GP conditional mean celerite2 gives for this noise diagonal."""
    from celerite2 import GaussianProcess, terms

    gp = GaussianProcess(terms.Matern32Term(sigma=10**LOG_AMP, rho=10**LOG_SCALE))
    gp.compute(X, diag=diag)
    return gp.predict(Y - (TRA + LM + MEAN))


def test_gp_overlay_uses_the_models_jitter_convention():
    """model.py builds its noise diagonal as exp(2*log_sigma_lc) + yerr**2. The
    overlay works from lcjit = exp(log_sigma_lc), so exponentiating a second
    time inflates the jitter variance from 0.135 to 2.09 here, and the GP
    drawn on fit.png is then a heavily shrunk version of the one subtracted
    from the data.
    """
    import matplotlib.pyplot as plt
    from timex import plot

    data = dict(x=X, y=Y, yerr=YERR, x_hr=X_HR, ref_time=2460000.)
    soln = {'g_mean': np.array(MEAN), 'g_lm': LM}

    fig = plot.light_curve(data, 'g', soln, 1, trace=_trace(), use_gp=True,
                           median=True)
    try:
        drawn = _systematics_ydata(fig)
    finally:
        plt.close(fig)

    expected = MEAN + LM + _gp_mean(np.exp(2 * LOG_SIGMA_LC) + YERR**2)
    assert np.allclose(drawn, expected, atol=1e-8)

    # the fixture has to be able to tell the two diagonals apart, or the
    # assertion above would hold either way
    doubly_exponentiated = MEAN + LM + _gp_mean(
        np.exp(2 * np.exp(LOG_SIGMA_LC)) + YERR**2)
    assert not np.allclose(expected, doubly_exponentiated, atol=1e-2)


def test_combined_error_bars_carry_the_fitted_jitter():
    """The faint error bars on every panel are the weight the likelihood gave
    each point, sqrt(yerr**2 + exp(2*log_sigma_lc)). They are what a reader
    judges the scatter against, so they must use the same jitter convention as
    the GP overlay above.
    """
    import matplotlib.pyplot as plt
    from timex import plot

    data = dict(x=X, y=Y, yerr=YERR, x_hr=X_HR, ref_time=2460000.)
    soln = {'g_mean': np.array(MEAN), 'g_lm': LM}

    fig = plot.light_curve(data, 'g', soln, 1, trace=_trace(), use_gp=True,
                           median=True)
    try:
        # each panel draws the photometric bars first, then the combined ones
        segments = fig.axes[0].containers[1][2][0].get_segments()
    finally:
        plt.close(fig)

    half_lengths = np.array([seg[1, 1] - seg[0, 1] for seg in segments]) / 2
    # yerr 0.5 ppt and jitter exp(-1) ppt combine to sqrt(0.25 + 0.1353353)
    assert np.allclose(half_lengths, 0.6207538, atol=1e-7)
