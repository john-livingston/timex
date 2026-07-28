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


def _fit_with_chunk_offsets(fp):
    """A fit stub over a real read_generic design matrix: one covariate, one
    trend column and two chunk offsets, in the order read_generic appends them.

    Deriving the block sizes from the config alone gives ncovariates = 4 - 1 =
    3, because the chunk columns are invisible to it. The covariate panel then
    shows the trend and the first chunk offset, and the trend panel shows the
    second chunk offset.
    """
    from timex import io

    x, _, _, X, _, _, _, layout = io.read_generic(
        fp, binsize=None, trend=1, chunk_offset=True, chunk_thresh=0.02,
        verbose=False)

    class _Fit:
        use_gp = False
        masks = {'g': None}
        map_soln = {'g_weights': np.array([2.0, 3.0, 4.0, 5.0])}
        data = {'g': dict(x=x, X=X, ncols=layout, band='g')}
        fit_params = {'data': {'g': dict(trend=1, spline=False, spline_knots=5,
                                         add_bias=False, chunk_offset=True,
                                         chunk_thresh=0.02)}}

    return _Fit()


def _panel(fig, title):
    return next(ax for ax in fig.axes if ax.get_title() == title)


# x = [0, .01, .02, .03, .10, .11, .12, .13] centered on 0.5*(0 + .13) = .065,
# which is the single column np.vander(., 2)[:, :-1] keeps for trend=1
TREND_COLUMN = np.array([-.065, -.055, -.045, -.035, .035, .045, .055, .065])
# what the old subtraction hands the trend panel instead: the second chunk
CHUNK1_COLUMN = np.array([0., 0., 0., 0., 1., 1., 1., 1.])


def test_systematics_trend_panel_shows_the_trend_not_a_chunk_offset(gapped_lc_aux):
    import matplotlib.pyplot as plt
    from timex import plot

    fig = plot.systematics(_fit_with_chunk_offsets(gapped_lc_aux), 'g', style=2)
    try:
        drawn = _panel(fig, 'trend').lines
        assert len(drawn) == 1
        ydata = drawn[0].get_ydata()
    finally:
        plt.close(fig)

    assert ydata == pytest.approx(TREND_COLUMN, abs=1e-9)
    # the fixture has to tell the two apart, or the assertion above would hold
    # whichever column got sliced
    assert not np.allclose(TREND_COLUMN, CHUNK1_COLUMN)


def test_systematics_covariate_panel_shows_only_the_covariate(gapped_lc_aux):
    """The mirror of the above at the other boundary: the covariate block runs
    to the start of the trend, not to the start of the chunk offsets."""
    import matplotlib.pyplot as plt
    from timex import plot

    fig = plot.systematics(_fit_with_chunk_offsets(gapped_lc_aux), 'g', style=2)
    try:
        # plot_basis draws one line per basis vector, then a black sum line
        basis = [ln for ln in _panel(fig, 'covariates').lines
                 if ln.get_color() != 'k']
        assert len(basis) == 1
        ydata = basis[0].get_ydata()
    finally:
        plt.close(fig)

    # airmass is 1.2 + 0.01*i standardized over the eight points: (i - 3.5)
    # divided by the population std sqrt(42/8) of [0..7]
    assert ydata == pytest.approx(
        (np.arange(8) - 3.5) / np.sqrt(42 / 8), rel=1e-9)


def _fit_without_covariates(fp):
    """A fit stub over a real read_generic design matrix holding a trend, a
    spline and two chunk offsets, and no covariate at all.

    Counting covariates as the columns the config cannot account for gives
    8 - (1 trend + 5 spline + 0 bias) = 2, the chunk offsets, so it reports
    covariates that do not exist.
    """
    from timex import io

    x, _, _, X, _, _, _, layout = io.read_generic(
        fp, binsize=None, trend=1, spline=True, spline_knots=5,
        chunk_offset=True, chunk_thresh=0.02, verbose=False)

    class _Fit:
        use_gp = False
        masks = {'g': None}
        map_soln = {'g_weights': np.arange(X.shape[1], dtype=float)}
        data = {'g': dict(x=x, X=X, ncols=layout, band='g')}
        fit_params = {'data': {'g': dict(trend=1, spline=True, spline_knots=5,
                                         add_bias=False, chunk_offset=True,
                                         chunk_thresh=0.02)}}

    return _Fit()


def test_systematics_draws_no_covariate_panel_when_there_are_no_covariates(gapped_lc):
    """Whether a covariates panel is drawn decides how many panels there are
    and which column each of the others lands in.

    The chunk offsets are appended after every block the config names, so a
    dataset with a trend and no covariate still has more columns than the
    config accounts for. Reading that difference as a covariate count opens an
    empty leading panel and pushes the trend, the spline and the sum one
    column right.
    """
    import matplotlib.pyplot as plt
    from timex import plot

    fig = plot.systematics(_fit_without_covariates(gapped_lc), 'g', style=2)
    try:
        titles = [ax.get_title() for ax in fig.axes]
    finally:
        plt.close(fig)

    assert titles == ['trend', 'spline', 'sum']


def test_systematics_draws_the_gp_when_there_is_no_design_matrix():
    """A GP only fit has no design matrix, and so no {name}_weights site
    either, so reading X.shape or that key raises.

    sample() calls plot_systematics for every dataset, which puts the crash
    after the sampling has already finished.
    """
    import matplotlib.pyplot as plt
    from timex import io, plot

    n = 12
    x = np.linspace(0.0, 1.0, n)
    gp_pred = np.sin(5 * x)

    class _Fit:
        use_gp = True
        masks = {'g': None}
        map_soln = {'g_gp_pred': gp_pred}
        data = {'g': dict(x=x, X=None, ncols=dict.fromkeys(io.COLUMN_BLOCKS, 0),
                          band='g')}
        fit_params = {'data': {'g': dict(trend=None, spline=False,
                                         spline_knots=5, add_bias=False,
                                         chunk_offset=False, chunk_thresh=0.02)}}

    fig = plot.systematics(_Fit(), 'g', style=2)
    try:
        ydata = _panel(fig, 'GP').lines[0].get_ydata()
    finally:
        plt.close(fig)

    assert ydata == pytest.approx(gp_pred)


def test_systematics_style_1_skips_a_dataset_with_no_design_matrix():
    """Style 1 has no GP panel, so a GP only fit leaves it nothing to draw and
    plt.subplots(2, 0) raises. It has to skip the way it does for any other
    dataset without enough components.
    """
    from timex import io, plot

    n = 12
    x = np.linspace(0.0, 1.0, n)

    class _Fit:
        use_gp = True
        masks = {'g': None}
        map_soln = {'g_gp_pred': np.sin(5 * x)}
        data = {'g': dict(x=x, X=None, ncols=dict.fromkeys(io.COLUMN_BLOCKS, 0),
                          band='g')}
        fit_params = {'data': {'g': dict(trend=None, spline=False,
                                         spline_knots=5, add_bias=False,
                                         chunk_offset=False, chunk_thresh=0.02)}}

    assert plot.systematics(_Fit(), 'g', style=1) is None
