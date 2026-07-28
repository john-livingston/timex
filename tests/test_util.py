import numpy as np
import pandas as pd
import pytest

from timex import util


def test_bin_df_median_error_carries_the_median_inflation_factor():
    """A binned point is the median of its bin, and the median of a Gaussian
    sample scatters more than its mean, so the standard error of the mean
    understates it. Without the sqrt(pi/2) inflation every binned error in
    every fit is about 20 percent too small.
    """
    n = 20
    binsize = 60 / 86400.
    t = np.linspace(0, 4 * binsize, n, endpoint=False)
    df = pd.DataFrame({
        'time': t,
        'flux': np.ones(n),
        'fluxerr': np.full(n, 0.05),
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=binsize)

    # 4 bins of 5 points, each point carrying an error of 0.05, so the
    # standard error of the mean is 0.05/sqrt(5) = 0.0223607 and the inflated
    # median error is sqrt(pi/2) * 0.0223607 = 0.0280250
    assert len(binned) == 4
    assert np.allclose(binned['fluxerr'].values, 0.0280250, atol=1e-6)


def _skewed_bins(n_bins=3, per_bin=5):
    """Bins of five points holding four copies of 1.0 and one copy of 20.0.

    Every bin's median is 1.0 and every bin's mean is (4*1 + 20)/5 = 4.8, so
    the two branches of bin_df cannot be confused for one another. Times sit at
    0.1 .. 0.5 inside unit bins so np.digitize gives each bin exactly five
    points and no point lands on a bin edge.
    """
    offsets = np.linspace(0.1, 0.5, per_bin)
    t = (np.arange(n_bins)[:, None] + offsets[None, :]).ravel()
    return pd.DataFrame({
        'time': t,
        'flux': np.tile([1.0, 1.0, 1.0, 1.0, 20.0], n_bins),
        'fluxerr': np.full(t.size, 0.01),
    })


def test_bin_df_flux_is_bin_median_by_default():
    """kind defaults to 'median', and the default is what every read time call
    in io.py uses. Hand derived: the median of each bin is 1.0, where the mean
    would be 4.8, so swapping the branches or changing the default is visible.
    """
    binned = util.bin_df(_skewed_bins(), 'time', 'fluxerr', binsize=1.0)

    assert len(binned) == 3
    assert np.allclose(binned['flux'].values, 1.0)


def test_bin_df_median_default_survives_a_single_ruined_frame():
    """Why the default is the median. Binning happens at read time, before any
    outlier clipping can help, so one bad frame in a bin would otherwise carry
    into the fit. Hand derived on ten points, nine at 1.0 and one at 1.5: the
    median is 1.0 exactly and the mean is (9 + 1.5)/10 = 1.05.
    """
    t = 0.1 * np.arange(1, 11)
    flux = np.full(10, 1.0)
    flux[0] = 1.5
    df = pd.DataFrame({'time': t, 'flux': flux, 'fluxerr': np.full(10, 0.01)})
    kwargs = dict(timecol='time', errcol='fluxerr', binsize=1.0)

    assert util.bin_df(df, kind='median', **kwargs)['flux'].iloc[0] == \
        pytest.approx(1.0, abs=1e-12)
    assert util.bin_df(df, kind='mean', **kwargs)['flux'].iloc[0] == \
        pytest.approx(1.05, abs=1e-12)


@pytest.mark.parametrize('kind,expected', [
    # median(err)/sqrt(5) = 0.01/2.2360680 = 0.0044721360
    ('mean', 0.004472135954999579),
    # and the same, inflated by sqrt(pi/2) for the median branch
    ('median', 0.0056049912163979275),
])
def test_bin_df_error_is_the_bin_median_error_not_its_mean(kind, expected):
    """The per point error entering the sqrt(N) average is the bin's median
    error in both branches, for the same reason the flux is: one ruined frame
    must not set the error of its whole bin. Hand derived on five points with
    errors 0.01, 0.01, 0.01, 0.01, 0.10, whose median is 0.01 and whose mean is
    0.028, a factor of 2.8 apart.
    """
    df = pd.DataFrame({
        'time': np.linspace(0.1, 0.5, 5),
        'flux': np.ones(5),
        'fluxerr': np.array([0.01, 0.01, 0.01, 0.01, 0.10]),
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=1.0, kind=kind)

    assert binned['fluxerr'].iloc[0] == pytest.approx(expected, rel=1e-12)


def test_bin_df_mean_flux_and_median_error():
    n = 60
    binsize = 60 / 86400.
    t = np.linspace(0, 3 * binsize, n, endpoint=False)
    # skewed pattern: four low values and one high outlier per group of 5,
    # so the per-bin mean and median are provably different
    flux = np.tile([1.0, 1.0, 1.0, 1.0, 20.0], n // 5)
    fluxerr = np.tile([0.01, 0.02, 0.03, 0.04, 0.05], n // 5)
    df = pd.DataFrame({
        'time': t,
        'flux': flux,
        'fluxerr': fluxerr,
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=binsize, kind='mean')

    # 3 bins of 20 points. each bin holds four copies of the 5 point pattern,
    # so its mean is (16*1 + 4*20)/20 = 4.8 while its median is 1.0
    assert len(binned) == 3
    assert np.allclose(binned['flux'].values, 4.8)
    # the binned point is the mean here, whose standard error needs no median
    # inflation: the median point error is 0.03 and sqrt(20) = 4.4721
    assert np.allclose(binned['fluxerr'].values, 0.00670820, atol=1e-7)


def test_bin_df_median_error_matches_the_scatter_of_the_binned_points():
    """The point of the error column: it has to describe how much a binned
    point actually moves. Measured against 1500 independent bins of 9 Gaussian
    draws, the reported error must land on that scatter, slightly high rather
    than low. The uncorrected standard error of the mean comes out at 0.82 of
    it, which is the understatement this guards.
    """
    sigma = 0.05
    nbins, per_bin = 1500, 9
    # unit bins with the 9 points at 0.1 ... 0.9 within each: no point sits
    # near a bin edge, so every bin holds exactly 9 points
    offsets = np.arange(1, per_bin + 1) / 10.
    t = (np.arange(nbins)[:, None] + offsets[None, :]).ravel()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'time': t,
        'flux': rng.normal(0., sigma, t.size),
        'fluxerr': np.full(t.size, sigma),
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=1.)

    assert len(binned) == nbins, 'fixture must give every bin the same count'
    reported = binned['fluxerr'].values
    assert np.allclose(reported, reported[0]), 'every bin holds 9 equal errors'
    observed = binned['flux'].values.std(ddof=1)
    ratio = reported[0] / observed
    # 1500 bins pin the observed scatter to a few percent; sqrt(pi/2) is the
    # asymptotic factor and N=9 is short of asymptotic, so a little high is
    # expected and correct. the uncorrected 0.82 is far outside
    assert 0.95 < ratio < 1.12, f'reported/observed error ratio was {ratio:.3f}'


def test_count_free_params_counts_scalars_and_vectors(map_soln):
    # t0, period, ror, b, dur = 1 each; u_star_g = 2; g_mean = 1; g_log_sigma_lc = 1
    assert util.count_free_params(map_soln) == 9


def test_count_free_params_excludes_derived_quantities(map_soln):
    # every derived suffix must be ignored no matter how large the array is
    baseline = util.count_free_params(map_soln)
    for suffix in util.DERIVED_SUFFIXES:
        map_soln[f'g{suffix}'] = np.zeros(500)
    assert util.count_free_params(map_soln) == baseline


def test_count_free_params_excludes_gp_predictions(map_soln):
    baseline = util.count_free_params(map_soln)
    map_soln['g_gp_pred'] = np.zeros(100)
    assert util.count_free_params(map_soln) == baseline


def _corrected_data(n=100):
    return dict(
        x=np.linspace(0.0, 0.1, n),
        y=np.zeros(n),
        yerr=np.full(n, 0.5),
        x_hr=np.linspace(0.0, 0.1, 500),
    )


def test_get_sys_model_sums_present_components(map_soln):
    map_soln['g_flare'] = np.full(100, 0.5)
    map_soln['g_gp_pred'] = np.full(100, 0.25)
    # g_mean 0.1 + g_lm 0.2 + g_flare 0.5 + g_gp_pred 0.25
    sys_mod = util.get_sys_model('g', map_soln, 100)
    assert sys_mod.shape == (100,)
    assert np.allclose(sys_mod, 1.05)


def test_get_sys_model_tolerates_missing_components():
    soln = {'g_mean': np.array(0.3)}
    assert np.allclose(util.get_sys_model('g', soln, 10), 0.3)


def test_get_sys_model_without_linear_model_does_not_raise():
    soln = {'g_mean': np.array(0.0), 'g_gp_pred': np.full(10, 2.0)}
    assert np.allclose(util.get_sys_model('g', soln, 10), 2.0)


def test_get_corrected_removes_gp_component(map_soln):
    data = _corrected_data()
    without_gp = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)

    map_soln['g_gp_pred'] = np.full(100, 0.75)
    with_gp = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)

    # the GP trend must be subtracted out, not left in the corrected flux
    assert np.allclose(without_gp['y'] - with_gp['y'], 0.75)


def test_get_corrected_removes_flare_and_bump(map_soln):
    data = _corrected_data()
    baseline = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)

    map_soln['g_flare'] = np.full(100, 1.5)
    map_soln['g_bump'] = np.full(100, 0.5)
    corrected = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)

    assert np.allclose(baseline['y'] - corrected['y'], 2.0)


def test_get_corrected_sums_the_high_resolution_planet_axis(map_soln_multiplanet):
    """tra_mod_hr is what the model curve is drawn from, so a two planet fit
    must plot both transits, and t0 is per planet once nplanets > 1."""
    data = _corrected_data()

    cor = util.get_corrected(data, 'g', map_soln_multiplanet, 2, subtract_tc=False)
    assert cor['tra_mod_hr'].shape == (500,)
    # either planet alone would give -1.0 or -0.25
    assert np.allclose(cor['tra_mod_hr'], -1.25)

    shifted = util.get_corrected(data, 'g', map_soln_multiplanet, 2, subtract_tc=True)
    # the first planet's t0, not the whole (2,) vector
    assert np.allclose(shifted['x'], data['x'] - 0.05)


def test_get_corrected_without_linear_model(map_soln):
    del map_soln['g_lm']
    data = _corrected_data()
    cor = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)
    # y is zeros, only g_mean = 0.1 is subtracted
    assert np.allclose(cor['y'], -0.1)


def test_get_corrected_error_includes_the_fitted_jitter(map_soln):
    """The likelihood weights each point by sqrt(yerr**2 + exp(2*log_sigma_lc)),
    and the fitted jitter routinely exceeds the photometric error, so a
    corrected light curve carrying bare photometric errors understates the
    scatter anyone refitting it will find.
    """
    data = _corrected_data()
    # g_log_sigma_lc is -1, so the jitter is exp(-1) ppt and its square is
    # exp(-2) = 0.1353353. with yerr = 0.5 ppt the combined error is
    # sqrt(0.25 + 0.1353353) = 0.6207538 ppt
    cor = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)
    assert np.allclose(cor['yerr'], 0.6207538, atol=1e-7)


def test_get_corrected_error_without_a_jitter_site(map_soln):
    """Not every configuration samples a jitter, and get_corrected is also
    called on solutions loaded from disk, so a missing site must leave the
    photometric error alone rather than raise."""
    del map_soln['g_log_sigma_lc']
    data = _corrected_data()
    cor = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)
    assert np.allclose(cor['yerr'], 0.5)


def test_get_corrected_respects_mask(map_soln):
    data = _corrected_data()
    mask = np.zeros(100, dtype=bool)
    mask[:40] = True
    # in real usage soln arrays are only ever computed on x[mask] (see
    # model.py), so g_lm must already be sized to mask.sum(), not the full
    # unmasked length
    map_soln['g_lm'] = np.full(40, 0.2)
    cor = util.get_corrected(data, 'g', map_soln, 1, mask=mask, subtract_tc=False)
    assert cor['y'].shape == (40,)
    assert cor['yerr'].shape == (40,)


def test_get_residuals_subtracts_all_components(map_soln):
    map_soln['g_gp_pred'] = np.full(100, 0.4)
    y = np.zeros(100)
    # transit -1.0, mean 0.1, lm 0.2, gp 0.4 -> resid = 0 - (-1.0) - 0.7
    resid = util.get_residuals('g', y, map_soln)
    assert np.allclose(resid, 0.3)


def test_get_residuals_sums_over_the_planet_axis(map_soln_multiplanet):
    """Two planets keep a planet axis that has to be summed, not indexed."""
    soln = map_soln_multiplanet
    soln['g_gp_pred'] = np.full(100, 0.4)
    y = np.zeros(100)
    # transits -1.0 and -0.25, mean 0.1, lm 0.2, gp 0.4
    resid = util.get_residuals('g', y, soln)
    # either planet alone would give 0.30 or 1.05
    assert np.allclose(resid, 0.55)


def test_get_outlier_mask_sums_over_the_planet_axis(map_soln_multiplanet):
    """An outlier is only detectable against the full transit model.

    y follows mean + both transits + lm plus a small alternating scatter, with
    one deliberate spike. Dropping the second planet leaves a 0.25 offset that
    inflates the robust rms fivefold, and the spike then falls inside the
    threshold and is kept.
    """
    n = 100
    soln = map_soln_multiplanet
    mod = 0.1 + (-1.0) + (-0.25) + 0.2   # mean + both transits + lm
    resid = np.where(np.arange(n) % 2 == 0, 0.05, -0.05)
    resid[10] = 1.0
    y = mod + resid
    x = np.linspace(0.0, 0.1, n)

    mask = util.get_outlier_mask(x, y, 'g', soln, use_gp=False)

    assert not mask[10], 'the injected outlier was not clipped'
    assert mask.sum() == n - 1, 'only the injected outlier may be clipped'


def test_get_var_names_omits_fixed_t0():
    var_names = util.get_var_names(
        data={'g': {}}, bands=['g'], fit_basis='duration',
        use_gp=False, fixed=['t0'])
    assert 't0' not in var_names


def test_get_var_names_includes_free_t0():
    var_names = util.get_var_names(
        data={'g': {}}, bands=['g'], fit_basis='duration',
        use_gp=False, fixed=[])
    assert 't0' in var_names


def test_format_tc_lines_from_samples_single_planet():
    """tc.txt reports the posterior mean plus ref_time and the posterior width.

    Hand derived on samples 0.4, 0.5, 0.6: the mean is 0.5 and the population
    standard deviation is sqrt(0.02/3) = 0.0816496580927726. The sample
    standard deviation would be sqrt(0.02/2) = 0.1 and the variance would be
    0.00667, so the reported uncertainty pins which statistic is used.
    """
    samples = np.array([0.4, 0.5, 0.6])
    lines = util.format_tc_lines(['b'], 2460000.0, t0_samples=samples)
    assert len(lines) == 1
    planet, tc, unc = lines[0].split()
    assert planet == 'b'
    assert float(tc) == pytest.approx(2460000.5, abs=1e-9)
    assert float(unc) == pytest.approx(0.0816496580927726, rel=1e-9)


def test_format_tc_lines_from_samples_two_planets():
    """Each planet gets its own row from its own samples. Hand derived: planet
    b has mean 0.5 and width sqrt(0.02/3), planet c has mean 1.2 and exactly
    twice that width. Transposing the array swaps the rows, and pooling all six
    samples gives both planets the same mean of 0.85.
    """
    samples = np.array([[0.4, 0.5, 0.6], [1.0, 1.2, 1.4]])
    lines = util.format_tc_lines(['b', 'c'], 2460000.0, t0_samples=samples)
    assert len(lines) == 2

    assert lines[0].split()[0] == 'b'
    assert float(lines[0].split()[1]) == pytest.approx(2460000.5, abs=1e-9)
    assert float(lines[0].split()[2]) == pytest.approx(0.0816496580927726, rel=1e-9)

    assert lines[1].split()[0] == 'c'
    assert float(lines[1].split()[1]) == pytest.approx(2460001.2, abs=1e-9)
    assert float(lines[1].split()[2]) == pytest.approx(0.1632993161855452, rel=1e-9)


def test_format_tc_lines_from_fixed_value():
    lines = util.format_tc_lines(['b'], 2460000.0, t0_fixed=np.array([0.5]))
    assert lines == ['b 2460000.5 0.0']


def test_get_map_soln_selects_max_logp_sample():
    import arviz as az
    import xarray as xr

    posterior = xr.Dataset(
        {'a': (('chain', 'draw'), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={'chain': [0, 1], 'draw': [0, 1]},
    )
    sample_stats = xr.Dataset(
        {'lp': (('chain', 'draw'), np.array([[0.0, 1.0], [5.0, 2.0]]))},
        coords={'chain': [0, 1], 'draw': [0, 1]},
    )
    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    soln, max_lp = util.get_map_soln(idata)

    # max lp is at chain=1, draw=0, where a == 3.0
    assert soln['a'] == 3.0
    assert max_lp == 5.0


def test_get_map_soln_handles_potential_energy():
    import arviz as az
    import xarray as xr

    posterior = xr.Dataset(
        {'a': (('chain', 'draw'), np.array([[1.0, 2.0]]))},
        coords={'chain': [0], 'draw': [0, 1]},
    )
    # potential_energy is -logp, so the smaller value is the better sample
    sample_stats = xr.Dataset(
        {'potential_energy': (('chain', 'draw'), np.array([[9.0, 1.0]]))},
        coords={'chain': [0], 'draw': [0, 1]},
    )
    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    soln, max_lp = util.get_map_soln(idata)

    assert soln['a'] == 2.0
    assert max_lp == -1.0


def test_get_map_soln_preserves_vector_variables():
    import arviz as az
    import xarray as xr

    posterior = xr.Dataset(
        {'v': (('chain', 'draw', 'v_dim'), np.arange(8.0).reshape(1, 2, 4))},
        coords={'chain': [0], 'draw': [0, 1]},
    )
    sample_stats = xr.Dataset(
        {'lp': (('chain', 'draw'), np.array([[0.0, 7.0]]))},
        coords={'chain': [0], 'draw': [0, 1]},
    )
    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    soln, _ = util.get_map_soln(idata)

    assert np.allclose(soln['v'], [4.0, 5.0, 6.0, 7.0])


def _clip_fixture():
    """21 points alternating +/-1 with a single 20 sigma spike at index 10.

    Against a zero model the residual squares are 1 at twenty points and 400 at
    the spike, so the robust rms is sqrt(median) = 1 and at nsig=7 the spike is
    the only point outside the threshold. The mean of the squares is 420/21 =
    20, giving an rms of 4.47 and a threshold of 31.3 that keeps the spike, so
    this fixture separates median(resid**2) from mean(resid**2).
    """
    n = 21
    x = np.arange(n, dtype=float)
    y = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    y[10] = 20.0
    return x, y, n


def test_get_outlier_mask_uses_the_robust_rms():
    """A single bad frame must not set the scale it is judged against. Using
    the mean of the squared residuals lets the spike inflate its own threshold
    and every point survives, which is clipping that quietly does nothing.
    """
    x, y, n = _clip_fixture()
    soln = {'g_light_curves': np.zeros(n)}

    mask = util.get_outlier_mask(x, y, 'g', soln, use_gp=False)

    assert not mask[10], 'the 20 sigma spike was not clipped'
    assert mask.sum() == 20, 'only the spike may be clipped'


def test_get_outlier_mask_without_mean_site():
    # include_mean=False means model_fn never creates a {name}_mean site;
    # get_outlier_mask must tolerate that the same way model.py does, and
    # treating it as zero is what makes the spike above the outlier
    x, y, n = _clip_fixture()
    soln = {'g_light_curves': np.zeros(n)}

    mask = util.get_outlier_mask(x, y, 'g', soln, use_gp=False)

    assert mask.shape == (n,)
    assert mask.dtype == bool
    assert mask.sum() == 20


def test_get_outlier_mask_subtracts_the_mean_before_clipping():
    """The control for the guard above: a missing mean is zero, but a present
    one has to be used. At a mean of 20 the spike becomes the typical point and
    the ordinary points become the large residuals, so the median of the squares
    is 19**2 and nothing is far enough out to clip. Ignoring the mean instead of
    defaulting it would clip index 10 here.
    """
    x, y, n = _clip_fixture()
    soln = {'g_light_curves': np.zeros(n), 'g_mean': np.array(20.0)}

    mask = util.get_outlier_mask(x, y, 'g', soln, use_gp=False)

    assert mask.all()


def test_get_map_soln_ignores_nan_logp():
    import arviz as az
    import xarray as xr

    posterior = xr.Dataset(
        {'a': (('chain', 'draw'), np.array([[1.0, 2.0], [3.0, 4.0]]))},
        coords={'chain': [0, 1], 'draw': [0, 1]},
    )
    # nan at chain=0,draw=0 comes before the true max (chain=1,draw=1) in
    # flattened order, so a plain np.argmax would pick the nan instead
    sample_stats = xr.Dataset(
        {'lp': (('chain', 'draw'), np.array([[np.nan, 1.0], [2.0, 5.0]]))},
        coords={'chain': [0, 1], 'draw': [0, 1]},
    )
    idata = az.InferenceData(posterior=posterior, sample_stats=sample_stats)

    soln, max_lp = util.get_map_soln(idata)

    # true max lp (ignoring nan) is at chain=1, draw=1, where a == 4.0
    assert soln['a'] == 4.0
    assert np.isfinite(max_lp)
    assert max_lp == 5.0


def test_get_map_soln_preserves_free_param_shapes():
    """A MAP is fed back to init_to_value on resume, and numpyro propagates the
    init shape into the sampled site. Collapsing (1,) to a scalar here makes a
    resumed run's posterior differently shaped from a fresh run's.
    """
    import arviz as az
    from timex import util

    idata = az.from_dict(
        posterior={
            't0': np.zeros((2, 5, 1)),           # free param, shape (1,) per draw
            'g_log_sigma_lc': np.zeros((2, 5)),  # free param, genuinely 0-d per draw
            'g_light_curves': np.zeros((2, 5, 10, 1)),  # derived, squeezed as before
        },
        sample_stats={'lp': np.zeros((2, 5))},
    )
    soln, _ = util.get_map_soln(idata)

    assert soln['t0'].shape == (1,), 'free param lost its trailing singleton dim'
    assert soln['g_log_sigma_lc'].shape == (), 'genuinely 0-d site was inflated'
    assert soln['g_light_curves'].shape == (10,), 'derived entry should still be squeezed'


def test_aicc_is_nan_when_denominator_is_not_positive(caplog):
    """AICc is undefined once nparams reaches ndata-1. Silently returning a
    negative number is how a broken parameter count went unnoticed before."""
    import logging
    from timex import util

    with caplog.at_level(logging.WARNING):
        ic = util.compute_ic(-100.0, nparams=50, ndata=50,
                             method='AICc', verbose=False)
    assert np.isnan(ic)
    assert '50' in caplog.text


def test_bic_penalty_counts_the_data_not_the_parameters():
    """These are the numbers written to ic.txt, so the formula has to be pinned
    by value. Hand derived: -2*(-100) + 3*ln(100) = 200 + 13.8155105579643.
    Reading log(nparams) instead gives 200 + 3*ln(3) = 203.30, and dropping the
    factor of 2 on the likelihood gives 113.82.
    """
    from timex import util

    assert util.compute_ic(-100.0, nparams=3, ndata=100,
                           method='BIC', verbose=False) == \
        pytest.approx(213.8155105579643, abs=1e-9)


def test_aic_penalty_is_twice_the_parameter_count():
    """Hand derived: 2*3 - 2*(-100) = 206. A penalty of nparams rather than
    2*nparams gives 203, and AIC would then rank models like a half strength
    BIC without anything saying so.
    """
    from timex import util

    assert util.compute_ic(-100.0, nparams=3, ndata=100,
                           method='AIC', verbose=False) == \
        pytest.approx(206.0, abs=1e-9)


def test_aicc_adds_the_small_sample_correction_to_aic():
    """Hand derived: AIC 206 + 2*(3**2 + 3)/(100 - 3 - 1) = 206 + 24/96 =
    206.25. Flipping nparams**2 + nparams to a difference gives 206.125, and
    dropping the leading 2 gives 206.125 as well, so the value is what
    separates them from the correct formula.
    """
    from timex import util

    assert util.compute_ic(-100.0, nparams=3, ndata=100,
                           method='AICc', verbose=False) == \
        pytest.approx(206.25, abs=1e-9)


def test_aicc_is_finite_just_above_the_denominator_boundary():
    """The control for the nan guard: at ndata - nparams - 1 = 1 the criterion
    is still defined, so the guard must not be satisfiable by returning nan for
    every small sample. Hand derived: 206 + 2*12/1 = 230.
    """
    from timex import util

    assert util.compute_ic(-100.0, nparams=3, ndata=5,
                           method='AICc', verbose=False) == \
        pytest.approx(230.0, abs=1e-9)


def test_bic_and_aic_unaffected_by_the_aicc_guard():
    """nparams=50 against ndata=50 is where AICc returns nan, and BIC and AIC
    are defined there. Hand derived: BIC 200 + 50*ln(50) = 395.6011502714073,
    AIC 2*50 + 200 = 300.
    """
    from timex import util

    assert util.compute_ic(-100.0, nparams=50, ndata=50,
                           method='BIC', verbose=False) == \
        pytest.approx(395.6011502714073, abs=1e-9)
    assert util.compute_ic(-100.0, nparams=50, ndata=50,
                           method='AIC', verbose=False) == \
        pytest.approx(300.0, abs=1e-9)


def test_compute_ic_rejects_an_unknown_method():
    """Without a final else the if/elif chain falls through with `ic` unbound,
    so a caller asking for an unsupported criterion gets an UnboundLocalError
    about a local variable rather than a message naming what it passed."""
    with pytest.raises(ValueError, match='WAIC'):
        util.compute_ic(-100.0, nparams=3, ndata=50,
                        method='WAIC', verbose=False)


def test_count_gp_hyper_ignores_a_dataset_named_gp():
    """A dataset called 'gp' has a jitter site 'gp_log_sigma_lc'. Matching on
    the 'gp_log_' prefix swallows it, so the GP is credited with a
    hyperparameter that is really that dataset's jitter and the edf corrected
    parameter count comes out one too low."""
    soln = {
        'gp_log_amp': np.array([0.0]),
        'gp_log_scale': np.array([-2.0]),
        'gp_log_sigma_lc': np.array(-1.0),
        'gp_mean': np.array(0.1),
        't0': np.array([0.05]),
    }
    assert util.count_gp_hyper(soln) == 2


def test_count_gp_hyper_counts_per_dataset_sites():
    soln = {
        'gp_log_amp_g': np.array([0.0]),
        'gp_log_scale_g': np.array([-2.0]),
        'gp_log_amp_r': np.array([0.1]),
        'gp_log_scale_r': np.array([-2.1]),
        'g_log_sigma_lc': np.array(-1.0),
    }
    assert util.count_gp_hyper(soln) == 4


def test_count_gp_hyper_counts_elements_not_sites():
    """nparams elsewhere is a count of elements (see count_free_params), so
    subtracting a count of sites would mix two different units the moment a
    hyperparameter becomes vector valued."""
    soln = {
        'gp_log_amp': np.zeros(3),
        'gp_log_scale': np.zeros(3),
    }
    assert util.count_gp_hyper(soln) == 6


@pytest.mark.parametrize('band,expected', [
    # Sloan filters: claret distinguishes these from the Stromgren filters of
    # the same letter by a trailing asterisk
    ('g', 'g*'),
    ('r', 'r*'),
    ('i', 'i*'),
    ('z', 'z*'),
    # multi character names that are substrings of 'griz'. these are the cases
    # `band in 'griz'` gets wrong: a substring test matches them and appends a
    # spurious asterisk, producing a band name claret does not have
    ('gr', 'gr'),
    ('ri', 'ri'),
    ('iz', 'iz'),
    ('gri', 'gri'),
    ('griz', 'griz'),
    ('', ''),
    # names that are not substrings, unaffected either way
    ('zs', 'zs'),
    ('ip', 'ip'),
    ('B', 'B'),
    ('Kp', 'Kp'),
])
def test_claret_band_maps_only_exact_sloan_filters(band, expected):
    assert util.claret_band(band) == expected
