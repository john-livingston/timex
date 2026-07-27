import numpy as np
import pandas as pd

from timex import util


def test_bin_df_error_is_median_error_over_sqrt_count():
    n = 60
    binsize = 60 / 86400.
    t = np.linspace(0, 3 * binsize, n, endpoint=False)
    df = pd.DataFrame({
        'time': t,
        'flux': np.ones(n),
        'fluxerr': np.full(n, 0.03),
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=binsize)

    bins = np.arange(df['time'].min(), df['time'].max(), binsize)
    groups = df.groupby(np.digitize(df['time'], bins))
    expected = (groups['fluxerr'].median() / np.sqrt(groups.size())).dropna()
    assert np.allclose(binned['fluxerr'].values, expected.values)


def test_bin_df_flux_is_bin_median():
    n = 40
    binsize = 60 / 86400.
    t = np.linspace(0, 2 * binsize, n, endpoint=False)
    df = pd.DataFrame({
        'time': t,
        'flux': np.arange(n, dtype=float),
        'fluxerr': np.full(n, 0.01),
    })

    binned = util.bin_df(df, 'time', 'fluxerr', binsize=binsize)

    bins = np.arange(df['time'].min(), df['time'].max(), binsize)
    expected = df.groupby(np.digitize(df['time'], bins))['flux'].median()
    assert np.allclose(binned['flux'].values, expected.dropna().values)


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

    bins = np.arange(df['time'].min(), df['time'].max(), binsize)
    groups = df.groupby(np.digitize(df['time'], bins))
    expected_flux_mean = groups['flux'].mean().dropna()
    expected_flux_median = groups['flux'].median().dropna()
    expected_err = (groups['fluxerr'].median() / np.sqrt(groups.size())).dropna()

    # confirm the fixture actually distinguishes mean from median in each bin
    assert not np.allclose(expected_flux_mean.values, expected_flux_median.values)

    assert np.allclose(binned['flux'].values, expected_flux_mean.values)
    assert np.allclose(binned['fluxerr'].values, expected_err.values)


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


def test_get_corrected_without_linear_model(map_soln):
    del map_soln['g_lm']
    data = _corrected_data()
    cor = util.get_corrected(data, 'g', map_soln, 1, subtract_tc=False)
    # y is zeros, only g_mean = 0.1 is subtracted
    assert np.allclose(cor['y'], -0.1)


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
    samples = np.full(100, 0.25)
    lines = util.format_tc_lines(['b'], 2460000.0, t0_samples=samples)
    assert len(lines) == 1
    planet, tc, unc = lines[0].split()
    assert planet == 'b'
    assert float(tc) == 2460000.25
    assert float(unc) == 0.0


def test_format_tc_lines_from_samples_two_planets():
    samples = np.vstack([np.full(50, 0.1), np.full(50, 0.2)])
    lines = util.format_tc_lines(['b', 'c'], 2460000.0, t0_samples=samples)
    assert len(lines) == 2
    assert float(lines[0].split()[1]) == 2460000.1
    assert float(lines[1].split()[1]) == 2460000.2


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


def test_get_outlier_mask_without_mean_site(map_soln):
    # include_mean=False means model_fn never creates a {name}_mean site;
    # get_outlier_mask must tolerate that the same way model.py does
    del map_soln['g_mean']
    n = 100
    x = np.linspace(0.0, 0.1, n)
    y = np.zeros(n)

    mask = util.get_outlier_mask(x, y, 'g', map_soln, use_gp=False)

    assert mask.shape == (n,)
    assert mask.dtype == bool


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
