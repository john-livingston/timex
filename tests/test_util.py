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
