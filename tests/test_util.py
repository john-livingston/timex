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
