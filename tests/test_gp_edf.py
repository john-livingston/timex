import numpy as np
import pytest


def _dataset(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 0.2, n))
    return {
        'g': dict(x=x, y=np.zeros(n), yerr=np.full(n, 0.2),
                  X=None, texp=0.001, x_hr=x, band='g', ref_time=0.0)
    }


def _soln(log_amp, log_scale, log_sigma_lc=np.log(0.35)):
    return {
        'gp_log_amp': np.array(float(log_amp)),
        'gp_log_scale': np.array(float(log_scale)),
        'g_log_sigma_lc': np.array(float(log_sigma_lc)),
    }


def test_edf_matches_dense_smoother_trace():
    """The identity edf = n - tr(S (K+S)^-1) must agree with the direct
    tr(K (K+S)^-1). This is the whole correctness claim of the feature."""
    from timex import model

    data = _dataset()
    x = data['g']['x']
    yerr = data['g']['yerr']
    amp, rho, jit = 1.3, 0.018, 0.35
    soln = _soln(np.log10(amp), np.log10(rho), np.log(jit))

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    diag = jit**2 + yerr**2
    d = x[:, None] - x[None, :]
    r = np.sqrt(3) * np.abs(d) / rho
    K = amp**2 * (1 + r) * np.exp(-r)
    edf_dense = np.trace(K @ np.linalg.inv(K + np.diag(diag)))

    assert edf == pytest.approx(edf_dense, abs=1e-6)


def test_negligible_amplitude_gives_near_zero_edf():
    from timex import model

    data = _dataset()
    soln = _soln(log_amp=-6.0, log_scale=-2.0)   # amplitude 1e-6 against noise ~0.4
    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']
    assert 0.0 <= edf < 0.1


def test_dominant_amplitude_approaches_n():
    from timex import model

    data = _dataset()
    n = len(data['g']['x'])
    soln = _soln(log_amp=4.0, log_scale=-3.0)    # huge amplitude, short scale
    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']
    assert edf > 0.9 * n
    assert edf <= n + 1e-6


def test_mask_restricts_the_computation():
    from timex import model

    data = _dataset()
    n = len(data['g']['x'])
    mask = np.zeros(n, dtype=bool)
    mask[:20] = True
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    edf = model.compute_gp_edf(soln, data, {'g': mask}, gp_config=None)['g']
    assert edf <= 20.0


def test_returns_none_and_warns_above_max_points(caplog):
    """O(n^2) is fine at n~140 and prohibitive at TESS scale, so it must skip
    rather than silently stall."""
    import logging
    from timex import model

    data = _dataset()
    soln = _soln(log_amp=0.0, log_scale=-2.0)
    with caplog.at_level(logging.WARNING):
        result = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None,
                                      max_points=10)
    assert result is None
    # assert on the distinctive parts, not a bare 'g', which the warning text
    # contains many times regardless
    assert 'max_points=10' in caplog.text
    assert 'dataset g' in caplog.text
    assert str(len(data['g']['x'])) in caplog.text
