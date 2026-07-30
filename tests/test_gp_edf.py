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


def _dense_edf(x, yerr, amp, rho, jit):
    """tr(K (K+S)^-1) for a Matern 3/2 kernel, written out densely."""
    diag = jit**2 + yerr**2
    d = x[:, None] - x[None, :]
    r = np.sqrt(3) * np.abs(d) / rho
    K = amp**2 * (1 + r) * np.exp(-r)
    return float(np.trace(K @ np.linalg.inv(K + np.diag(diag))))


def test_edf_matches_dense_smoother_trace():
    """The identity edf = n - tr(S (K+S)^-1) must agree with the direct
    tr(K (K+S)^-1). This is the whole correctness claim of the feature."""
    from timex import model

    data = _dataset()
    amp, rho, jit = 1.3, 0.018, 0.35
    soln = _soln(np.log10(amp), np.log10(rho), np.log(jit))

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    edf_dense = _dense_edf(data['g']['x'], data['g']['yerr'], amp, rho, jit)

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


MAX_LOGLIKE = -7.0
MAX_LOGP = -1.0


def _save_results_fit(tmp_path, monkeypatch, map_soln, nparams, edf,
                      trace=None, data=None, masks=None):
    """A TransitFit carrying only what save_results reads, with the edf and the
    raw parameter count pinned so ic.txt's nparams_edf row is a pure function
    of how many GP hyperparameters save_results decides to remove.

    Without a `trace`, the maximized likelihood and the maximized log posterior
    are stubbed to deliberately different numbers, so an ic.txt row built from
    the wrong one is visible. Passing a real `trace` runs both against it
    instead, and `edf=None` leaves model.compute_gp_edf real, for the test that
    cares which draw's parameters reach it.
    """
    from timex import fit

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.planets = 'b'
    tf.ref_time = 0.0
    tf.priors = {'t0': 0.1}
    tf.use_gp = True
    tf.map_soln = map_soln
    tf.data = {} if data is None else data
    tf.masks = {} if masks is None else masks
    tf.gp_config = None
    tf.model_fn = None

    if trace is None:
        class _FakeStacked:
            data_vars = {}  # no 't0' entry -> save_results takes the fixed-t0 branch

        class _FakePosterior:
            def stack(self, **kwargs):
                return _FakeStacked()

        class _FakeTrace:
            posterior = _FakePosterior()

        tf.trace = _FakeTrace()
        monkeypatch.setattr(fit.util, 'get_map_soln', lambda trace: ({}, MAX_LOGP))
        monkeypatch.setattr(fit.util, 'get_max_loglike',
                            lambda trace, model_fn=None: (MAX_LOGLIKE, (0, 0)))
        monkeypatch.setattr(fit.util, 'get_soln_at',
                            lambda trace, chain, draw: map_soln)
    else:
        tf.trace = trace

    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: nparams)
    monkeypatch.setattr(fit.TransitFit, '_count_data', lambda self: 500)
    if edf is not None:
        monkeypatch.setattr(fit.model, 'compute_gp_edf',
                            lambda soln, data, masks, gp_config: dict(edf))
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples', lambda self: None)
    monkeypatch.setattr(fit.TransitFit, 'save_corrected', lambda self: None)
    return tf


def _ic_rows(tmp_path):
    rows = (tmp_path / 'ic.txt').read_text().split('\n')
    return dict(r.split() for r in rows if r.strip())


def test_nparams_edf_does_not_charge_a_dataset_named_gp_as_a_hyperparameter(
        tmp_path, monkeypatch):
    """save_results swaps the GP hyperparameters out for the edf. A dataset
    literally named 'gp' has a jitter site 'gp_log_sigma_lc', which a
    'gp_log_' prefix match removes as though it were a GP hyperparameter,
    leaving nparams_edf one parameter short and every corrected criterion
    biased in favor of the GP model."""
    map_soln = {
        'gp_log_amp': np.array([0.0]),
        'gp_log_scale': np.array([-2.0]),
        'gp_log_sigma_lc': np.array(-1.0),
    }
    tf = _save_results_fit(tmp_path, monkeypatch, map_soln,
                           nparams=10, edf={'gp': 4.0})
    tf.save_results()

    rows = _ic_rows(tmp_path)
    assert float(rows['edf']) == 4.0
    # 10 parameters, minus the 2 GP hyperparameters, plus 4 edf
    assert float(rows['nparams_edf']) == 12.0
    # -2 * (-7) + 10 * ln(500) = 14 + 62.146081; from the log posterior
    # instead it would be 64.15
    assert float(rows['BIC']) == 76.15
    # and the corrected row is the same likelihood against nparams_edf:
    # 14 + 12 * ln(500) = 88.575297
    assert float(rows['BIC_edf']) == 88.58


MAP_DRAW_AMP = 0.05      # amplitude at the maximum posterior draw
LL_DRAW_AMP = 3.0        # amplitude at the maximum likelihood draw
RHO = 0.018
JIT = 0.35


def _two_draw_trace():
    """One chain of two draws whose best posterior draw and best likelihood
    draw are different draws, carrying different GP amplitudes.

    This is the shape of a real trace: get_map_soln takes the argmax of the
    log posterior and get_max_loglike the argmax of the likelihood, and on the
    shipped example those land 1255 draws apart.
    """
    import arviz as az

    return az.from_dict(
        posterior={
            'gp_log_amp': np.log10([[MAP_DRAW_AMP, LL_DRAW_AMP]]),
            'gp_log_scale': np.full((1, 2), np.log10(RHO)),
            'g_log_sigma_lc': np.full((1, 2), np.log(JIT)),
        },
        # draw 0 is the best posterior draw, draw 1 the best likelihood draw
        sample_stats={'lp': np.array([[10.0, 0.0]])},
        log_likelihood={'g_y_observed': np.array([[[-5.0], [-1.0]]])},
    )


def test_the_edf_penalty_is_measured_at_the_likelihood_maximizing_draw(
        tmp_path, monkeypatch):
    """BIC_edf pairs a likelihood with a penalty, and both have to describe the
    same parameter vector.

    max_loglike is maximized over draws while self.map_soln is the maximum
    posterior draw, and those are different draws. Measuring the GP's
    flexibility at the maximum posterior draw therefore charges the criteria a
    penalty the reported likelihood never paid: on the shipped trace the edf
    ranges over 24 units across draws, which is 150 BIC units on 518 points,
    and it is correlated with the likelihood rather than scattered about it.
    """
    data = _dataset()
    tf = _save_results_fit(
        tmp_path, monkeypatch,
        map_soln=_soln(np.log10(MAP_DRAW_AMP), np.log10(RHO), np.log(JIT)),
        nparams=10, edf=None, trace=_two_draw_trace(),
        data=data, masks={'g': None})

    tf.save_results()

    x, yerr = data['g']['x'], data['g']['yerr']
    expected = _dense_edf(x, yerr, LL_DRAW_AMP, RHO, JIT)
    assert float(_ic_rows(tmp_path)['edf']) == pytest.approx(expected, abs=0.005)

    # the two draws have to be tellable apart, or the assertion above would
    # hold whichever draw the penalty came from
    at_map_draw = _dense_edf(x, yerr, MAP_DRAW_AMP, RHO, JIT)
    assert abs(at_map_draw - expected) > 1.0


def test_nparams_edf_removes_every_hyperparameter_element(tmp_path, monkeypatch):
    """Counting sites rather than elements silently undercounts a vector
    valued hyperparameter, which is what a shared amplitude over several
    bands would be."""
    map_soln = {
        'gp_log_amp': np.zeros(3),
        'gp_log_scale': np.zeros(3),
        'g_log_sigma_lc': np.array(-1.0),
    }
    tf = _save_results_fit(tmp_path, monkeypatch, map_soln,
                           nparams=20, edf={'g': 5.0})
    tf.save_results()

    rows = _ic_rows(tmp_path)
    # 20 parameters, minus 6 hyperparameter elements, plus 5 edf
    assert float(rows['nparams_edf']) == 19.0
    # 14 + 20 * ln(500) = 138.292162, and 14 + 19 * ln(500) = 132.077554
    assert float(rows['BIC']) == 138.29
    assert float(rows['BIC_edf']) == 132.08


def test_save_results_survives_an_edf_failure(tmp_path, monkeypatch, caplog):
    """compute_gp_edf runs inside save_results' `with open(ic.txt)` block and
    reads GP hyperparameters back out of map_soln. If it raises (e.g. a
    force-loaded map.pkl whose gp.per_dataset no longer matches the current
    config, so _build_gp looks up the wrong gp_log_amp key), that must not
    truncate ic.txt to nothing and must not skip save_posterior_samples /
    save_corrected, which run after the `with` block.
    """
    import logging
    from timex import fit

    # a TransitFit carrying only what save_results reads; same construction
    # shortcut as test_resume.py::_bare_fit and
    # test_fit_defaults.py::test_get_ic_counts_only_unmasked_points
    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.planets = 'b'
    tf.ref_time = 0.0
    tf.priors = {'t0': 0.1}
    tf.use_gp = True
    tf.map_soln = {'gp_log_amp': np.array(1.0)}
    tf.data = {}
    tf.masks = {}
    tf.gp_config = None
    tf.model_fn = None

    class _FakeStacked:
        data_vars = {}  # no 't0' entry -> save_results takes the fixed-t0 branch

    class _FakePosterior:
        def stack(self, **kwargs):
            return _FakeStacked()

    class _FakeTrace:
        posterior = _FakePosterior()

    tf.trace = _FakeTrace()

    monkeypatch.setattr(fit.util, 'get_map_soln', lambda trace: ({}, MAX_LOGP))
    monkeypatch.setattr(fit.util, 'get_max_loglike',
                        lambda trace, model_fn=None: (MAX_LOGLIKE, (0, 0)))
    monkeypatch.setattr(fit.util, 'get_soln_at',
                        lambda trace, chain, draw: tf.map_soln)
    monkeypatch.setattr(fit.TransitFit, '_count_params', lambda self: 3)
    monkeypatch.setattr(fit.TransitFit, '_count_data', lambda self: 50)

    def _raise_stale_hyperparam_lookup(*args, **kwargs):
        raise KeyError('gp_log_amp')

    monkeypatch.setattr(fit.model, 'compute_gp_edf', _raise_stale_hyperparam_lookup)

    calls = []
    monkeypatch.setattr(fit.TransitFit, 'save_posterior_samples',
                        lambda self: calls.append('posterior'))
    monkeypatch.setattr(fit.TransitFit, 'save_corrected',
                        lambda self: calls.append('corrected'))

    with caplog.at_level(logging.WARNING):
        tf.save_results()

    ic_text = (tmp_path / 'ic.txt').read_text()
    rows = _ic_rows(tmp_path)
    # the surviving rows must also still be right: 14 + 3 * ln(50) = 25.736069
    assert float(rows['BIC']) == 25.74, (
        'the three uncorrected rows must survive an edf failure intact'
    )
    assert set(rows) == {'BIC', 'AIC', 'AICc'}
    assert 'edf' not in ic_text, 'a failed edf computation must not write corrected rows'
    assert calls == ['posterior', 'corrected'], (
        'save_posterior_samples/save_corrected must still run after an edf failure'
    )
    assert 'KeyError' in caplog.text, 'the warning must name the exception'


def test_edf_matches_the_joint_hat_matrix_trace():
    """The joint effective degrees of freedom of a parametric mean plus a GP is
    tr(P + K C^-1 (I - P)), not p + tr(K C^-1). The difference is the overlap
    between the GP and the design, which approaches p whenever the GP can
    reproduce the design columns. Reference built densely with numpy, so it
    shares no code with the implementation.
    """
    from timex import model

    rng = np.random.default_rng(0)
    n, p = 60, 3
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.35)
    X = np.column_stack([np.ones(n), x - x.mean(), rng.normal(size=n)])
    amp, rho = 1.3, 0.02

    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X,
                      texp=0.001, x_hr=x, band='g', ref_time=0.0)}
    # _soln's third argument is log_sigma_lc and already defaults to log(0.35),
    # which is the jitter the dense reference below assumes
    soln = _soln(np.log10(amp), np.log10(rho))
    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    diag = 0.35**2 + yerr**2
    d = x[:, None] - x[None, :]
    r = np.sqrt(3) * np.abs(d) / rho
    K = amp**2 * (1 + r) * np.exp(-r)
    C = K + np.diag(diag)
    Ci = np.linalg.inv(C)
    P = X @ np.linalg.inv(X.T @ Ci @ X) @ X.T @ Ci
    joint = np.trace(P + K @ Ci @ (np.eye(n) - P))

    # compute_gp_edf returns the GP's share; the p design columns are counted
    # separately in nparams, so p + edf is the joint figure
    assert p + edf == pytest.approx(joint, abs=1e-6)


def test_edf_is_strictly_below_the_gp_alone_trace_when_a_design_is_present():
    """The overlap is non-negative, so correcting it can only reduce the count.
    Guards the sign of the subtraction."""
    from timex import model

    rng = np.random.default_rng(1)
    n = 50
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    X = np.column_stack([np.ones(n), x - x.mean()])
    soln = _soln(0.0, -1.7, np.log(0.3))

    base = dict(x=x, y=np.zeros(n), yerr=yerr, texp=0.001, x_hr=x,
                band='g', ref_time=0.0)
    with_design = model.compute_gp_edf(
        soln, {'g': dict(base, X=X)}, {'g': None}, gp_config=None)['g']
    without = model.compute_gp_edf(
        soln, {'g': dict(base, X=None)}, {'g': None}, gp_config=None)['g']

    assert with_design < without
    assert without - with_design <= X.shape[1] + 1e-9, 'overlap cannot exceed p'


def test_edf_is_unchanged_for_a_gp_only_fit():
    """X is None means p = 0 and there is nothing to overlap with, so the
    corrected value must equal the plain smoother trace."""
    from timex import model

    rng = np.random.default_rng(2)
    n = 40
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    soln = _soln(0.0, -1.7, np.log(0.3))
    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=None, texp=0.001,
                      x_hr=x, band='g', ref_time=0.0)}

    edf = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)['g']

    gp, diag = model._build_gp(soln, 'g', x, yerr, None)
    basis = np.eye(n)
    inv_diag = np.array([gp.apply_inverse(basis[:, i])[i] for i in range(n)])
    assert edf == pytest.approx(float(n - np.sum(diag * inv_diag)), abs=1e-12)


def test_edf_returns_none_for_a_rank_deficient_design(caplog):
    """A singular X^T A means the design is degenerate. Reporting an
    uncorrected upper bound under a label that now promises exactness is the
    failure this change removes, so no rows are written at all."""
    import logging
    from timex import model

    rng = np.random.default_rng(3)
    n = 40
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    col = x - x.mean()
    X = np.column_stack([col, col])          # exactly collinear
    soln = _soln(0.0, -1.7, np.log(0.3))
    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X, texp=0.001,
                      x_hr=x, band='g', ref_time=0.0)}

    with caplog.at_level(logging.WARNING):
        result = model.compute_gp_edf(soln, data, {'g': None}, gp_config=None)

    assert result is None
    assert 'g' in caplog.text and 'design' in caplog.text


def test_edf_masks_the_design_matrix_the_same_as_x_and_yerr():
    """Every other test pairs a design matrix with mask=None (i.e. mask is
    all True) or pairs a real mask with X=None, so neither exercises the
    masking of X itself. Here mask keeps exactly half the points and is not a
    contiguous prefix, so a design matrix that is not restricted the same way
    as x and yerr would change both the shape of the linear algebra and the
    numeric result.
    """
    from timex import model

    rng = np.random.default_rng(4)
    n, p = 50, 2
    x = np.sort(rng.uniform(0.0, 0.2, n))
    yerr = np.full(n, 0.3)
    X = np.column_stack([np.ones(n), x - x.mean()])
    mask = np.zeros(n, dtype=bool)
    mask[::2] = True
    amp, rho = 1.1, 0.02

    data = {'g': dict(x=x, y=np.zeros(n), yerr=yerr, X=X,
                      texp=0.001, x_hr=x, band='g', ref_time=0.0)}
    soln = _soln(np.log10(amp), np.log10(rho))
    edf = model.compute_gp_edf(soln, data, {'g': mask}, gp_config=None)['g']

    xm, yerrm, Xm = x[mask], yerr[mask], X[mask]
    diag = 0.35**2 + yerrm**2
    d = xm[:, None] - xm[None, :]
    r = np.sqrt(3) * np.abs(d) / rho
    K = amp**2 * (1 + r) * np.exp(-r)
    C = K + np.diag(diag)
    Ci = np.linalg.inv(C)
    P = Xm @ np.linalg.inv(Xm.T @ Ci @ Xm) @ Xm.T @ Ci
    joint = np.trace(P + K @ Ci @ (np.eye(len(xm)) - P))

    assert p + edf == pytest.approx(joint, abs=1e-6)
