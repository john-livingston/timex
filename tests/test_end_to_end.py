import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

pytestmark = pytest.mark.slow


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """One short end to end run of the shipped example, shared by every test.

    The tests below only read from it, and the fit is by far the slowest thing
    in the suite, so it runs once per module rather than once per test.
    """
    from timex import fit

    wd = tmp_path_factory.mktemp('e2e') / 'hip67522c'
    # ignore 'out': examples/hip67522c/out ships with a real trace.nc and
    # map.pkl from a previous run, and it is ~120MB. clobber=True below
    # means these files are never read, so copying them would only slow
    # down every test run for no benefit.
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))

    with open(wd / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(wd / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)

    # sampler settings are flat top level keys in fit.yaml, not nested:
    # validate() merges defaults['sampler'] into fit_params at the top level.
    # keep the run short, this checks pipeline wiring, not the science
    #
    # clobber=True is required here: examples/hip67522c/out ships with a real
    # trace.nc and map.pkl from a previous run, and copytree carries them into
    # wd. TransitFit.__init__ calls load_saved() unconditionally, and with the
    # default clobber=False it would silently adopt that pre-existing trace,
    # so build_model would skip MAP optimization and sample() would skip
    # MCMC entirely, this would make the test pass in seconds without ever
    # running the pipeline it is meant to exercise.
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True,
                           random_seed=0))

    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)
    tf.save_results()
    return wd, tf


def _unmasked(tf, name):
    mask = tf.masks[name]
    if mask is None:
        return np.ones(len(tf.data[name]['x']), dtype=bool)
    return mask


def _cor(wd, name):
    return pd.read_csv(os.path.join(wd, 'out', f'{os.path.basename(wd)}-{name}-cor.csv'))


def test_the_pipeline_writes_every_documented_output(run):
    wd, _ = run
    outdir = os.path.join(wd, 'out')
    for fn in ('tc.txt', 'ic.txt', 'summary.csv', 'map.pkl', 'trace.nc',
               'posterior_samples.csv.gz', 'cache.json'):
        assert os.path.exists(os.path.join(outdir, fn)), fn


def test_lc_pred_is_recorded_for_every_dataset(run):
    """{name}_lc_pred is a numpyro deterministic that was changed to reuse the
    already computed transit light curve rather than recompute it. Nothing in
    the production code reads its value, so this is the only thing that would
    catch a regression that breaks it.
    """
    _, tf = run
    for name in tf.data:
        n_unmasked = int(_unmasked(tf, name).sum())
        var = f'{name}_lc_pred'
        assert var in tf.trace.posterior.data_vars, f'{var} missing from posterior'
        assert tf.trace.posterior[var].shape[-1] == n_unmasked, (
            f'{var} last axis length {tf.trace.posterior[var].shape[-1]} '
            f'!= {n_unmasked} unmasked points'
        )


def test_one_corrected_light_curve_per_dataset(run):
    wd, tf = run
    cor_files = sorted(os.listdir(os.path.join(wd, 'out')))
    cor_files = [fn for fn in cor_files if fn.endswith('-cor.csv')]
    assert len(cor_files) == len(tf.data), (
        f'expected {len(tf.data)} *-cor.csv files, found {cor_files}'
    )


def test_corrected_flux_is_relative_flux_about_a_baseline_of_one(run):
    """*-cor.csv is what downstream analysis reads, and it is in relative flux
    with the baseline restored, not in the ppt the model works in.

    Dropping the `y += 1` in save_corrected leaves a light curve scattered
    about zero, which no shape or NaN assertion can see. Dropping the ppt to
    relative flux conversion leaves the scatter a thousand times too large,
    which the mean alone cannot see because ppt flux is already centered on
    zero, so both the level and the scale are pinned here.
    """
    wd, tf = run
    for name in tf.data:
        y = _cor(wd, name)['y'].values
        assert y.mean() == pytest.approx(1.0, abs=0.02), (
            f'{name}: corrected flux is not centered on 1'
        )
        # a real light curve scatters by well under a percent; in ppt the same
        # numbers would scatter by of order 1
        assert y.std() < 0.05, f'{name}: corrected flux looks like ppt, not relative'


def test_corrected_error_is_the_one_the_likelihood_used(run):
    """The published error is the photometric error and the fitted jitter added
    in quadrature, converted to relative flux, matching what the likelihood
    weights by and what fit.png draws.

    The jitter routinely exceeds the photometric error, so publishing the bare
    photometric error understates the scatter anyone refitting the file will
    find. This also pins the ppt to relative flux conversion on the error
    column, which is a separate line from the one on the flux.
    """
    wd, tf = run
    for name, data in tf.data.items():
        mask = _unmasked(tf, name)
        jitter = np.exp(float(np.squeeze(tf.map_soln[f'{name}_log_sigma_lc'])))
        expected = np.sqrt(data['yerr'][mask]**2 + jitter**2) * 1e-3

        yerr = _cor(wd, name)['yerr'].values

        assert yerr == pytest.approx(expected, rel=1e-9), f'{name}: wrong error column'
        # and the comparison is not vacuous: the jitter has to have moved it
        assert (yerr > data['yerr'][mask] * 1e-3).all(), (
            f'{name}: jitter was not added to the photometric error'
        )


def test_corrected_time_is_in_the_data_native_system(run):
    """ref_time is subtracted at read time and has to go back on, otherwise
    every published transit time is off by a few thousand days."""
    wd, tf = run
    for name, data in tf.data.items():
        x = _cor(wd, name)['x'].values
        assert x == pytest.approx(data['x'][_unmasked(tf, name)] + tf.ref_time,
                                  abs=1e-9)
        assert x.min() > 2.4e6, 'ref_time was not restored'


def test_corrected_columns_are_named_and_complete(run):
    wd, tf = run
    for name in tf.data:
        cor = _cor(wd, name)
        assert list(cor.columns) == ['x', 'y', 'yerr']
        assert len(cor) == int(_unmasked(tf, name).sum())
        assert not cor.isna().any().any()


def test_summary_reports_every_sampled_parameter_and_no_fixed_one(run):
    """summary.csv is the headline table and get_var_names decides its rows.

    Asserting the file exists cannot see a dropped row: removing the per
    dataset jitter from get_var_names silently shrinks the table, and leaving a
    fixed parameter in makes az.summary raise after sampling has finished. The
    example fixes period and u_star, so those must be absent.
    """
    wd, tf = run
    index = pd.read_csv(os.path.join(wd, 'out', 'summary.csv'), index_col=0).index
    # arviz indexes vector parameters as 'name[0]'; the claim is about which
    # parameters are reported, not about that convention
    reported = {name.split('[')[0] for name in index}

    assert {'t0', 'ror', 'b', 'dur'} <= reported
    assert {f'{name}_log_sigma_lc' for name in tf.data} <= reported
    assert 'period' not in reported, 'a fixed parameter was reported'
    assert not any(n.startswith('u_star') for n in reported), 'u_star is fixed'


def _ic_rows(wd):
    rows = {}
    for line in open(os.path.join(wd, 'out', 'ic.txt')):
        key, value = line.split()
        rows[key] = float(value)
    return rows


def test_ic_txt_reports_the_uncorrected_criteria(run):
    """The example fits no GP, so only the three plain rows are written and the
    edf block must not appear."""
    wd, _ = run
    rows = _ic_rows(wd)
    assert set(rows) == {'BIC', 'AIC', 'AICc'}
    assert all(np.isfinite(v) for v in rows.values())


def test_ic_txt_is_built_from_the_maximized_likelihood(run):
    """BIC is -2 lnL_max + k ln(n), and lnL_max is the maximized likelihood.

    save_results has its own copy of the criterion computation, separate from
    get_ic, and it is the copy that writes the file anyone reads. Feeding it
    the log probability that get_map_soln returns alongside the MAP puts the
    joint log posterior there instead, so every prior term and the
    unconstraining Jacobian enter the criteria, including the systematics
    weight prior whose width is a constant in model.py rather than a modelling
    choice.
    """
    wd, tf = run
    from timex import util

    # the maximized likelihood, written out from the log_likelihood group
    # arviz fills from the observed sites: sum the observation dimensions of
    # every site, add the sites together, then maximize over draws
    total = None
    for var in tf.trace.log_likelihood.data_vars.values():
        arr = np.asarray(var.values)
        per_draw = arr.reshape(arr.shape[0], arr.shape[1], -1).sum(axis=2)
        total = per_draw if total is None else total + per_draw
    max_loglike = float(total.max())

    expected = -2 * max_loglike + tf._count_params() * np.log(tf._count_data())

    # ic.txt is written to two decimals
    assert _ic_rows(wd)['BIC'] == pytest.approx(expected, abs=0.01)

    # and the two candidate quantities are far apart here, so the assertion
    # above is not satisfied by either one
    _, max_logp = util.get_map_soln(tf.trace)
    assert abs(max_logp - max_loglike) > 1.0
