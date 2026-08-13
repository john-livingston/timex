from pathlib import Path

import pytest

from .pipeline_fixtures import copy_example, load_params, run_pipeline, use_gp


def _gp_no_mean(fit_params):
    use_gp(fit_params)
    fit_params['include_mean'] = False


def _clip_no_mean(fit_params):
    fit_params['include_mean'] = False
    for spec in fit_params['data'].values():
        spec['clip'] = True
        spec['clip_nsig'] = 3


def _setup(tmp_path, name, mutate):
    """Copy the example, shorten the sampler, apply `mutate`, return inputs."""
    wd = tmp_path / name
    copy_example(wd)
    fit_params, sys_params = load_params(wd)
    mutate(fit_params)
    return wd, fit_params, sys_params


def _assert_outputs(wd, tf):
    outdir = Path(wd) / 'out'
    assert (outdir / 'tc.txt').exists()
    assert (outdir / 'ic.txt').exists()
    assert len(sorted(outdir.glob('*-cor.csv'))) == len(tf.data)


@pytest.mark.slow
def test_pipeline_runs_for_gp(gp_fit):
    wd, tf = gp_fit
    _assert_outputs(wd, tf)


@pytest.mark.slow
def test_pipeline_runs_for_fixed_t0(fixed_t0_fit):
    wd, tf = fixed_t0_fit
    _assert_outputs(wd, tf)


@pytest.mark.slow
@pytest.mark.parametrize('name,mutate', [
    ('gp_no_mean', _gp_no_mean),
    ('clip_no_mean', _clip_no_mean),
])
def test_pipeline_runs_for_config(tmp_path, name, mutate):
    wd, fit_params, sys_params = _setup(tmp_path, name, mutate)
    tf = run_pipeline(wd, fit_params, sys_params)
    _assert_outputs(wd, tf)


@pytest.mark.slow
def test_fixed_t0_reports_zero_uncertainty(fixed_t0_fit):
    """t0 held fixed must still produce tc.txt, with zero uncertainty."""
    wd, _ = fixed_t0_fit
    line = (Path(wd) / 'out' / 'tc.txt').read_text().split()
    assert line[0] == 'c'
    assert float(line[2]) == 0.0


@pytest.mark.slow
def test_gp_ic_excludes_gp_predictions(gp_fit):
    """A GP fit must not count its per point GP prediction as free parameters.

    Before this was fixed, nparams was inflated by roughly ndata, which
    reversed the GP versus no-GP verdict on real data.
    """
    from timex import util

    _, tf = gp_fit

    ndata = sum(int(tf.masks[n].sum()) if tf.masks[n] is not None else len(d['x'])
                for n, d in tf.data.items())
    nparams = util.count_free_params(tf.map_soln)
    assert any(k.endswith('_gp_pred') for k in tf.map_soln), (
        'expected gp predictions in map_soln, test is not exercising the GP path'
    )
    assert nparams < ndata / 4, (
        f'nparams {nparams} is inflated toward ndata {ndata}: '
        'gp predictions are being counted as free parameters'
    )


def _read_ic(outdir):
    return {line.split()[0]: float(line.split()[1])
            for line in open(outdir / 'ic.txt')}


@pytest.mark.slow
def test_gp_fit_reports_edf_corrected_ic(gp_fit):
    """The GP is charged for a handful of hyperparameters but absorbs far more
    degrees of freedom, so the corrected criteria must be reported alongside."""
    wd, tf = gp_fit
    ic = _read_ic(Path(wd) / 'out')
    ndata = sum(int(tf.masks[n].sum()) if tf.masks[n] is not None else len(d['x'])
                for n, d in tf.data.items())

    for key in ('BIC', 'AIC', 'AICc', 'edf', 'nparams', 'nparams_edf',
                'BIC_edf', 'AIC_edf', 'AICc_edf'):
        assert key in ic, f'{key} missing from ic.txt'

    assert 0 < ic['edf'] < ndata
    assert ic['nparams_edf'] > ic['nparams'], (
        'the GP absorbs more degrees of freedom than its hyperparameter count'
    )
    # the correction only ever penalises, so the corrected BIC cannot be lower
    assert ic['BIC_edf'] > ic['BIC']

    # pin the arithmetic, not just its direction: nparams_edf must be the
    # uncorrected count with the GP hyperparameters swapped out for the edf
    n_gp_hyper = sum(1 for k in tf.map_soln if k.startswith('gp_log_'))
    assert n_gp_hyper > 0, 'expected GP hyperparameters in the MAP solution'
    assert ic['nparams_edf'] == pytest.approx(
        ic['nparams'] - n_gp_hyper + ic['edf'])

    # and the reported edf must be the SUM over datasets, not a mean.
    #
    # recompute at the draw save_results used, which is the likelihood
    # maximizing one. tf.map_soln is the maximum posterior draw instead, and
    # the two criteria differ by the prior and Jacobian term: on this fixture
    # that term varies about 2.5 units across draws while the likelihood
    # separates the top two draws by about 2.7, so which draw wins flips with
    # the sampling. the edf ranges over tens of units between draws, so
    # recomputing at map_soln asserts only that the two happened to coincide.
    from timex import model, util
    _, ll_index = util.get_max_loglike(tf.trace, model_fn=tf.model_fn)
    edf_by_dataset = model.compute_gp_edf(
        util.get_soln_at(tf.trace, *ll_index),
        tf.data, tf.masks, tf.gp_config)
    assert edf_by_dataset is not None
    assert len(edf_by_dataset) == len(tf.data)
    # abs tolerance, not rel: ic.txt writes edf via '{:.2f}', so up to 0.005
    # is lost on the round trip through the file before we ever compare here.
    # A mean instead of a sum would miss by tens of points, far outside this.
    assert ic['edf'] == pytest.approx(sum(edf_by_dataset.values()), abs=1e-2)


@pytest.mark.slow
def test_non_gp_fit_reports_no_edf_rows(default_fit):
    wd, tf = default_fit
    assert not tf.use_gp
    ic = _read_ic(Path(wd) / 'out')
    assert 'BIC' in ic
    for key in ('edf', 'nparams_edf', 'BIC_edf', 'AIC_edf', 'AICc_edf'):
        assert key not in ic, f'{key} should not appear for a non-GP fit'
