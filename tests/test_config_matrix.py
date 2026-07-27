import os
import shutil

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

GP_BLOCK = dict(
    log_amp=-1, log_amp_unc=4, log_amp_prior='uniform',
    log_scale=-1, log_scale_unc=4, log_scale_prior='uniform',
    per_dataset=['log_amp'],
)


def _use_gp(fit_params):
    """Mirrors the GP configuration documented in the example's own comments."""
    for spec in fit_params['data'].values():
        spec['spline'] = False
        spec['trend'] = 1
    fit_params['use_gp'] = True
    fit_params['gp'] = dict(GP_BLOCK)


def _gp_only(fit_params):
    _use_gp(fit_params)


def _gp_no_mean(fit_params):
    _use_gp(fit_params)
    fit_params['include_mean'] = False


def _clip_no_mean(fit_params):
    fit_params['include_mean'] = False
    for spec in fit_params['data'].values():
        spec['clip'] = True
        spec['clip_nsig'] = 3


def _fixed_t0(fit_params):
    fit_params['fixed'] = list(fit_params.get('fixed', [])) + ['t0']


def _setup(tmp_path, name, mutate):
    """Copy the example, shorten the sampler, apply `mutate`, return inputs."""
    wd = tmp_path / name
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))
    with open(wd / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(wd / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True))
    mutate(fit_params)
    return wd, fit_params, sys_params


def _run(wd, fit_params, sys_params):
    """Mirrors cli()'s pipeline order: build, clip outliers, then sample.

    clip_outliers is safe to call unconditionally: it skips any dataset
    whose config lacks clip: true, so this only changes behavior for
    matrix entries that actually enable clipping.
    """
    from timex import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.clip_outliers()
    tf.sample(plot_fit=False, plot_systematics=False)
    return tf


@pytest.mark.slow
@pytest.mark.parametrize('name,mutate', [
    ('gp', _gp_only),
    ('gp_no_mean', _gp_no_mean),
    ('clip_no_mean', _clip_no_mean),
    ('fixed_t0', _fixed_t0),
])
def test_pipeline_runs_for_config(tmp_path, name, mutate):
    wd, fit_params, sys_params = _setup(tmp_path, name, mutate)
    tf = _run(wd, fit_params, sys_params)
    tf.save_results()

    outdir = wd / 'out'
    assert (outdir / 'tc.txt').exists()
    assert (outdir / 'ic.txt').exists()
    assert len(sorted(outdir.glob('*-cor.csv'))) == len(tf.data)


@pytest.mark.slow
def test_fixed_t0_reports_zero_uncertainty(tmp_path):
    """t0 held fixed must still produce tc.txt, with zero uncertainty."""
    wd, fit_params, sys_params = _setup(tmp_path, 'fixed_t0_tc', _fixed_t0)
    tf = _run(wd, fit_params, sys_params)
    tf.save_results()

    line = (wd / 'out' / 'tc.txt').read_text().split()
    assert line[0] == 'c'
    assert float(line[2]) == 0.0


@pytest.mark.slow
def test_gp_ic_excludes_gp_predictions(tmp_path):
    """A GP fit must not count its per point GP prediction as free parameters.

    Before this was fixed, nparams was inflated by roughly ndata, which
    reversed the GP versus no-GP verdict on real data.
    """
    from timex import util

    wd, fit_params, sys_params = _setup(tmp_path, 'gp_ic', _use_gp)
    tf = _run(wd, fit_params, sys_params)

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
def test_gp_fit_reports_edf_corrected_ic(tmp_path):
    """The GP is charged for a handful of hyperparameters but absorbs far more
    degrees of freedom, so the corrected criteria must be reported alongside."""
    wd, fit_params, sys_params = _setup(tmp_path, 'gp_edf', _use_gp)
    tf = _run(wd, fit_params, sys_params)
    tf.save_results()

    ic = _read_ic(wd / 'out')
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


@pytest.mark.slow
def test_non_gp_fit_reports_no_edf_rows(tmp_path):
    wd, fit_params, sys_params = _setup(tmp_path, 'no_edf', lambda p: None)
    tf = _run(wd, fit_params, sys_params)
    tf.save_results()

    ic = _read_ic(wd / 'out')
    assert 'BIC' in ic
    for key in ('edf', 'nparams_edf', 'BIC_edf', 'AIC_edf', 'AICc_edf'):
        assert key not in ic, f'{key} should not appear for a non-GP fit'
