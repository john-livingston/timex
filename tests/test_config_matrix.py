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
    # seeded so the limb darkening priors and the chain are fixed: unseeded,
    # the edf on the GP fixture ranges over tens of units between runs, which
    # is enough to flip which draw maximizes the likelihood
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True,
                           random_seed=0))
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
def test_non_gp_fit_reports_no_edf_rows(tmp_path):
    wd, fit_params, sys_params = _setup(tmp_path, 'no_edf', lambda p: None)
    tf = _run(wd, fit_params, sys_params)
    tf.save_results()

    assert not tf.use_gp
    ic = _read_ic(wd / 'out')
    assert 'BIC' in ic
    for key in ('edf', 'nparams_edf', 'BIC_edf', 'AIC_edf', 'AICc_edf'):
        assert key not in ic, f'{key} should not appear for a non-GP fit'
