"""Shared short pipeline runs for the slow tests.

Several modules only read a finished fit of the shipped example. Building
that fit is the expensive part, so default, GP, and fixed-t0 each run once
per session. Tests that must rebuild (resume cache misses, two-run seed
checks, unique configs) keep their own copies.
"""
import os
import shutil
from pathlib import Path

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

SHORT = dict(tune=5, draws=5, chains=1, cores=1, clobber=True, random_seed=0)

GP_BLOCK = dict(
    log_amp=-1, log_amp_unc=4, log_amp_prior='uniform',
    log_scale=-1, log_scale_unc=4, log_scale_prior='uniform',
    per_dataset=['log_amp'],
)


def use_gp(fit_params):
    """Mirrors the GP configuration documented in the example's comments."""
    for spec in fit_params['data'].values():
        spec['spline'] = False
        spec['trend'] = 1
    fit_params['use_gp'] = True
    fit_params['gp'] = dict(GP_BLOCK)


def use_fixed_t0(fit_params):
    fit_params['fixed'] = list(fit_params.get('fixed', [])) + ['t0']


def copy_example(dest):
    dest = Path(dest)
    # ignore 'out': the example ships a real ~120MB trace that clobber=True
    # would never read. copying it only slows the setup.
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns('out'))
    return dest


def load_params(wd, **overrides):
    with open(Path(wd) / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(Path(wd) / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(SHORT)
    fit_params.update(overrides)
    return fit_params, sys_params


def run_pipeline(wd, fit_params, sys_params, save=True):
    """Mirrors cli() order: build, clip outliers, sample, optionally save.

    clip_outliers skips datasets that do not set clip: true.
    """
    from timex import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.clip_outliers()
    tf.sample(plot_fit=False, plot_systematics=False)
    if save:
        tf.save_results()
    return tf


def session_fit(tmp_path_factory, name, mutate):
    wd = tmp_path_factory.mktemp(name) / 'hip67522c'
    copy_example(wd)
    fit_params, sys_params = load_params(wd)
    mutate(fit_params)
    tf = run_pipeline(wd, fit_params, sys_params)
    return wd, tf
