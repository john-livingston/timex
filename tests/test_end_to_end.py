import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')


@pytest.mark.slow
def test_fit_pipeline_runs(tmp_path):
    from timex import fit

    wd = tmp_path / 'hip67522c'
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
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True))

    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)
    tf.save_results()

    outdir = wd / 'out'
    assert (outdir / 'tc.txt').exists()
    assert (outdir / 'ic.txt').exists()

    # {name}_lc_pred is a numpyro deterministic that was recently changed to
    # reuse the already computed transit light curve rather than recompute
    # it. Nothing in the production code reads its value, so this is the
    # only thing that would catch a regression that breaks it (wrong shape,
    # missing entirely, etc). Check every dataset, not just one.
    for name, data in tf.data.items():
        mask = tf.masks[name]
        n_unmasked = len(data['x']) if mask is None else int(np.sum(mask))
        lc_pred_var = f'{name}_lc_pred'
        assert lc_pred_var in tf.trace.posterior.data_vars, (
            f'{lc_pred_var} missing from posterior'
        )
        lc_pred = tf.trace.posterior[lc_pred_var]
        assert lc_pred.shape[-1] == n_unmasked, (
            f'{lc_pred_var} last axis length {lc_pred.shape[-1]} '
            f'!= {n_unmasked} unmasked points'
        )

    # one corrected light curve csv per dataset, not just "at least one"
    cor_files = sorted(outdir.glob('*-cor.csv'))
    assert len(cor_files) == len(tf.data), (
        f'expected {len(tf.data)} *-cor.csv files, found {len(cor_files)}: {cor_files}'
    )

    # the corrected csvs must have real content, not just exist
    cor_df = pd.read_csv(cor_files[0])
    assert len(cor_df) > 0
    assert list(cor_df.columns) == ['x', 'y', 'yerr']
    assert not cor_df.isna().any().any()
