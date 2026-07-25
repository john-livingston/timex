import os
import shutil

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')


@pytest.mark.slow
def test_cli_pipeline_runs(tmp_path):
    from timex import fit

    wd = tmp_path / 'hip67522c'
    shutil.copytree(EXAMPLE, wd)

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
    assert list(outdir.glob('*-cor.csv'))
