import os
import shutil

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

# every scalar or vector site that is a genuine free parameter, by exact name
# or by prefix. anything in the posterior matching neither this nor
# DERIVED_SUFFIXES is unclassified and would be counted in BIC.
FREE_EXACT = {'t0', 'period', 'ror', 'b', 'dur'}
FREE_PREFIXES = ('u_star_', 'gp_log_amp', 'gp_log_scale')
FREE_SUFFIXES = ('_mean', '_log_sigma_lc', '_weights')


def _is_classified(name):
    from timex import util
    if name in FREE_EXACT:
        return True
    if name.startswith(FREE_PREFIXES):
        return True
    if name.endswith(FREE_SUFFIXES):
        return True
    return name.endswith(util.DERIVED_SUFFIXES)


@pytest.mark.slow
def test_every_posterior_var_is_classified(tmp_path):
    from timex import fit

    wd = tmp_path / 'suffixes'
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))
    with open(wd / 'fit.yaml') as f:
        fit_params = yaml.safe_load(f)
    with open(wd / 'sys.yaml') as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(dict(tune=5, draws=5, chains=1, cores=1, clobber=True))

    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)

    # this run is deliberately the default (non GP) config: it is the one that
    # must stay classified as the model grows. GP sites are covered by the
    # gp_log_amp/gp_log_scale prefixes for when this is extended.
    unclassified = [v for v in tf.trace.posterior.data_vars if not _is_classified(v)]
    assert not unclassified, (
        f'unclassified posterior variables: {unclassified}. If these are '
        'deterministics, add their suffix to util.DERIVED_SUFFIXES or they '
        'will be counted as free parameters in BIC/AIC. If they are genuine '
        'free parameters, add them to this test.'
    )


def test_count_free_params_excludes_every_derived_suffix():
    """Direct unit check, no sampling required."""
    from timex import util
    soln = {'t0': np.array(0.05)}
    for suffix in util.DERIVED_SUFFIXES:
        soln[f'g{suffix}'] = np.zeros(500)
    assert util.count_free_params(soln) == 1
