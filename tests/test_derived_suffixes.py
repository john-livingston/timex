import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PY = os.path.join(REPO_ROOT, 'timex', 'model.py')

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
@pytest.mark.parametrize('fit_fixture', ['default_fit', 'gp_fit'])
def test_every_posterior_var_is_classified(fit_fixture, request):
    _, tf = request.getfixturevalue(fit_fixture)

    # parametrized over both the default and a GP config, so a site that
    # only ever exists on the GP path (gp_log_amp/gp_log_scale, gp_pred, a
    # future GP-only deterministic, ...) is exercised here too, not just the
    # default non-GP run.
    unclassified = [v for v in tf.trace.posterior.data_vars if not _is_classified(v)]
    assert not unclassified, (
        f'unclassified posterior variables: {unclassified}. If these are '
        'deterministics, add their suffix to util.DERIVED_SUFFIXES or they '
        'will be counted as free parameters in BIC/AIC. If they are genuine '
        'free parameters, add them to this test.'
    )


def _derived_suffixes_from_model_source():
    """Every derived-quantity suffix that timex/model.py actually creates.

    Read straight from the source text rather than from
    util.DERIVED_SUFFIXES, so the "should be excluded" set can't silently
    shrink along with the tuple it exists to check. Derived quantities
    appear in model.py in exactly two forms: a
    numpyro.deterministic(f"{name}_XXX", ...) site, or the one post-hoc
    site, map_soln[f'{name}_gp_pred'], assigned in _add_gp_predictions. A
    regex over the file finds every site without building a model, keeping
    this test fast.
    """
    with open(MODEL_PY) as f:
        src = f.read()
    suffixes = set(re.findall(
        r'numpyro\.deterministic\(\s*f["\']\{name\}(_\w+)["\']', src))
    suffixes.update(re.findall(
        r'map_soln\[f["\']\{name\}(_\w+)["\']\]\s*=(?!=)', src))
    return suffixes


def test_derived_suffixes_tuple_covers_every_model_py_site():
    """util.DERIVED_SUFFIXES must cover every derived site model.py creates.

    A fixture built by iterating util.DERIVED_SUFFIXES itself (the previous
    version of this test) can only confirm that count_free_params excludes
    whatever the tuple currently holds; it can never notice the tuple being
    incomplete. This derives the "must be excluded" set independently, from
    model.py's source, instead. This is the exact historical bug: nparams
    was inflated from 41 to 601 against 560 data points because _gp_pred
    was missing from the tuple, reversing a GP versus no-GP model
    comparison.
    """
    from timex import util
    for suffix in sorted(_derived_suffixes_from_model_source()):
        assert suffix.endswith(util.DERIVED_SUFFIXES), (
            f'{suffix!r} is a derived quantity created in timex/model.py '
            'but util.DERIVED_SUFFIXES does not cover it. Add it there, or '
            'it will be double counted as a free parameter, inflating '
            'BIC and AIC.'
        )
