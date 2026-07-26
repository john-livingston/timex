import logging
import os
import shutil

import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

SHORT = dict(tune=5, draws=5, chains=1, cores=1)


def _load_params(wd):
    with open(os.path.join(wd, 'fit.yaml')) as f:
        fit_params = yaml.safe_load(f)
    with open(os.path.join(wd, 'sys.yaml')) as f:
        sys_params = yaml.safe_load(f)
    fit_params.update(SHORT)
    fit_params['clobber'] = False
    return fit_params, sys_params


def _run(wd, fit_params, sys_params):
    from timex import fit
    tf = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf.build_model(verbose=False, plot=False)
    tf.sample(plot_fit=False, plot_systematics=False)
    return tf


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    """One real fit, copied per test so each gets an isolated directory."""
    wd = tmp_path_factory.mktemp('baseline') / 'hip67522c'
    shutil.copytree(EXAMPLE, wd, ignore=shutil.ignore_patterns('out'))
    fit_params, sys_params = _load_params(wd)
    _run(wd, fit_params, sys_params)
    return wd


@pytest.fixture
def wd(baseline, tmp_path):
    target = tmp_path / 'hip67522c'
    shutil.copytree(baseline, target)
    return target


@pytest.mark.slow
def test_unchanged_rerun_reuses_everything(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    with caplog.at_level(logging.INFO):
        # the module scoped baseline fixture also logs 'sampling for' when it
        # materializes during this test's setup; clear so the assertions below
        # only see the rerun
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'reusing cached MAP solution' in caplog.text
    assert 'sampling for' not in caplog.text


@pytest.mark.slow
def test_config_edit_recomputes_map(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    fit_params['uniform']['ror'] = [0.02, 0.16]
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_draws_bump_reuses_map_and_resamples(wd, caplog):
    """The property the two tier split exists for."""
    fit_params, sys_params = _load_params(wd)
    fit_params['draws'] = SHORT['draws'] + 1
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'reusing cached MAP solution' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_missing_manifest_recomputes_everything(wd, caplog):
    """Every output directory that predates this feature is in this state."""
    from timex import cache
    os.remove(os.path.join(wd, 'out', cache.MANIFEST_NAME))
    fit_params, sys_params = _load_params(wd)
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text
    assert 'sampling for' in caplog.text


@pytest.mark.slow
def test_data_edit_recomputes_map(wd, caplog):
    fit_params, sys_params = _load_params(wd)
    target = wd / fit_params['data']['g']['file']
    # a trailing blank line: guaranteed byte level change, and pandas skips
    # blank lines, so the parsed data is identical and only the hash moves
    target.write_text(target.read_text() + '\n')
    with caplog.at_level(logging.INFO):
        caplog.clear()
        _run(wd, fit_params, sys_params)
    assert 'building and optimizing model' in caplog.text


@pytest.mark.slow
def test_from_dir_loads_mismatched_artifacts_with_a_warning(wd, caplog):
    """from_dir is an explicit request for the artifacts in a directory.

    Loading a finished run for plotting must still work after the config has
    moved on, so a mismatch warns rather than skipping. This is the only
    behavior that distinguishes _force_load_saved, and nothing else covers it.
    """
    from timex import fit

    # edit fit.yaml on disk so from_dir, which re-reads it, sees a changed config
    with open(wd / 'fit.yaml') as f:
        on_disk = yaml.safe_load(f)
    on_disk['uniform']['ror'] = [0.03, 0.17]
    with open(wd / 'fit.yaml', 'w') as f:
        yaml.safe_dump(on_disk, f)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))

    assert 'does not match the current config' in caplog.text
    assert 'loading map.pkl anyway' in caplog.text
    # loaded despite the mismatch, which is the whole point
    assert tf.trace is not None
    assert hasattr(tf, 'map_soln')


@pytest.mark.slow
def test_resumed_run_has_same_posterior_shapes_as_a_fresh_run(wd, caplog):
    """A resumed run must not silently relabel summary.csv.

    The MAP is fed back to init_to_value, so if get_map_soln collapses a site's
    shape the resumed run samples it 0-d and arviz labels it 't0' instead of
    't0[0]'. This is the reuse-MAP-then-resample path, so it is the one that
    would drift.
    """
    fit_params, sys_params = _load_params(wd)
    fresh = set(pd.read_csv(os.path.join(wd, 'out', 'summary.csv'), index_col=0).index)

    fit_params['draws'] = SHORT['draws'] + 1
    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = _run(wd, fit_params, sys_params)
    assert 'reusing cached MAP solution' in caplog.text, 'not exercising the reuse path'

    resumed = set(tf.summary.index)
    assert resumed == fresh, (
        f'resumed run relabelled parameters: only fresh {sorted(fresh - resumed)}, '
        f'only resumed {sorted(resumed - fresh)}'
    )
