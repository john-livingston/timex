import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO_ROOT, 'examples', 'hip67522c')

SHORT = dict(tune=5, draws=5, chains=1, cores=1)


def _bare_fit(tmp_path, stale):
    """A TransitFit carrying only what build_model reads, with a cached MAP.

    Same construction shortcut as
    test_fit_defaults.py::test_get_ic_counts_only_unmasked_points: a real
    TransitFit needs data files and priors, and none of that is what these
    tests are about.
    """
    from timex import fit

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.outdir = str(tmp_path)
    tf.clobber = False
    tf._stale_force_loaded = set(stale)
    tf._cache_keys = {'model': 'MODELKEY', 'run': 'RUNKEY'}
    tf.map_soln = {'t0': np.array([0.05]), 'cached': np.array(1.0)}
    tf.data, tf.priors, tf.masks = {}, {}, {}
    tf.nplanets, tf.use_gp, tf.chromatic = 1, False, False
    tf.fixed, tf.fit_basis = [], 'duration'
    tf.include_mean = True
    tf.include_flare = tf.chromatic_flare = False
    tf.include_bump = tf.chromatic_bump = False
    tf.use_custom_optimizer = True
    tf.gp_config = None
    tf.n_restarts = 1
    return tf


def _stub_build(monkeypatch, captured):
    from timex import fit

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return object(), {'t0': np.array([0.07]), 'fresh': np.array(1.0)}

    monkeypatch.setattr(fit.model, 'build', fake_build)


def _add_clippable_dataset(tf, name='g', outlier=True):
    """Give a bare fit one dataset, holding one 7-sigma outlier or none.

    The clipping itself stays real: util.get_outlier_mask is plain numpy over
    y minus the model, so nineteen points at 1e-3 set rms=1e-3 and the single
    point at 1.0 is the only one beyond 7 rms.
    """
    n = 20
    y = np.full(n, 1e-3)
    if outlier:
        y[-1] = 1.0
    tf.data[name] = dict(x=np.arange(n, dtype=float), y=y)
    tf.masks[name] = None
    tf.map_soln[f'{name}_light_curves'] = np.zeros(n)
    tf.fit_params = {'data': {name: dict(clip=True, clip_nsig=7)}}
    return tf


def test_build_model_does_not_reuse_a_stale_force_loaded_map(tmp_path, monkeypatch):
    """A map.pkl force-loaded past a key mismatch belongs to another config.

    It is present, so the plain hasattr check reuses it and skips optimization;
    the run then samples from a MAP that the current config never produced.
    """
    from timex import cache

    captured = {}
    _stub_build(monkeypatch, captured)
    tf = _bare_fit(tmp_path, stale={'map.pkl'})

    tf.build_model(plot=False)

    assert captured['optimize'] is True, 'stale MAP was reused instead of re-optimized'
    assert 'fresh' in tf.map_soln, 'the re-optimized solution did not replace the stale one'
    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'map.pkl' not in manifest, (
        'a MAP written during a session that force-loaded a stale one must not '
        'be vouched for under the current key'
    )


def test_build_model_still_reuses_a_matching_cached_map(tmp_path, monkeypatch):
    """The control for the test above: without the mismatch, reuse is the point."""
    from timex import cache

    captured = {}
    _stub_build(monkeypatch, captured)
    tf = _bare_fit(tmp_path, stale=set())

    tf.build_model(plot=False)

    assert captured['optimize'] is False, 'a valid cached MAP must not be re-optimized'
    assert 'cached' in tf.map_soln
    assert cache.read_manifest(str(tmp_path)) is None, (
        'reusing a cached MAP writes nothing, so there is nothing to record'
    )


def test_a_stale_mask_disqualifies_the_map_built_on_it(tmp_path, monkeypatch):
    """A stale artifact taints everything downstream of it, not just itself.

    mask.pkl feeds model.build, the likelihood and the log_sigma_lc priors, so
    a MAP optimized against a mask the current config no longer produces is no
    more trustworthy than a stale map.pkl. This is what a deleted map.pkl, or
    a run that died before writing one, leaves behind.
    """
    from timex import cache, fit

    captured = {}
    _stub_build(monkeypatch, captured)
    tf = _bare_fit(tmp_path, stale={'mask.pkl'})
    del tf.map_soln    # map.pkl absent, so this build really does optimize

    tf.build_model(plot=False)

    assert captured['optimize'] is True, 'setup: the MAP must actually be recomputed'
    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'map.pkl' not in manifest, (
        'a MAP optimized against a stale mask was recorded under the current key'
    )


def test_a_stale_map_disqualifies_the_mask_clipped_with_it(tmp_path):
    """The same property in the other direction.

    get_outlier_mask subtracts the MAP model from the data, so a mask clipped
    with a stale MAP describes the wrong outliers. Recording it tells the next
    ordinary run that clipping is already settled, and the outliers the right
    MAP would have found are never looked for.
    """
    from timex import cache, fit

    tf = _add_clippable_dataset(_bare_fit(tmp_path, stale={'map.pkl'}), outlier=False)

    tf.clip_outliers()

    manifest = cache.read_manifest(str(tmp_path)) or {}
    assert 'mask.pkl' not in manifest, (
        'a mask clipped with a stale MAP was recorded under the current key'
    )


def test_a_refit_that_dies_leaves_no_entry_vouching_for_the_stale_map(tmp_path, monkeypatch):
    """clip_outliers records the new mask, then the refit on it never finishes.

    Ctrl-C in the optimizer is a deliberate live exit path; an OOM or a
    scheduler timeout does the same. Nothing distinguishes the pre-clip MAP
    from the post-clip one by key, so if map.pkl's entry survives the
    interrupted rebuild, mask.pkl and map.pkl both read as valid under the
    same model key while describing different maskings. The next run then
    loads both, skips clipping because the mask is not None, and samples a
    likelihood that disagrees with _count_data.
    """
    from timex import cache, fit

    tf = _add_clippable_dataset(_bare_fit(tmp_path, stale=set()))
    # the pre-clip build left a MAP recorded under the current model key
    cache.write_manifest(str(tmp_path), 'map.pkl', 'MODELKEY')

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fit.model, 'build', interrupted)

    with pytest.raises(KeyboardInterrupt):
        tf.clip_outliers()

    manifest = cache.read_manifest(str(tmp_path))
    assert manifest['mask.pkl'] == 'MODELKEY', (
        'setup: clip_outliers must have recorded the post-clip mask'
    )
    assert 'map.pkl' not in manifest, (
        'map.pkl still vouches for the pre-clip masking under the same key '
        'as the post-clip mask.pkl'
    )


def test_sampling_that_dies_leaves_no_entry_vouching_for_the_previous_trace(tmp_path, monkeypatch):
    """clobber recomputes the mask, then MCMC never finishes.

    A clobber run re-optimizes, so the mask it clips can differ from the one
    the trace on disk was sampled under. MCMC is the step that runs for hours
    and therefore the one that gets interrupted. If trace.nc's entry survives
    that, the next run loads the old trace against the new mask, skips MCMC,
    and reports a summary and IC whose ndata never entered that posterior.
    """
    from timex import cache, fit

    tf = _bare_fit(tmp_path, stale=set())
    tf.clobber = True
    tf.trace = None
    tf.model_fn = object()
    tf.tune = tf.draws = tf.chains = tf.cores = 1
    # a finished run left a trace recorded under the current run key
    cache.write_manifest(str(tmp_path), 'trace.nc', 'RUNKEY')

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(fit.model, 'sample', interrupted)

    with pytest.raises(KeyboardInterrupt):
        tf.sample(plot_fit=False, plot_systematics=False)

    manifest = cache.read_manifest(str(tmp_path))
    assert 'trace.nc' not in manifest, (
        'the trace on disk predates this run and no longer matches the mask, '
        'but the manifest still vouches for it'
    )


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


def _write_fit_yaml(wd, fit_params):
    """from_dir re-reads fit.yaml, so an edit only counts once it is on disk.

    The baseline runs with SHORT applied in memory while the example's own
    fit.yaml still asks for thousands of draws, so tests that go on to sample
    through from_dir must write the shortened config out first.
    """
    with open(os.path.join(wd, 'fit.yaml'), 'w') as f:
        yaml.safe_dump(fit_params, f)


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
def test_from_dir_clip_does_not_launder_stale_mask_under_current_key(wd, caplog):
    """The write side must respect a force-loaded mismatch too.

    from_dir force-loads a mismatched mask.pkl so a saved run can still be
    inspected, and clip_outliers then does not recompute it (masks[name] is
    not None). If clip_outliers went on to record it under the CURRENT model
    key, a later CLI run would silently adopt an old-config mask with no
    warning. The safe outcome is that dropping the stale entry first, then
    skipping the re-record, leaves no entry at all, so the next run recomputes.
    """
    from timex import cache, fit

    # produce a mask.pkl recorded under the ORIGINAL (matching) model key
    fit_params, sys_params = _load_params(wd)
    tf0 = fit.TransitFit(sys_params, fit_params, wd=str(wd))
    tf0.clip_outliers()
    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert 'mask.pkl' in manifest, 'setup: clip_outliers must record mask.pkl before the edit'

    # edit fit.yaml on disk so from_dir, which re-reads it, sees a changed config
    with open(wd / 'fit.yaml') as f:
        on_disk = yaml.safe_load(f)
    on_disk['uniform']['ror'] = [0.03, 0.17]
    with open(wd / 'fit.yaml', 'w') as f:
        yaml.safe_dump(on_disk, f)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.clip_outliers()

    assert 'does not match the current config' in caplog.text
    assert 'loading mask.pkl anyway' in caplog.text

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest.get('mask.pkl') != tf._cache_keys['model'], (
        'mask.pkl was recorded under the current key despite being '
        'force-loaded from a mismatched config'
    )
    assert 'mask.pkl' not in manifest, 'the safe outcome is no entry, so the next run recomputes'


@pytest.mark.slow
def test_from_dir_run_tier_edit_resamples_and_records_nothing_derived(wd, caplog):
    """A stale trace must be treated as absent, not silently resumed from.

    A finished run exists at the baseline's draws. Bumping draws only changes
    the run key, so from_dir force-loads the mismatched trace.nc with a
    warning. If sample() then trusts self.trace it skips MCMC entirely and
    rederives summary.csv and map.pkl from the old draws, and because map.pkl
    itself was never flagged it gets recorded under the current key, so the
    next ordinary run adopts it with no warning at all.
    """
    from timex import cache, fit

    fit_params, sys_params = _load_params(wd)
    fit_params['draws'] = SHORT['draws'] + 1
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.build_model(verbose=False, plot=False)
        tf.sample(plot_fit=False, plot_systematics=False)

    assert 'loading trace.nc anyway' in caplog.text, 'setup: the trace must be force-loaded stale'
    assert 'reusing cached MAP solution' in caplog.text, (
        'setup: only the run key may have moved, so the MAP is still valid'
    )
    assert 'sampling for' in caplog.text, 'MCMC was skipped on a stale trace'

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest.get('trace.nc') != tf._cache_keys['run']
    assert 'trace.nc' not in manifest, 'a force-loaded stale trace was re-recorded as valid'
    assert 'map.pkl' not in manifest, (
        'map.pkl is derived from the trace, so it may not be recorded either'
    )


@pytest.mark.slow
def test_from_dir_build_model_does_not_launder_stale_map_under_current_key(wd, caplog):
    """The build_model half of the same property, on the model tier.

    A model-tier edit invalidates map.pkl; from_dir loads it anyway. Reusing
    it would sample from a MAP the current config never produced, and the
    write site would then have nothing to guard.
    """
    from timex import cache, fit

    fit_params, sys_params = _load_params(wd)
    fit_params['uniform']['ror'] = [0.03, 0.17]
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.build_model(verbose=False, plot=False)

    assert 'loading map.pkl anyway' in caplog.text, 'setup: the MAP must be force-loaded stale'
    assert 'building and optimizing model' in caplog.text, 'a stale MAP was reused'

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest.get('map.pkl') != tf._cache_keys['model']
    assert 'map.pkl' not in manifest, 'the safe outcome is no entry, so the next run recomputes'


@pytest.mark.slow
def test_from_dir_sample_does_not_launder_stale_map_when_the_trace_is_fresh(wd, caplog):
    """Isolates the map.pkl guard at the end of sample from the trace's.

    Removing trace.nc leaves nothing to force-load on the run tier, so MCMC
    runs from scratch and its trace is recorded normally. Only map.pkl is
    flagged, so it is the one artifact that must not be recorded.
    """
    from timex import cache, fit

    os.remove(os.path.join(wd, 'out', 'trace.nc'))
    fit_params, sys_params = _load_params(wd)
    fit_params['uniform']['ror'] = [0.03, 0.17]
    _write_fit_yaml(wd, fit_params)

    with caplog.at_level(logging.INFO):
        caplog.clear()
        tf = fit.TransitFit.from_dir(str(wd))
        tf.build_model(verbose=False, plot=False)
        tf.sample(plot_fit=False, plot_systematics=False)

    assert 'loading map.pkl anyway' in caplog.text, 'setup: the MAP must be force-loaded stale'
    assert 'loading trace' not in caplog.text, 'setup: there must be no trace to force-load'

    manifest = cache.read_manifest(os.path.join(wd, 'out'))
    assert manifest.get('trace.nc') == tf._cache_keys['run'], (
        'the freshly sampled trace is valid, so this test really does isolate map.pkl'
    )
    assert 'map.pkl' not in manifest, 'a force-loaded stale MAP was re-recorded as valid'


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
