import copy

import pytest

from timex import cache


def _fit_params(tmp_path, contents='time,flux,fluxerr\n1.0,1.0,0.001\n'):
    fp = tmp_path / 'a.csv'
    fp.write_text(contents)
    return dict(
        data={'g': dict(file='a.csv', band='g')},
        planets='c',
        tc_pred=2460423.03,
        fixed=['period'],
        tune=2000, draws=2000, chains=2, cores=2, clobber=False,
        n_restarts=1,
    )


def test_keys_are_stable_across_dict_ordering(tmp_path):
    fp = _fit_params(tmp_path)
    reordered = dict(reversed(list(fp.items())))
    sys_params = {'m_star': 1.0, 'r_star': 1.0}
    a = cache.compute_keys(fp, sys_params, str(tmp_path))
    b = cache.compute_keys(reordered, dict(reversed(list(sys_params.items()))), str(tmp_path))
    assert a == b


def test_config_edit_changes_model_key(tmp_path):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited['fixed'] = ['period', 't0']
    assert cache.compute_keys(edited, {}, str(tmp_path))['model'] != base['model']


def test_sys_params_edit_changes_model_key(tmp_path):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {'m_star': 1.0}, str(tmp_path))
    assert cache.compute_keys(fp, {'m_star': 1.1}, str(tmp_path))['model'] != base['model']


def test_data_edit_changes_model_key(tmp_path):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    (tmp_path / 'a.csv').write_text('time,flux,fluxerr\n1.0,2.0,0.001\n')
    assert cache.compute_keys(fp, {}, str(tmp_path))['model'] != base['model']


def test_format_version_bump_changes_model_key(tmp_path, monkeypatch):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    monkeypatch.setattr(cache, 'FORMAT_VERSION', cache.FORMAT_VERSION + 1)
    assert cache.compute_keys(fp, {}, str(tmp_path))['model'] != base['model']


@pytest.mark.parametrize('key', ['tune', 'draws', 'chains'])
def test_sampler_change_moves_run_key_only(tmp_path, key):
    """The property that motivates two tiers: bumping draws must not discard the MAP."""
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited[key] = fp[key] + 1
    new = cache.compute_keys(edited, {}, str(tmp_path))
    assert new['model'] == base['model']
    assert new['run'] != base['run']


@pytest.mark.parametrize('key,value', [('cores', 8), ('clobber', True)])
def test_no_effect_keys_change_neither_key(tmp_path, key, value):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited[key] = value
    assert cache.compute_keys(edited, {}, str(tmp_path)) == base


def test_unknown_key_lands_in_model_tier(tmp_path):
    """A future option nobody classified must invalidate more, never less."""
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited['some_future_option'] = True
    assert cache.compute_keys(edited, {}, str(tmp_path))['model'] != base['model']


def test_random_seed_lands_in_model_tier(tmp_path):
    """random_seed reaches the limb darkening priors, so it changes the model
    and the MAP, not only the chain. Keeping it out of RUN_TIER is what stops a
    reseeded rerun from reusing a map.pkl fitted under the previous priors."""
    fp = _fit_params(tmp_path)
    fp['random_seed'] = None
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited['random_seed'] = 7
    new = cache.compute_keys(edited, {}, str(tmp_path))
    assert new['model'] != base['model']
    assert new['run'] != base['run']


def test_data_hashed_by_content_not_path(tmp_path):
    """Copying a project directory must not invalidate a byte identical cache."""
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    other = tmp_path / 'elsewhere'
    other.mkdir()
    (other / 'a.csv').write_text((tmp_path / 'a.csv').read_text())
    assert cache.compute_keys(fp, {}, str(other)) == base


def test_dataset_rename_changes_model_key(tmp_path):
    fp = _fit_params(tmp_path)
    base = cache.compute_keys(fp, {}, str(tmp_path))
    edited = copy.deepcopy(fp)
    edited['data'] = {'r': edited['data']['g']}
    assert cache.compute_keys(edited, {}, str(tmp_path))['model'] != base['model']


def test_read_manifest_missing_returns_none(tmp_path):
    assert cache.read_manifest(str(tmp_path)) is None


def test_read_manifest_malformed_returns_none(tmp_path):
    (tmp_path / cache.MANIFEST_NAME).write_text('{not json')
    assert cache.read_manifest(str(tmp_path)) is None


def test_read_manifest_wrong_format_version_returns_none(tmp_path):
    """The version field is the lever for invalidating every cache in the wild."""
    (tmp_path / cache.MANIFEST_NAME).write_text(
        '{"format_version": 999, "map.pkl": "abc"}'
    )
    assert cache.read_manifest(str(tmp_path)) is None


def test_write_then_read_roundtrip(tmp_path):
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    manifest = cache.read_manifest(str(tmp_path))
    assert manifest['map.pkl'] == 'abc'
    assert manifest['format_version'] == cache.FORMAT_VERSION


def test_write_manifest_preserves_other_entries(tmp_path):
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    cache.write_manifest(str(tmp_path), 'trace.nc', 'def')
    manifest = cache.read_manifest(str(tmp_path))
    assert manifest['map.pkl'] == 'abc'
    assert manifest['trace.nc'] == 'def'


def test_is_valid_matches_only_on_exact_key(tmp_path):
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    manifest = cache.read_manifest(str(tmp_path))
    assert cache.is_valid(manifest, 'map.pkl', 'abc')
    assert not cache.is_valid(manifest, 'map.pkl', 'xyz')
    assert not cache.is_valid(manifest, 'trace.nc', 'abc')


def test_is_valid_false_for_missing_manifest():
    assert not cache.is_valid(None, 'map.pkl', 'abc')


def test_drop_entry_removes_only_that_artifact(tmp_path):
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    cache.write_manifest(str(tmp_path), 'trace.nc', 'def')
    cache.drop_entry(str(tmp_path), 'map.pkl')
    manifest = cache.read_manifest(str(tmp_path))
    assert 'map.pkl' not in manifest
    assert manifest['trace.nc'] == 'def'
    assert manifest['format_version'] == cache.FORMAT_VERSION


def test_drop_entry_is_a_noop_when_nothing_to_drop(tmp_path):
    cache.drop_entry(str(tmp_path), 'map.pkl')          # no manifest at all
    cache.write_manifest(str(tmp_path), 'trace.nc', 'def')
    cache.drop_entry(str(tmp_path), 'map.pkl')          # manifest lacks the entry
    assert cache.read_manifest(str(tmp_path))['trace.nc'] == 'def'


def _raise_on_dump(monkeypatch):
    """Simulate dying partway through serializing the manifest."""
    def boom(*args, **kwargs):
        raise RuntimeError('crash')

    monkeypatch.setattr(cache.json, 'dump', boom)


def test_write_manifest_leaves_the_previous_manifest_intact_when_it_dies(tmp_path, monkeypatch):
    """Opening the manifest in place truncates it before anything is written.

    A crash in that window costs every entry, including a trace.nc that took
    hours, and this module exists to survive crashes.
    """
    cache.write_manifest(str(tmp_path), 'trace.nc', 'def')
    _raise_on_dump(monkeypatch)

    with pytest.raises(RuntimeError):
        cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')

    assert cache.read_manifest(str(tmp_path)) == {
        'format_version': cache.FORMAT_VERSION, 'trace.nc': 'def'}


def test_drop_entry_leaves_the_previous_manifest_intact_when_it_dies(tmp_path, monkeypatch):
    """drop_entry rewrites the whole manifest too, so it has the same window."""
    cache.write_manifest(str(tmp_path), 'trace.nc', 'def')
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    _raise_on_dump(monkeypatch)

    with pytest.raises(RuntimeError):
        cache.drop_entry(str(tmp_path), 'map.pkl')

    assert cache.read_manifest(str(tmp_path))['trace.nc'] == 'def'


def test_dropped_entry_no_longer_validates(tmp_path):
    """The point of dropping: a half written artifact must read as stale."""
    cache.write_manifest(str(tmp_path), 'map.pkl', 'abc')
    cache.drop_entry(str(tmp_path), 'map.pkl')
    assert not cache.is_valid(cache.read_manifest(str(tmp_path)), 'map.pkl', 'abc')
