"""Cache keys tying resume artifacts to the config and data that produced them.

Imports only the standard library so this logic stays testable without
building a model.
"""
import hashlib
import json
import os

FORMAT_VERSION = 1

# keys that affect sampling only, never the model or the MAP solution
RUN_TIER = frozenset({'tune', 'draws', 'chains'})
# keys that affect neither the model nor the results
NO_EFFECT = frozenset({'cores', 'clobber'})


def _digest(obj):
    """Stable sha256 of a json serializable object, insensitive to dict ordering."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def hash_data_files(fit_params, wd):
    """sha256 of each dataset's file contents, keyed by dataset name.

    Keyed by name rather than by path so relocating a project directory, or
    copying an example into a temporary directory, does not invalidate a cache
    whose data is byte identical. Dataset names are themselves part of
    fit_params, so renaming or adding a dataset still invalidates.
    """
    digests = {}
    for name, spec in fit_params['data'].items():
        h = hashlib.sha256()
        with open(os.path.join(wd, spec['file']), 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        digests[name] = h.hexdigest()
    return digests


def compute_keys(fit_params, sys_params, wd):
    """Return {'model': ..., 'run': ...} for the current config and data.

    Anything not named in RUN_TIER or NO_EFFECT falls into the model tier, so
    an option added later invalidates more rather than less.
    """
    model_params = {k: v for k, v in fit_params.items()
                    if k not in RUN_TIER and k not in NO_EFFECT}
    model_key = _digest({
        'format_version': FORMAT_VERSION,
        'fit': model_params,
        'sys': sys_params,
        'data': hash_data_files(fit_params, wd),
    })
    run_key = _digest({
        'model': model_key,
        'sampler': {k: fit_params.get(k) for k in sorted(RUN_TIER)},
    })
    return {'model': model_key, 'run': run_key}
