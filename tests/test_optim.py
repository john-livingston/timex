import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from timex import optim


def _tiny_model():
    x = numpyro.sample('x', dist.Normal(0.0, 1.0))
    numpyro.sample('obs', dist.Normal(x, 1.0), obs=np.array(0.5))


def test_optimize_returns_constrained_params():
    result = optim.optimize(_tiny_model, verbose=False, progress=False)
    assert 'x' in result
    assert np.isfinite(result['x'])


def test_optimize_propagates_keyboard_interrupt(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(optim, 'minimize', interrupt)

    with pytest.raises(KeyboardInterrupt):
        optim.optimize(_tiny_model, verbose=False, progress=False)


def test_optimize_raises_when_no_solution(monkeypatch):
    def stop(*args, **kwargs):
        raise StopIteration

    monkeypatch.setattr(optim, 'minimize', stop)

    # StopIteration is still swallowed, but with no evaluated objective there
    # is no usable solution, so this must fail loudly rather than crash deep
    # inside unravel_fn
    with pytest.raises(RuntimeError, match='no valid solution'):
        optim.optimize(_tiny_model, verbose=False, progress=False)
