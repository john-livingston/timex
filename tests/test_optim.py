import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from timex import optim


def _tiny_model():
    x = numpyro.sample('x', dist.Normal(0.0, 1.0))
    numpyro.sample('obs', dist.Normal(x, 1.0), obs=np.array(0.5))


def test_optimize_finds_the_closed_form_map():
    """Pinned against the answer derived by hand, not against the code.

    A Normal(0,1) prior on x with one observation of 0.5 under Normal(x,1)
    gives a Gaussian posterior with mean (0*1 + 0.5*1)/(1+1) = 0.25, so the
    MAP is exactly 0.25. Asserting only that 'x' is present and finite passes
    even when optimize() returns the unoptimized initial point, which is the
    one failure this module exists to prevent.
    """
    result = optim.optimize(_tiny_model, verbose=False, progress=False)
    assert result['x'] == pytest.approx(0.25, abs=1e-3)


def test_optimize_propagates_keyboard_interrupt(monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(optim, 'minimize', interrupt)

    with pytest.raises(KeyboardInterrupt):
        optim.optimize(_tiny_model, verbose=False, progress=False)


def test_optimize_propagates_keyboard_interrupt_after_eval(monkeypatch):
    # the dangerous case: the interrupt lands after the objective has run, so
    # initial_nll is finite and the old code accepted the init point as the MAP
    def interrupt_after_eval(objective, x0, *args, **kwargs):
        objective(x0)
        raise KeyboardInterrupt

    monkeypatch.setattr(optim, 'minimize', interrupt_after_eval)

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
