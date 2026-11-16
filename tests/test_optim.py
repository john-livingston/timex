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


def _narrow_and_broad_model():
    numpyro.sample('p', dist.Normal(14.334894, 3e-5))
    numpyro.sample('w', dist.Normal(0.0, 1000.0))
    numpyro.sample('r', dist.Uniform(0.01, 0.15))


def test_prior_scaling_rescales_only_narrow_priors():
    """A narrow prior gets its own center and width, everything else is left alone.

    Scaling a broad prior by its spread is the version that broke the
    optimizer, so the broad and the bounded site must keep center 0, scale 1.
    """
    import jax
    from numpyro.infer.util import initialize_model

    z = initialize_model(jax.random.PRNGKey(0), _narrow_and_broad_model)[0].z
    center, scale = optim._prior_scaling(_narrow_and_broad_model, z)
    names = sorted(z)  # ravel_pytree flattens dict keys in sorted order
    got = {k: (center[i], scale[i]) for i, k in enumerate(names)}
    assert got['p'][0] == pytest.approx(14.334894, abs=1e-4)
    assert got['p'][1] == pytest.approx(3e-5, rel=0.3)
    assert got['w'] == (0.0, 1.0)
    assert got['r'] == (0.0, 1.0)
