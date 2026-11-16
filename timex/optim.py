"""
MAP optimization using JAX autodiff and scipy L-BFGS-B
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from numpyro.infer import Predictive
from numpyro.infer.util import initialize_model, unconstrain_fn
import sys
import logging

# A coordinate whose prior spread, in unconstrained space, is below this is
# optimized in units of that spread. Uniform priors (spread near 1.8) and unit
# normal priors (spread 1) stay well clear of it.
NARROW_PRIOR_SCALE = 0.1


def _prior_scaling(model_fn, init_params, n_draws=100):
    """Center and scale for each unconstrained coordinate of a model.

    L-BFGS-B is not scale invariant. A parameter with a very narrow prior,
    such as a period known to 3e-5 days, has a coordinate millions of times
    narrower than the rest, and the optimizer then stalls or its line search
    aborts. Such a coordinate is returned with the median and the standard
    deviation of its prior draws, so it can be optimized as
    center + scale * u.

    Every other coordinate gets center 0 and scale 1, which leaves it as it
    is. Prior spread is only the right scale for a narrow prior: a broad
    prior is usually far wider than the posterior, and stretching the
    coordinate to match it makes the problem worse.

    Args:
        model_fn: The numpyro model function
        init_params: Unconstrained parameter dict from initialize_model
        n_draws: Number of prior draws used for the estimate

    Returns:
        (center, scale): 1-d numpy arrays in the order of
        jax.flatten_util.ravel_pytree(init_params)
    """
    # A fixed key, so the scales do not depend on the restart
    predictive = Predictive(model_fn, num_samples=n_draws,
                            return_sites=list(init_params))
    draws = predictive(jax.random.PRNGKey(0))

    def flat_unconstrained(draw):
        unconstrained = unconstrain_fn(model_fn, (), {}, draw)
        return jax.flatten_util.ravel_pytree(unconstrained)[0]

    # lax.map, not vmap: celerite2 has no batching rule, and one draw at a
    # time keeps memory flat for large datasets
    flat = np.asarray(jax.lax.map(flat_unconstrained, draws))
    spread = np.std(flat, axis=0)
    narrow = np.isfinite(spread) & (spread > 0) & (spread < NARROW_PRIOR_SCALE)
    center = np.where(narrow, np.median(flat, axis=0), 0.0)
    scale = np.where(narrow, spread, 1.0)
    return center, scale


def optimize(
    model_fn,
    maxeval=10000,
    n_restarts=1,
    verbose=True,
    progress=True,
    **kwargs
):
    """Maximize the log prob of a numpyro model using scipy

    Uses JAX autodiff for gradients and numpyro's initialize_model
    to handle parameter transformations (constrained <-> unconstrained).

    Args:
        model_fn: The numpyro model function
        maxeval: Maximum number of function evaluations
        n_restarts: Number of random restarts (best result kept)
        verbose: Print the success flag and log probability to the screen
        progress: Show progress during optimization
    """
    best_x = None
    best_nll = np.inf
    best_info = None
    postprocess_fn = None
    unravel_fn = None
    center = scale = None

    for restart in range(n_restarts):
        rng_key = jax.random.PRNGKey(restart)
        model_info = initialize_model(rng_key, model_fn)
        param_info, potential_fn, pf = model_info[:3]
        postprocess_fn = pf
        init_params = param_info.z
        flat_init, uf = jax.flatten_util.ravel_pytree(init_params)
        unravel_fn = uf

        # The optimizer works on u, with x = center + scale * u. Only
        # coordinates with a narrow prior have a center and scale other than
        # 0 and 1, see _prior_scaling. The scaling is the same for every
        # restart, so it is estimated once.
        if center is None:
            try:
                center, scale = _prior_scaling(model_fn, init_params)
            except Exception as e:
                logging.warning(
                    f"could not draw from the prior to scale the MAP "
                    f"optimization, running unscaled: {e}")
                center = np.zeros(flat_init.size)
                scale = np.ones(flat_init.size)
            center_jax, scale_jax = jnp.array(center), jnp.array(scale)

        if restart == 0 and verbose:
            sys.stderr.write(
                f"optimizing logp for {flat_init.size} parameters\n"
            )

        # JIT-compiled objective and gradient
        def _scaled_potential(u):
            return potential_fn(unravel_fn(center_jax + scale_jax * u))

        _potential = jax.jit(_scaled_potential)
        _grad = jax.jit(jax.grad(_scaled_potential))

        neval = 0
        initial_nll = None

        def objective(x):
            nonlocal neval, initial_nll
            neval += 1
            x_jax = jnp.array(x)
            nll = float(_potential(x_jax))
            grad = np.array(_grad(x_jax))
            if initial_nll is None:
                initial_nll = nll
            if verbose and progress:
                prefix = f"[{restart+1}/{n_restarts}] " if n_restarts > 1 else ""
                sys.stderr.write(f"\r{prefix}Eval {neval}: logp = {-nll:.3e}")
                sys.stderr.flush()
            return nll, grad

        # numpyro's uniform(-2, 2) draw, read as u. An unscaled coordinate
        # starts where it always did. A narrow one starts within two prior
        # widths of its prior median.
        x0 = np.array(flat_init)
        kw = dict(kwargs)
        kw["jac"] = True

        try:
            info = minimize(objective, x0, method='L-BFGS-B',
                            options={'maxiter': maxeval}, **kw)
        except StopIteration:
            # KeyboardInterrupt deliberately propagates: an interrupted run
            # must not fall through and cache the init point as the MAP
            info = None
        finally:
            if verbose and progress:
                sys.stderr.write("\n")

        # Accept if better than initial
        if info is not None and np.isfinite(info.fun) and info.fun < initial_nll:
            x = info.x
            nll = info.fun
        else:
            x = x0
            nll = initial_nll if initial_nll is not None else np.inf

        if verbose and info is not None:
            sys.stderr.write(f"message: {info.message}\n")
            sys.stderr.write(f"logp: {-initial_nll} -> {-info.fun}\n")

        # Track best across restarts
        if nll < best_nll:
            best_nll = nll
            # Back to unconstrained parameter units for the postprocessing
            best_x = center + scale * x
            best_info = info

    if verbose and n_restarts > 1:
        sys.stderr.write(f"best logp across {n_restarts} restarts: {-best_nll:.3e}\n")

    if best_x is None:
        raise RuntimeError(
            "MAP optimization produced no valid solution")

    if not np.isfinite(best_nll):
        sys.stderr.write("WARNING: final logp not finite\n")

    # Convert back to constrained parameter space
    opt_unconstrained = unravel_fn(jnp.array(best_x))
    constrained = postprocess_fn(opt_unconstrained)

    # Convert JAX arrays to numpy
    result = {}
    for k, v in constrained.items():
        result[k] = np.asarray(v)

    return result
