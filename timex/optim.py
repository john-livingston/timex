"""
MAP optimization using JAX autodiff and scipy L-BFGS-B
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from numpyro.infer.util import initialize_model
import sys
import logging


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

    for restart in range(n_restarts):
        rng_key = jax.random.PRNGKey(restart)
        model_info = initialize_model(rng_key, model_fn)
        param_info, potential_fn, pf = model_info[:3]
        postprocess_fn = pf
        init_params = param_info.z
        flat_init, uf = jax.flatten_util.ravel_pytree(init_params)
        unravel_fn = uf

        if restart == 0 and verbose:
            sys.stderr.write(
                f"optimizing logp for {flat_init.size} parameters\n"
            )

        # JIT-compiled objective and gradient
        @jax.jit
        def _potential(x):
            return potential_fn(unravel_fn(x))

        @jax.jit
        def _grad(x):
            return jax.grad(lambda x: potential_fn(unravel_fn(x)))(x)

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

        x0 = np.array(flat_init)
        kw = dict(kwargs)
        kw["jac"] = True

        try:
            info = minimize(objective, x0, method='L-BFGS-B',
                            options={'maxiter': maxeval}, **kw)
        except (KeyboardInterrupt, StopIteration):
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
            best_x = x
            best_info = info

    if verbose and n_restarts > 1:
        sys.stderr.write(f"best logp across {n_restarts} restarts: {-best_nll:.3e}\n")

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
