# GP performance

Why `timex` uses `celerite2.jax` for Gaussian process noise models.

## Summary

tinygp and celerite2 implement the same O(N) semiseparable solver, but celerite2's
compiled C++ backend has far lower constant factors. Gradients, which dominate NUTS
sampling cost, are 9 to 18 times faster with celerite2.jax. That is why `timex` uses
`celerite2.jax` rather than tinygp.

celerite2 must be installed from git HEAD, as described in
[Installation](installation.md). The PyPI release (0.3.2) predates the fixes needed
for JAX 0.6 and later.

## Benchmark results

Measured on an Apple M-series CPU with JAX 0.9.2.

### Log-likelihood evaluation (value only)

| N | celerite2 numpy (C) | celerite2.jax (C+XLA) | tinygp (pure JAX) |
|---|---|---|---|
| 100 | 0.006 ms | 0.008 ms | 0.067 ms |
| 500 | 0.009 ms | 0.031 ms | 0.083 ms |
| 1000 | 0.012 ms | 0.044 ms | 0.098 ms |
| 5000 | 0.038 ms | 0.147 ms | 0.232 ms |

### Gradient computation (value and grad via `jax.value_and_grad`)

| N | celerite2.jax | tinygp | Speedup |
|---|---|---|---|
| 100 | 0.027 ms | 0.231 ms | 8.6x |
| 500 | 0.052 ms | 0.655 ms | 12.7x |
| 1000 | 0.084 ms | 1.222 ms | 14.6x |
| 5000 | 0.321 ms | 5.736 ms | 17.9x |

Gradients are the critical metric, since NUTS evaluates value and gradient at every
leapfrog step. The celerite2.jax advantage grows with N.

### End-to-end: timex (numpyro) vs timer (PyMC)

[`timer`](https://github.com/john-livingston/timer) is the PyMC/exoplanet package that
`timex` is a port of. Both runs below use a celerite2 Matern-3/2 kernel on the same
dataset (N of about 140 per band, 4 bands), with 2000 tune and 2000 draws over 2 chains.

| | timer (PyMC) | timex (numpyro) |
|---|---|---|
| Wall time | 125 s | 320 s |
| t0 ESS | 453 | 2754 |
| ror ESS | 3456 | 4253 |
| gp_log_scale ESS | 352 | 732 |

Effective samples per second, where higher is better:

| Parameter | timer ESS/s | timex ESS/s | Winner |
|---|---|---|---|
| t0 | 3.6 | 8.6 | timex 2.4x |
| b | 20.7 | 9.0 | timer 2.3x |
| ror | 27.6 | 13.3 | timer 2.1x |
| gp_log_scale | 2.8 | 2.3 | timer 1.2x |

timer is faster per step, but numpyro's NUTS explores the t0 posterior more
efficiently. For transit timing, where t0 is the parameter of interest, `timex`
produces better samples despite the longer wall time.

### JIT compilation

Compilation is a one-time cost per unique N and is not a bottleneck for long MCMC runs:
0.1 to 0.2 s for a tinygp log-likelihood, and 0.2 to 0.4 s for value and gradient.

## Why tinygp is slower

1. **Pure JAX vs C.** tinygp is implemented entirely in JAX/Python via equinox, while
   celerite2 uses compiled C++ (Eigen) through pybind11. The algorithm is the same
   O(N) semiseparable factorization, but C has much lower per-operation overhead.

2. **Pytree overhead.** tinygp represents kernels as equinox dataclasses, which are JAX
   pytrees. Every GP evaluation traverses that pytree structure, unlike celerite2's flat
   C arrays.

3. **Gradient method.** tinygp relies on reverse-mode automatic differentiation through
   the whole forward pass. celerite2 provides hand-written C reverse-mode kernels
   (`factor_rev`, `solve_lower_rev`, and friends).

4. **Constant factors, not scaling.** Both are O(N). At TESS scale (N of 20k to 100k),
   tinygp would be roughly 4 to 10 times slower per evaluation, and considerably worse
   once gradients are included.

## celerite2 architecture

### Forward pass (C)

- `driver.factor(t, c, a, U, V)` returns `(d, W, S)`, the Cholesky factorization, O(N)
- `driver.solve_lower(t, c, U, W, Y)` returns the solution, O(N)
- `driver.norm(t, c, a, U, V, Y)` returns the log determinant, O(N)

### Reverse pass (C)

- `backprop.factor_rev(t, c, a, U, V, d, W, S, bd, bW)` returns `(bt, bc, ba, bU, bV)`
- `backprop.solve_lower_rev(...)` returns gradients with respect to the inputs
- `backprop.solve_upper_rev(...)` and `backprop.matmul_*_rev(...)` likewise

All reverse functions are compiled C++ exposed through pybind11.

### Kernel parameterization

`Matern32Term(sigma, rho)` maps to the celerite coefficients `(c, a, U, V)` with
`w0 = sqrt(3) / rho` and `S0 = sigma^2 / w0`, with the `a`, `U`, and `V` matrices
derived from the SHO representation using an `eps` regularization parameter.

### JAX layer

The 0.3.2 PyPI release is incompatible with JAX 0.6 and later, where `xla_client.ops`
was removed. Both that and the later removal of `lax.zeros_like_array` in JAX 0.9 are
fixed in git HEAD (see celerite2
[PR #174](https://github.com/exoplanet-dev/celerite2/pull/174)), which is why
[Installation](installation.md) specifies a git install.

### API

The GP interface used by `timex` in `timex/model.py`:

```python
from celerite2.jax import GaussianProcess, terms

kernel = terms.Matern32Term(sigma=amp, rho=scale)
gp = GaussianProcess(kernel)
gp.compute(t, diag=diag)

# inside a numpyro model
numpyro.sample("obs", gp.numpyro_dist(), obs=residuals)

# for predictions, outside the model
gp.predict(residuals)
```
