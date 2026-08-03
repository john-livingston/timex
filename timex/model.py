import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.light_curves.transforms import integrate
from celerite2.jax import GaussianProcess as CeleriteGP, terms as celerite_terms
import logging
from . import optim


def bump_model(t, t_center, width, amplitude):
    """
    Model a "bump" in a light curve using a simple Gaussian profile.
    Used to model phenomena like spot-crossing during a transit.
    """
    return amplitude * jnp.exp(-(t - t_center)**2 / (2 * width**2))


def aflare1(t, tpeak, fwhm, ampl):
    """
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    x = (t - tpeak) / fwhm
    f1 = _fr[0] + _fr[1] * x + _fr[2] * x**2 + _fr[3] * x**3 + _fr[4] * x**4
    f2 = _fd[0] * jnp.exp(x * _fd[1]) + _fd[2] * jnp.exp(x * _fd[3])
    part1 = jnp.where((t > tpeak - fwhm) & (t <= tpeak), f1, 0.0)
    part2 = jnp.where(t > tpeak, f2, 0.0)
    flare = part1 + part2

    return flare * ampl


def get_rv(key=None, priors=None, prior_dist=None, shape=None, name=None, bounded=None,
           mu=None, sd=None, lower=None, upper=None, verbose=False, bounds=None):
    if priors is not None:
        prior_dist = priors[f'{key}_prior']
    if name is None:
        name = key
    if prior_dist == 'gaussian':
        if priors is not None:
            mu, sd = priors[key], priors[f'{key}_unc']
        mu = jnp.atleast_1d(jnp.array(mu, dtype=jnp.float64))
        sd = jnp.atleast_1d(jnp.array(sd, dtype=jnp.float64))
        if bounds is not None:
            lower_bound, upper_bound = bounds
            rv = numpyro.sample(name, dist.TruncatedNormal(
                loc=mu, scale=sd, low=lower_bound, high=upper_bound))
        else:
            rv = numpyro.sample(name, dist.Normal(mu, sd))
        if shape is not None and shape == 1:
            rv = rv.squeeze()
        spec = f'{prior_dist}({mu},{sd})'
    elif prior_dist == 'uniform':
        if priors is not None:
            lower = priors[key] - priors[f'{key}_unc']/2
            upper = priors[key] + priors[f'{key}_unc']/2
        lower = jnp.atleast_1d(jnp.array(lower, dtype=jnp.float64))
        upper = jnp.atleast_1d(jnp.array(upper, dtype=jnp.float64))
        if bounded is not None:
            lower = jnp.where(lower < bounded[0], bounded[0], lower)
            upper = jnp.where(upper > bounded[1], bounded[1], upper)
        rv = numpyro.sample(name, dist.Uniform(lower, upper))
        if shape is not None and shape == 1:
            rv = rv.squeeze()
        spec = f'{prior_dist}({lower},{upper})'
    else:
        raise ValueError(f'dist={prior_dist} not supported')
    if verbose:
        print(f'{name} ~ {spec}')
    return rv


def _as_init_arrays(values):
    """Coerce MCMC init values to arrays.

    map_soln, whether freshly produced by optim.optimize or loaded from a
    map.pkl written by a completed run, is fed straight back to init_to_value
    on a resume. numpyro substitutes those values verbatim for their sample
    sites, so a raw float reaches get_rv's rv.squeeze() and raises. get_map_soln
    now preserves free parameter shapes instead of unwrapping them, but a
    map.pkl pickled before that change can still hold plain Python floats for
    scalar sites, so this stays a defensive guard. Arrays in, arrays out, so
    downstream sites always get something array-like.
    """
    return {k: jnp.asarray(v) for k, v in values.items()}


def sample(
    model_fn,
    map_soln,
    tune=1000,
    draws=1000,
    chains=2,
    cores=2,
    random_seed=None
):
    # None keeps the historical fixed key, so an unseeded config behaves
    # exactly as it did before random_seed existed
    rng_key = jax.random.PRNGKey(0 if random_seed is None else random_seed)
    init_strategy = init_to_value(values=_as_init_arrays(map_soln))
    nuts = NUTS(model_fn, dense_mass=True, init_strategy=init_strategy,
                target_accept_prob=0.95)
    mcmc = MCMC(nuts, num_warmup=tune, num_samples=draws, num_chains=chains)
    mcmc.run(rng_key, extra_fields=("potential_energy",))
    return mcmc


def _compute_light_curve(orbit, u_star, time, texp):
    """Compute transit light curve using jaxoplanet."""
    lc_fn = limb_dark_light_curve(orbit, u_star[0], u_star[1])
    if texp is not None and texp > 0:
        lc_fn = integrate(lc_fn, exposure_time=texp, num_samples=7)
    return lc_fn(time)


def build(
    datasets,
    priors,
    nplanets,
    masks={},
    start=None,
    basis='duration',
    chromatic=False,
    use_gp=False,
    include_mean=True,
    include_flare=False,
    chromatic_flare=False,
    include_bump=False,
    chromatic_bump=False,
    fixed=[],
    verbose=False,
    logp_threshold=1,
    sequential_opt=False,
    use_custom_optimizer=True,
    gp_config=None,
    n_restarts=1,
    optimize=True
):
    logging.info("Building model")

    bands = set([i['band'] for i in datasets.values()])

    # Determine flare/bump counts from priors (needed at model-build time, not trace time)
    nflares = 0
    if include_flare:
        nflares = len(priors['flare_tpeak']) if isinstance(priors['flare_tpeak'], np.ndarray) else 1
    nbumps = 0
    if include_bump:
        nbumps = len(priors['bump_tcenter']) if isinstance(priors['bump_tcenter'], np.ndarray) else 1

    # Print prior specs once (not inside model_fn which gets called repeatedly)
    if verbose:
        for p in ['dur', 't0', 'period', 'ror', 'b']:
            if p not in fixed and f'{p}_prior' in priors:
                pri = priors[f'{p}_prior']
                val = priors[p]
                unc = priors[f'{p}_unc']
                print(f'{p} ~ {pri}({val},{unc})')
        if use_gp:
            for p in ['log_amp', 'log_scale']:
                key = f'gp_{p}'
                print(f'{key} ~ {priors[f"{key}_prior"]}({priors[key]},{priors[f"{key}_unc"]})')

    def model_fn():

        v = {}

        # Parameters for the stellar properties (limb darkening)
        for band in bands:
            p = f'u_star_{band}'
            if 'u_star' in priors.keys():
                if 'u_star' in fixed:
                    v[p] = jnp.array(priors['u_star'][band], dtype=jnp.float64)
                else:
                    if priors['u_star_prior'] == 'uniform':
                        # get_priors encodes the configured range as a midpoint
                        # and a full width per band, the same convention get_rv
                        # decodes for every other uniform parameter. hardcoding
                        # 0 to 1 here instead discarded the range the config
                        # asked for, silently and without an error.
                        #
                        # the uniform branch stores one scalar midpoint and
                        # width per band, not one per coefficient, so broadcast
                        # across both. the site has to stay shape (2,):
                        # _compute_light_curve reads u_star[0] and u_star[1]
                        mid = jnp.asarray(priors['u_star'][band], dtype=jnp.float64)
                        wid = jnp.asarray(priors['u_star_unc'][band], dtype=jnp.float64)
                        lower = jnp.broadcast_to(mid - wid/2, (2,))
                        upper = jnp.broadcast_to(mid + wid/2, (2,))
                        v[p] = numpyro.sample(p, dist.Uniform(lower, upper))
                        if verbose:
                            print(f'{p} ~ uniform({lower},{upper})')
                    else:
                        mu = jnp.array(priors['u_star'][band], dtype=jnp.float64)
                        sd = jnp.array(priors['u_star_unc'][band], dtype=jnp.float64)
                        v[p] = get_rv(
                            name=p,
                            prior_dist=priors['u_star_prior'],
                            shape=2,
                            mu=mu,
                            sd=sd,
                            verbose=False
                        )
            else:
                # No u_star priors provided - use wide uniform
                v[p] = numpyro.sample(p, dist.Uniform(
                    jnp.zeros(2), jnp.ones(2)))

        if basis == 'duration':
            p = "dur"
            if p in fixed:
                v[p] = jnp.array(priors[p], dtype=jnp.float64)
            else:
                v[p] = get_rv(key=p, priors=priors, shape=nplanets, verbose=False)
        elif basis == 'density':
            raise NotImplementedError

        # flare parameters
        if include_flare:
            flare_tpeak = get_rv(
                key='flare_tpeak', priors=priors, shape=nflares, verbose=False)
            flare_fwhm = get_rv(
                key='flare_fwhm', priors=priors, shape=nflares, verbose=False)
            if chromatic_flare:
                for band in bands:
                    name = f'flare_ampl_{band}'
                    v[name] = get_rv(
                        key='flare_ampl', name=name, priors=priors,
                        shape=nflares, verbose=False)
            else:
                flare_ampl = get_rv(
                    key='flare_ampl', priors=priors, shape=nflares, verbose=False)

        # bump parameters
        if include_bump:
            bump_tcenter = get_rv(
                key='bump_tcenter', priors=priors, shape=nbumps, verbose=False)
            bump_width = get_rv(
                key='bump_width', priors=priors, shape=nbumps, verbose=False)
            if chromatic_bump:
                for band in bands:
                    name = f'bump_ampl_{band}'
                    v[name] = get_rv(
                        key='bump_ampl', name=name, priors=priors,
                        shape=nbumps, verbose=False)
            else:
                bump_ampl = get_rv(
                    key='bump_ampl', priors=priors, shape=nbumps, verbose=False)

        # parameters for the planets
        for p in "t0 period ror b".split():
            if p in fixed:
                v[p] = jnp.array(priors[p], dtype=jnp.float64)
            else:
                if p == 'ror' and chromatic:
                    for band in bands:
                        name = f'ror_{band}'
                        v[name] = get_rv(
                            key=p, name=name, priors=priors,
                            shape=nplanets, verbose=False, bounds=[0, 1])
                elif p in ['ror', 'b']:
                    v[p] = get_rv(
                        key=p, priors=priors, shape=nplanets,
                        verbose=False, bounds=[0, 1])
                else:
                    v[p] = get_rv(
                        key=p, priors=priors, shape=nplanets, verbose=False)

        # Orbit model
        if basis == 'duration':
            if chromatic:
                ror_mean = jnp.mean(jnp.stack([v[f'ror_{band}'] for band in bands]), axis=0)
            else:
                ror_mean = v['ror']
            orbit = TransitOrbit(
                duration=v['dur'],
                period=v['period'],
                time_transit=v['t0'],
                impact_param=v['b'],
                radius_ratio=ror_mean
            )
        else:
            raise ValueError(f'basis={basis} not supported')

        # GP shared parameters (sampled once, used by all datasets)
        gp_shared = {}
        if use_gp:
            per_ds = gp_config.get('per_dataset', []) if gp_config else []
            for p in ['log_amp', 'log_scale']:
                if p not in per_ds:
                    gp_shared[p] = get_rv(key=f'gp_{p}', priors=priors, shape=1, verbose=False)

        # loop over the datasets
        for n, (name, data) in enumerate(datasets.items()):

            x, y, yerr, X, texp, x_hr, band = [data.get(i) for i in 'x y yerr X texp x_hr band'.split()]
            mask = masks[name]
            if mask is None:
                mask = np.ones(len(x), dtype=bool)

            if include_mean:
                mean = numpyro.sample(f"{name}_mean", dist.Normal(0.0, 10.0))
            else:
                mean = 0.0

            # linear systematics model
            if X is not None:
                ncols = X[mask].shape[1]
                mu_w = jnp.zeros(ncols)
                sd_w = jnp.ones(ncols) * 1e3
                weights = numpyro.sample(f'{name}_weights', dist.Normal(mu_w, sd_w))
                lm = numpyro.deterministic(f"{name}_lm", jnp.dot(X[mask], weights))
            else:
                lm = 0.0

            # Transit jitter
            lower_jit = -10.0
            upper_jit = float(np.log(10*np.std(y[mask])))
            log_sigma_lc = numpyro.sample(
                f'{name}_log_sigma_lc',
                dist.Uniform(lower_jit, upper_jit))

            # GP kernel for this dataset
            if use_gp:
                per_ds = gp_config.get('per_dataset', []) if gp_config else []
                gp_log_amp = gp_shared.get('log_amp', None)
                if gp_log_amp is None:
                    gp_log_amp = get_rv(key='gp_log_amp', name=f'gp_log_amp_{name}',
                                        priors=priors, shape=1, verbose=False)
                gp_log_scale = gp_shared.get('log_scale', None)
                if gp_log_scale is None:
                    gp_log_scale = get_rv(key='gp_log_scale', name=f'gp_log_scale_{name}',
                                          priors=priors, shape=1, verbose=False)
                gp_amp = 10**gp_log_amp
                gp_scale = 10**gp_log_scale
                gp_kernel = celerite_terms.Matern32Term(sigma=gp_amp, rho=gp_scale)

            if include_flare:
                if chromatic_flare:
                    flare_ampl_band = v[f'flare_ampl_{band}']
                else:
                    flare_ampl_band = flare_ampl

                if nflares == 1:
                    tpeak_val = flare_tpeak[0] if flare_tpeak.ndim > 0 else flare_tpeak
                    fwhm_val = flare_fwhm[0] if flare_fwhm.ndim > 0 else flare_fwhm
                    ampl_val = flare_ampl_band[0] if flare_ampl_band.ndim > 0 else flare_ampl_band
                    flare = aflare1(x[mask], tpeak=tpeak_val, fwhm=fwhm_val, ampl=ampl_val)
                else:
                    flare_total = jnp.zeros_like(x[mask])
                    for i in range(nflares):
                        flare_component = aflare1(
                            x[mask], tpeak=flare_tpeak[i],
                            fwhm=flare_fwhm[i], ampl=flare_ampl_band[i])
                        flare_total = flare_total + flare_component
                    flare = flare_total
                numpyro.deterministic(f"{name}_flare", flare)
            else:
                flare = 0.0

            if include_bump:
                if chromatic_bump:
                    bump_ampl_band = v[f'bump_ampl_{band}']
                else:
                    bump_ampl_band = bump_ampl

                if nbumps == 1:
                    tcenter_val = bump_tcenter[0] if bump_tcenter.ndim > 0 else bump_tcenter
                    width_val = bump_width[0] if bump_width.ndim > 0 else bump_width
                    ampl_val = bump_ampl_band[0] if bump_ampl_band.ndim > 0 else bump_ampl_band
                    bump = bump_model(x[mask], t_center=tcenter_val, width=width_val, amplitude=ampl_val)
                else:
                    bump_total = jnp.zeros_like(x[mask])
                    for i in range(nbumps):
                        bump_component = bump_model(
                            x[mask], t_center=bump_tcenter[i],
                            width=bump_width[i], amplitude=bump_ampl_band[i])
                        bump_total = bump_total + bump_component
                    bump = bump_total
                numpyro.deterministic(f"{name}_bump", bump)
            else:
                bump = 0.0

            # Compute the model light curve
            if chromatic:
                ror = v[f'ror_{band}']
                orbit_band = TransitOrbit(
                    duration=v['dur'],
                    period=v['period'],
                    time_transit=v['t0'],
                    impact_param=v['b'],
                    radius_ratio=ror
                )
                light_curves = _compute_light_curve(
                    orbit_band, v[f'u_star_{band}'], x[mask], texp) * 1e3
            else:
                ror = v['ror']
                light_curves = _compute_light_curve(
                    orbit, v[f'u_star_{band}'], x[mask], texp) * 1e3

            numpyro.deterministic(f"{name}_light_curves", light_curves)
            light_curve = jnp.sum(light_curves, axis=-1) + mean + lm + flare + bump
            resid = y[mask] - light_curve

            # Compute high-res model light curve
            if chromatic:
                light_curves_hr = _compute_light_curve(
                    orbit_band, v[f'u_star_{band}'], x_hr, texp) * 1e3
            else:
                light_curves_hr = _compute_light_curve(
                    orbit, v[f'u_star_{band}'], x_hr, texp) * 1e3
            numpyro.deterministic(f"{name}_light_curves_hr", light_curves_hr)

            # Likelihood
            if use_gp:
                x_gp = jnp.array(x[mask])
                yerr_gp = jnp.array(yerr[mask])
                diag = yerr_gp**2 + jnp.exp(2*log_sigma_lc) * jnp.ones_like(yerr_gp)
                gp_obj = CeleriteGP(gp_kernel)
                gp_obj.compute(x_gp, diag=diag)
                obs_resid = jnp.array(y[mask]) - light_curve
                numpyro.sample(
                    f"{name}_y_observed",
                    gp_obj.numpyro_dist(),
                    obs=obs_resid
                )
            else:
                sigma = jnp.sqrt(jnp.exp(2*log_sigma_lc) + yerr[mask]**2)
                numpyro.sample(
                    f"{name}_y_observed",
                    dist.Normal(light_curve, sigma),
                    obs=y[mask]
                )

            # Phased light curve model: identical to the transit model used in
            # the likelihood above, so reuse it rather than recomputing
            numpyro.deterministic(
                f"{name}_lc_pred",
                jnp.sum(light_curves, axis=-1)
            )

    # MAP optimization (skipped when reusing a cached solution)
    if not optimize:
        return model_fn, None

    logging.info("Running MAP optimization")
    map_soln = optim.optimize(model_fn, verbose=verbose, n_restarts=n_restarts)
    print("MAP optimization complete")

    # Compute GP predictions from MAP solution
    if use_gp:
        map_soln = _add_gp_predictions(map_soln, datasets, masks, gp_config)

    return model_fn, map_soln


def _build_gp(map_soln, name, x, yerr, gp_config):
    """A computed celerite2 GP and its noise diagonal, from MAP hyperparameters.

    Uses the numpy celerite2 backend: both callers run post-hoc, outside the
    JAX model context. x and yerr must already be masked.

    Shared so that the kernel, the log10 amplitude/scale convention, and the
    jitter diagonal cannot drift apart between the GP prediction and the
    effective degrees of freedom.
    """
    from celerite2 import GaussianProcess as C2NumpyGP, terms as c2np_terms

    per_ds = gp_config.get('per_dataset', []) if gp_config else []
    if 'log_amp' in per_ds:
        log_amp = map_soln[f'gp_log_amp_{name}']
    else:
        log_amp = map_soln['gp_log_amp']
    if 'log_scale' in per_ds:
        log_scale = map_soln[f'gp_log_scale_{name}']
    else:
        log_scale = map_soln['gp_log_scale']

    amp = float(10**np.squeeze(log_amp))
    scale = float(10**np.squeeze(log_scale))
    log_sigma_lc = float(np.squeeze(map_soln[f'{name}_log_sigma_lc']))
    diag = np.exp(2*log_sigma_lc) + yerr**2

    gp = C2NumpyGP(c2np_terms.Matern32Term(sigma=amp, rho=scale))
    gp.compute(x, diag=diag)
    return gp, diag


def compute_gp_edf(map_soln, datasets, masks, gp_config, max_points=5000):
    """Effective degrees of freedom the GP absorbs, per dataset.

    A GP posterior mean is the linear smoother y_hat = K (K+S)^-1 y, and its
    effective degrees of freedom is the trace of that smoother. Counting the
    GP as its handful of hyperparameters understates it badly: on a real fit
    a GP charged for 5 hyperparameters absorbed about 110 degrees of freedom
    out of 560 points.

    The GP's own smoother trace is computed as n - tr(S (K+S)^-1), which needs
    only the diagonal of the inverse: n solves, each O(n), so O(n^2) overall.
    The design overlap is then subtracted from it, at the cost of one further
    n by p solve.

    The value returned is the GP's share of the joint effective degrees of
    freedom of a fit with a parametric mean and a GP, with the overlap
    between the GP and the design matrix already subtracted out. The caller
    adds the p design columns on top, since those are already counted in
    nparams. This is a tight upper bound rather than an exact figure: the
    overlap with the per-dataset offset and the transit parameters is not
    subtracted, since neither is a column of the design matrix.

    Returns {dataset_name: edf}, or None if any dataset exceeds max_points,
    since the cost is prohibitive at survey scale, or if any dataset's design
    matrix is rank deficient, since the overlap is then undefined. Returning
    None rather than a partial sum keeps the caller from reporting a number
    that omits a dataset.
    """
    for name, data in datasets.items():
        mask = masks[name]
        n = len(data['x']) if mask is None else int(np.sum(mask))
        if n > max_points:
            logging.warning(
                f'skipping GP effective degrees of freedom: dataset {name} has '
                f'{n} points, above max_points={max_points}; the calculation is '
                'quadratic in the number of points'
            )
            return None

    edf = {}
    for name, data in datasets.items():
        x, yerr = data['x'], data['yerr']
        mask = masks[name]
        if mask is None:
            mask = np.ones(len(x), dtype=bool)
        gp, diag = _build_gp(map_soln, name, x[mask], yerr[mask], gp_config)
        n = int(np.sum(mask))
        basis = np.eye(n)
        inv_diag = np.array([gp.apply_inverse(basis[:, i])[i] for i in range(n)])
        gp_trace = n - np.sum(diag * inv_diag)

        X = data.get('X')
        if X is None:
            # no parametric mean, so nothing for the GP to overlap with
            edf[name] = float(gp_trace)
            continue

        Xm = X[mask]
        p = Xm.shape[1]
        # check the rank rather than waiting for the solve below to fail on it.
        # LAPACK raises only on an exactly zero pivot, and the configurations
        # that land here are singular by structure rather than by repetition,
        # so rounding leaves them nominally invertible and the solve returns a
        # confident wrong number
        rank = np.linalg.matrix_rank(Xm)
        if rank < p:
            logging.warning(
                f'skipping GP effective degrees of freedom: dataset {name} has a '
                f'rank deficient design matrix ({p} columns, rank {rank}), so the '
                f'overlap between the GP and the design is undefined'
            )
            return None
        # A = C^-1 X in one solve; celerite2 accepts the n by p matrix directly
        A = np.asarray(gp.apply_inverse(Xm))
        # tr(K C^-1 P) = p - tr((A^T S A)(X^T A)^-1), where C = K + S and P is
        # the design-matrix projection under C, from K C^-1 = I - S C^-1 and
        # the cyclic property of the trace
        try:
            overlap = p - np.trace(
                np.linalg.solve(Xm.T @ A, A.T @ (diag[:, None] * A)))
        except np.linalg.LinAlgError:
            # backstop for a full rank X whose X^T C^-1 X is still singular
            logging.warning(
                f'skipping GP effective degrees of freedom: dataset {name} has a '
                f'rank deficient design matrix, so the overlap between the GP and '
                f'the design is undefined'
            )
            return None
        edf[name] = float(gp_trace - overlap)
    return edf


def _add_gp_predictions(map_soln, datasets, masks, gp_config):
    """Compute GP conditional mean from MAP params and add to map_soln."""
    for name, data in datasets.items():
        x, y, yerr = data['x'], data['y'], data['yerr']
        mask = masks[name]
        if mask is None:
            mask = np.ones(len(x), dtype=bool)

        # Residuals = data - deterministic model
        # Squeeze all values to remove trailing singleton dims from numpyro trace
        # the mean is only a model site when include_mean=True
        if f'{name}_mean' in map_soln:
            light_curve = float(np.squeeze(map_soln[f'{name}_mean']))
        else:
            light_curve = 0.0
        lcs = np.squeeze(map_soln[f'{name}_light_curves'])
        if lcs.ndim > 1:
            light_curve = light_curve + np.sum(lcs, axis=-1)
        else:
            light_curve = light_curve + lcs
        if f'{name}_lm' in map_soln:
            light_curve = light_curve + np.squeeze(map_soln[f'{name}_lm'])
        if f'{name}_flare' in map_soln:
            light_curve = light_curve + np.squeeze(map_soln[f'{name}_flare'])
        if f'{name}_bump' in map_soln:
            light_curve = light_curve + np.squeeze(map_soln[f'{name}_bump'])

        residuals = y[mask] - light_curve

        gp, _ = _build_gp(map_soln, name, x[mask], yerr[mask], gp_config)
        map_soln[f'{name}_gp_pred'] = gp.predict(residuals)

    return map_soln
