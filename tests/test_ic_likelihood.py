"""Information criteria must be built from the maximized likelihood.

BIC, AIC and AICc are defined in terms of the maximized likelihood. Feeding
them the maximized joint log posterior instead makes them depend on every
prior width in the model, including the systematics weight prior in model.py,
whose width is a fixed constant nobody chose as a modelling statement.
"""
import numpy as np
import pytest

from timex import util


NDATA = 8
SIGMA = 0.5
# a two column design matrix, standing in for a systematics model, and data
# that is not in its span, so the residuals depend on the weights
X = np.column_stack([np.ones(NDATA), np.linspace(-1.0, 1.0, NDATA)])
Y = np.array([0.35, 0.10, -0.20, 0.05, 0.40, -0.15, 0.30, 0.00])
# two chains of three draws of the two weights
WEIGHTS = np.array([
    [[0.10, -0.20], [0.30, 0.05], [-0.05, 0.02]],
    [[0.02, 0.11], [-0.30, 0.25], [0.15, -0.05]],
])


def _log_likelihood_by_hand(w):
    """Gaussian log likelihood of Y under the linear model, written out."""
    resid = Y - X @ np.asarray(w)
    return float(np.sum(
        -0.5 * (resid / SIGMA) ** 2 - np.log(SIGMA) - 0.5 * np.log(2 * np.pi)
    ))


def _model(sd_w):
    """The timex likelihood in miniature: a Normal prior on the systematics
    weights whose width is a constant, and a Gaussian observed site."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model_fn():
        w = numpyro.sample('weights', dist.Normal(jnp.zeros(2), sd_w * jnp.ones(2)))
        # model.py assigns the systematics model from numpyro.deterministic the
        # same way, so a substituted value would reach the likelihood
        lm = numpyro.deterministic('g_lm', jnp.dot(jnp.asarray(X), w))
        numpyro.sample('g_y_observed', dist.Normal(lm, SIGMA), obs=jnp.asarray(Y))

    return model_fn


def _trace(sd_w, with_log_likelihood=True):
    """An InferenceData shaped like az.from_numpyro's output for _model(sd_w).

    Both groups come from the model itself: sample_stats from the joint density
    NUTS reports as potential energy, log_likelihood from the observed site
    alone. Hardcoding either would make the prior width test vacuous.
    """
    import arviz as az
    import jax.numpy as jnp
    from numpyro.infer.util import log_density, log_likelihood

    model_fn = _model(sd_w)
    # potential energy is the negative joint density in unconstrained space;
    # the only latent site here is already unconstrained, so no Jacobian term
    pe = np.array([[-float(log_density(model_fn, (), {}, {'weights': jnp.asarray(w)})[0])
                    for w in chain] for chain in WEIGHTS])
    groups = dict(
        posterior={'weights': WEIGHTS},
        sample_stats={'potential_energy': pe},
    )
    if with_log_likelihood:
        ll = log_likelihood(model_fn, {'weights': jnp.asarray(WEIGHTS)}, batch_ndims=2)
        groups['log_likelihood'] = {k: np.asarray(v) for k, v in ll.items()}
    return az.from_dict(**groups)


def _fit(trace, model_fn):
    """A TransitFit carrying only the attributes get_ic reads."""
    from timex import fit

    tf = fit.TransitFit.__new__(fit.TransitFit)
    tf.trace = trace
    tf.model_fn = model_fn
    tf.data = {'g': dict(x=np.arange(float(NDATA)))}
    tf.masks = {'g': None}
    tf.map_soln = {'weights': np.zeros(2)}
    return tf


@pytest.fixture(scope='module')
def traces():
    """One trace per prior width, built once: the tests only read them."""
    return {1e3: _trace(1e3), 1e4: _trace(1e4)}


@pytest.mark.parametrize('method', ['BIC', 'AIC', 'AICc'])
def test_ic_does_not_move_when_the_weight_prior_widens(method, traces):
    """sd_w is a constant in model.py, not a modelling choice, so it must not
    reach the criteria. Built from the joint log posterior, each design matrix
    column contributes log N(0; 0, sd_w), so a 10x wider prior shifts the
    criteria by 2 ln(10) per column and can decide a detrending comparison.
    """
    ic_narrow = _fit(traces[1e3], _model(1e3)).get_ic(method)
    ic_wide = _fit(traces[1e4], _model(1e4)).get_ic(method)

    assert ic_narrow == pytest.approx(ic_wide, abs=1e-6)


def test_the_prior_width_fixture_moves_the_log_posterior(traces):
    """Guards the test above: were the two traces to carry the same joint
    density, it would pass against code that reads the posterior."""
    _, lp_narrow = util.get_map_soln(traces[1e3])
    _, lp_wide = util.get_map_soln(traces[1e4])

    # both weights are far inside either prior, so widening it by 10x costs
    # only the Normal normalization, ln(10) per weight
    assert lp_narrow - lp_wide == pytest.approx(2 * np.log(10), abs=1e-3)


def test_get_max_loglike_maximizes_over_draws():
    """The definition asks for the maximized likelihood. Reading the likelihood
    off the maximum posterior draw instead is a different number."""
    import arviz as az

    idata = az.from_dict(
        posterior={'a': np.zeros((1, 3))},
        # the best posterior draw is draw 1, whose likelihood is the worst
        sample_stats={'lp': np.array([[0.0, 10.0, 0.0]])},
        log_likelihood={'g_y_observed': np.array(
            [[[-1.0, -2.0], [-3.0, -4.0], [-0.5, -0.25]]])},
    )

    assert util.get_max_loglike(idata)[0] == pytest.approx(-0.75)


def test_get_max_loglike_sums_every_observed_site():
    """One observed site per dataset, and the GP branch's site carries a single
    multivariate log probability per draw rather than one per point. Dropping
    either site understates the likelihood of a multi dataset fit."""
    import arviz as az

    idata = az.from_dict(
        posterior={'a': np.zeros((1, 2))},
        sample_stats={'lp': np.zeros((1, 2))},
        log_likelihood={
            'g_y_observed': np.array([[[-1.0, -2.0], [-0.5, -0.5]]]),
            'r_y_observed': np.array([[-10.0, -4.0]]),
        },
    )

    # per draw totals are -13.0 and -5.0
    assert util.get_max_loglike(idata)[0] == pytest.approx(-5.0)


def test_get_max_loglike_falls_back_to_the_model():
    """A trace saved without the log_likelihood group still has to yield the
    likelihood, and numpyro can evaluate the observed sites on the draws."""
    trace = _trace(1e3, with_log_likelihood=False)

    got, _ = util.get_max_loglike(trace, model_fn=_model(1e3))

    expected = max(_log_likelihood_by_hand(w) for w in WEIGHTS.reshape(-1, 2))
    assert got == pytest.approx(expected, abs=1e-6)


def test_the_fallback_evaluates_the_model_rather_than_reusing_deterministics():
    """numpyro substitutes whatever posterior entries it is handed, and the
    posterior carries the deterministic sites alongside the free parameters, so
    handing those back would let a recorded array stand in for the model's own
    systematics computation."""
    import arviz as az

    trace = az.from_dict(
        posterior={'weights': WEIGHTS, 'g_lm': np.zeros((2, 3, NDATA))},
        sample_stats={'lp': np.zeros((2, 3))},
    )

    got, _ = util.get_max_loglike(trace, model_fn=_model(1e3))

    expected = max(_log_likelihood_by_hand(w) for w in WEIGHTS.reshape(-1, 2))
    assert got == pytest.approx(expected, abs=1e-6)


def test_get_max_loglike_refuses_to_guess_without_a_likelihood():
    """Falling back to the log posterior here is the bug this all exists to
    stop, and it would be silent."""
    trace = _trace(1e3, with_log_likelihood=False)

    with pytest.raises(ValueError):
        util.get_max_loglike(trace)
