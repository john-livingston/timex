# Usage

## Command-line interface

```bash
timex <working_directory> [options]
```

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Enable verbose console output |
| `-o`, `--outdir` | Output directory name (default: `out`) |

### Examples

```bash
timex examples/hip67522b
timex examples/hip67522b -v
timex examples/hip67522b -o model1
```

## Python API

```python
import yaml
from timex.fit import TransitFit

sys_params = yaml.load(open('sys.yaml'), Loader=yaml.FullLoader)
fit_params = yaml.load(open('fit.yaml'), Loader=yaml.FullLoader)

fit = TransitFit(sys_params, fit_params, wd='.')
fit.build_model()
fit.clip_outliers()
fit.sample()
fit.plot_corner()
fit.save_results()
```

### Loading saved results

```python
from timex.fit import TransitFit

fit = TransitFit.from_dir('examples/hip67522b')
```

## Pipeline

The `timex` CLI runs the following steps in order:

1. **Load data** -- read light curve files, bin, detrend
2. **Build model** -- construct PyMC model with priors, optimize for MAP solution
3. **Clip outliers** -- sigma-clip residuals (if `clip: true` in data config)
4. **Re-fit** -- rebuild model with outlier mask applied
5. **Sample** -- MCMC sampling with PyMC
6. **Plot** -- light curve fits, corner plots, trace plots, limb darkening
7. **Save** -- summary statistics, transit times, posterior samples, corrected light curves

## Outputs

All outputs are saved to the `out/` directory (or custom `--outdir`):

| File | Description |
|------|-------------|
| `fit.png` | Light curve fit with residuals |
| `corner.png` | Corner plot of posterior distributions |
| `trace.png` | MCMC trace plot |
| `summary.csv` | Parameter summary statistics |
| `tc.txt` | Fitted transit center times |
| `ic.txt` | Information criteria (BIC, AIC, AICc); for GP fits also the effective-degrees-of-freedom corrected values |
| `posterior_samples.csv.gz` | Full posterior samples |
| `*-cor.csv` | Corrected (detrended) light curves |
| `timex.log` | Full log file |
| `fit.yaml`, `sys.yaml` | Copies of input configuration |
| `cache.json` | Records which config and data each cached artifact was produced from |

### Information criteria

`ic.txt` reports BIC, AIC and AICc built from the maximized log likelihood.

!!! warning "Values written before this change are not comparable"

    The criteria used to be built from the joint log posterior that the sampler
    reports alongside the MAP draw, which also carries every prior term and the
    transform Jacobian. They are now built from the likelihood alone, so every
    number in `ic.txt` shifts, by an amount that depends on the model's priors
    rather than on its fit. Do not compare an `ic.txt` written before this
    change against one written after it, and do not carry an old number into a
    model comparison table.

    Nothing marks the old files as stale. No configuration changed, so the
    cache keys in `cache.json` are unchanged: a resumed run reuses the same
    trace and MAP as before and gives no sign that the criteria mean something
    different now. `save_results` does rewrite `ic.txt` on every run, so
    rerunning any fit refreshes it, but an `ic.txt` sitting in an output
    directory you have not rerun still holds a number on the old definition.

### Effective degrees of freedom

For a GP fit, `ic.txt` carries a second set of rows (`edf`, `nparams_edf`,
`BIC_edf`, `AIC_edf`, `AICc_edf`) that charge the GP for the flexibility it
actually uses instead of for its handful of hyperparameters.

`nparams_edf` is an upper bound. `compute_gp_edf` measures the trace of the
GP's own smoother and does not subtract the flexibility the GP shares with the
design matrix, so the GP is charged for variation the design already accounts
for. The overlap is typically close to the full number of design columns: a 10
column design can inflate `BIC_edf` by up to about 63 on 560 points. The bias
always runs one way, penalising the GP, so a GP that still wins on `BIC_edf`
really wins, while one that loses by less than the number of design columns has
not been ruled out.

### Corrected light curves

One `*-cor.csv` is written per dataset, with columns `x`, `y` and `yerr`.

`x` is in the data's native time system, with `ref_time` added back on. `y` is
relative flux: every non-transit component of the model (the mean, the linear
or spline systematics model, any flare or bump term, and the GP prediction)
subtracted from the data, converted from ppt, with the baseline restored to 1.

`yerr` is the photometric error and the fitted jitter added in quadrature,
`sqrt(yerr**2 + exp(2*log_sigma_lc))`, converted from ppt. It is not the
photometric error from the input file: the fitted jitter routinely exceeds it,
so these bars are wider, and a refit of the published file sees a scatter close
to the one the original fit did.

Two details are worth knowing before comparing the file against a figure or a
likelihood:

- The jitter here is the one at the maximum posterior draw, while the wider of
  the two error bars `fit.png` draws uses the posterior median of
  `log_sigma_lc`. The two estimators can differ by tens of percent on a given
  dataset, so the published column and the drawn bar are close but not equal.
- Without a GP this is exactly the weight the likelihood gave each point. With
  a GP the likelihood weights by the full covariance `K + S`, and only the
  diagonal `S` is published, so the file carries none of the correlated noise
  the fit accounted for.
