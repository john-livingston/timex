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

### Effective degrees of freedom

For a GP fit, `ic.txt` carries a second set of rows (`edf`, `nparams_edf`,
`BIC_edf`, `AIC_edf`, `AICc_edf`) that charge the GP for the flexibility it
actually uses instead of for its handful of hyperparameters.

`nparams_edf` is an upper bound. It measures the GP on its own and does not
subtract the flexibility the GP shares with the design matrix, which is
typically close to the full number of design columns: a 10 column design can
inflate `BIC_edf` by up to about 63 on 560 points. The bias always runs one
way, charging the GP too much, so a GP that still wins on `BIC_edf` really
wins, while one that loses by less than the number of design columns has not
been ruled out.
