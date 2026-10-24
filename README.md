# Conformal PID Control for Time-Series Prediction

[![Paper: arXiv](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2307.16895)
[![Package management: uv](https://img.shields.io/badge/package%20management-uv-green)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![Simplified PID diagram](media/PID-simplified.svg)

This repository is about producing **prediction sets for time series**.

The methods here are guaranteed to have coverage for any, possibly _adversarial sequence_.
We take a **control systems outlook** on performing this task, introducing a method called [Conformal PID Control](https://arxiv.org/abs/2307.16895).

Several methods are implemented herein, including online quantile regression (quantile tracking/P control), adaptive conformal prediction, and more!
**This codebase makes it easy to extend the methods/add new datasets.**
We will describe how to do so below.

## Getting Started

This project requires Python 3.11.13 or newer and uses [uv](https://docs.astral.sh/uv/) for package management. To reproduce the experiments in the original paper, clone this repo, install `uv`, and run the following from the repository root. The paper configs use Prophet, which is supplied to the upstream harness without adding it to the fork's runtime dependencies.

```bash
uv sync
cd tests
uv run --with prophet bash run_tests.sh
uv run bash make_plots.sh
```

The one exception is the COVID experiment. It requires the `deaths.csv` data file from [this Drive link](https://drive.google.com/file/d/1p_l3bKJjypmJDmIZ0tqrwIZWDo8JDjvY/view?usp=sharing). Put the file beside the [COVID preprocessing notebook](tests/datasets/covid-ts-proc/statewide/death-forecasting-perstate-lasso-qr.ipynb), then run the notebook before launching the paper harness:

```bash
uv run --with jupyter jupyter lab tests/datasets/covid-ts-proc/statewide/death-forecasting-perstate-lasso-qr.ipynb
```

## Adding New Methods

### Step 1: Defining the method

The `core/methods.py` file contains all methods.
Consider the following method header, for the P controller/quantile tracker, as an example:

```python
def quantile(
    scores,
    alpha,
    lr,
    ahead,
    proportional_lr=True,
    *args,
    **kwargs
):
```

The first three arguments, `scores`, `alpha`, and `lr`, are _required_ arguments for all methods.
The first argument, `scores`, expects a numpy array of conformal scores. The second argument, `alpha`, is the desired miscoverage. Finally, the third argument, `lr`, is the learning rate. (In our paper, this is η, and in the language of control, this is Kₚ.)

The rest of the arguments listed are required arguments specific to the given method. The argument `ahead` determines how many steps ahead the prediction is made --- for example, if `ahead=4`, that means we are making 4-step-ahead predictions (one step is defined by the resolution of the input array `scores`). The function of `*args` and `**kwargs` is to allow methods to take arguments given in a dictionary form.

All methods should `return` a dictionary of results that includes the method name and the sequence of qₜ. In the quantile example case, the dictionary should look like the following, where `qs` is a numpy array of quantiles the same length as `scores`:

`results = {"method": "Quantile", "q" : qs}`
Methods that follow this formatting will be able to be processed automatically by our testing infrastructure.

### Step 2: Edit the config to include your method

TL;DR: go to each config in `tests/configs`, and add a line under `methods:` for each method you want to run, along with what learning rates to test. The below example, from `tests/configs/AMZN.yaml`, will ask the testing suite to run the quantile tracker on the Amazon stock price dataset with five different learning rate choices.

```yaml
Quantile:
  lrs:
    - 1
    - 0.5
    - 0.1
    - 0.05
    - 0
```

As background, this is part of our little testing infrastructure for online conformal.
The infrastructure spawns a parallel process for every dataset, making it efficient to test one method on all datasets with only one command (the commands to run the tests and plots are `uv run --with prophet bash run_tests.sh` and `uv run bash make_plots.sh`, respectively, from `tests/`).

The infrastructure works like this.

- The user defines a file in `tests/configs/` describing an experiment, i.e., a dataset name and a combination of methods and settings for each method to run.
- The script `tests/run_tests.sh` calls `tests/base_test.py` on every `.yaml` file in the `tests/configs` directory.
- The script `tests/make_plots.sh` calls `inset_plot.py` and `base_plots.py` to produce the plots in the main text and appendix of our paper, respectively.

### Step 3: Edit `base_test.py` to include your method

Near the top of [`base_test.py`](tests/base_test.py), import your method from `core`.
Then add it to the method-dispatch `if/elif` block in the main experiment loop.

## Adding New Datasets

### Step 1: Load and preprocess the dataset

First, download your dataset and put it in `tests/datasets`.
Then, edit the `tests/datasets.py` file to add a name for your dataset and some processing code for it.
The dataset should be a `pandas` dataframe with a valid `datetime` index (it has to be evenly spaced, and correctly formatted with no invalid values), and at least one column simply titled `y`. This column represents the target value.

Alternatively, including a column titled `forecasts` or `scorecasts` will cause the infrastructure to use these forecasts/scorecasts instead of the ones it would have produced on its own. This is useful if you have defined a good forecaster/scorecaster outside our framework, and you just want to use our code to run conformal on top of that.
Extra columns can be used to add information for more complex forecasting/scorecasting strategies.

### Step 2: Create a config file for the dataset

As mentioned above, a config file should be made for each dataset, describing what methods should be run with what parameters.
The example of `tests/configs/AMZN.yaml` can be followed.

After executing these two steps, you should be able to run `uv run python base_test.py configs/your_dataset.yaml` from `tests/` and the results will be computed!
Alternatively, you can just execute the bash scripts.

## Workarounds for Known Bugs

On M1/M2 Mac, in order to use Prophet, follow the instructions at this link: <https://github.com/facebook/prophet/issues/2250>.

## Fork: hierarchical reconciliation (state → US)

This fork asks whether reconciling 4-week-ahead death quantiles across the US → 56-leaf
hierarchy before conformal P/PI/PID control improves temporal coverage. The frozen protocols
and outcome decisions are in [`docs/protocol-decisions.md`](docs/protocol-decisions.md); detailed
operating instructions are in [`docs/development.md`](docs/development.md).

The completed experiments are negative results rather than evidence of a general hierarchy
benefit:

- [Phase 1](docs/analysis/us-hier-v2.md) found a null result for the single COVIDhub ensemble
  series.
- [Phase 2](docs/analysis/panel-leaves-v1.md) found no supported leaf-level benefit across the
  eligible-forecaster panel; unreconciled and reconciled arms instead expose level- and
  coverage-width trade-offs (D29).
- The [probabilistic model benchmark](docs/analysis/model-benchmarks-v1.md) found that the
  COVIDhub ensemble remained stronger than fixed ARIMA, SARIMA, and DeepAR-style forecasts,
  with or without MinT(Shrink), and found no general hierarchy benefit (D31).

### Reproducing the fork experiments

Download the same `deaths.csv` source used above to `data/deaths.csv` and verify it before
running an experiment:

```bash
sha256sum data/deaths.csv
# cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06

uv run python -m hcp.run configs/us_hier.toml
uv run python -m hcp.run configs/panel_leaves.toml
uv run python -m hcp.run configs/model_benchmarks.toml
```

Each command publishes an immutable directory under `results/<run_id>/` and refuses to
overwrite an existing run. Copy the relevant TOML file and choose a new `run_id` before a
rerun. The probabilistic benchmark is CPU-heavy because ARIMA and SARIMA are refit for every
node and weekly origin and DeepAR trains deterministically on CPU; see the development guide
for focused checks, runtime guidance, stderr review, and artifact details.

The cumulative v2 exploratory benchmark is separately registered by D32 and adds Prophet,
Transformer, local L1 quantile regression, and an enhanced DeepAR-style model. It uses four
principal arms and two declared Holm families; prior v1 outcomes informed its design, so v2
remains exploratory. Protocol, review, verification, and limitations are recorded in
`docs/analysis/model-benchmarks-v2-protocol-readiness.md`. The prospective CUDA amendment
registers neural execution on the RTX 4080 (`cuda:0`, deterministic FP32, TF32 disabled).
Two pre-amendment attempts remain unpublished. CUDA repeatability checks passed twice;
full verification still has documented warning-fixture and formatting failures.
After final review, fresh full verification, and explicit launch authorization, use
an interactive CUDA terminal from this worktree:

```bash
UV_CACHE_DIR=/tmp/model-benchmarks-v2-cuda UV_NO_SYNC=1 uv run python -m hcp.run configs/model_benchmarks_v2.toml
```

Run publication is immutable at `results/model-benchmarks-v2/`; an existing result is never
overwritten.

The fixed work directory is `results/.model-benchmarks-v2.partial/`. Immutable
`run-state.json` binds code, inputs, config, packages, seed plan, and GPU/runtime identity;
`batches/<model>/<origin>.batch/` holds immutable hash-validated origin forecasts and
diagnostics. From a second terminal in the same worktree, monitor:

```bash
watch -n 5 cat results/.model-benchmarks-v2.partial/progress.json
tail -f results/.model-benchmarks-v2.partial/runtime.jsonl
```

Resume by repeating the exact launch command from the same clean Git SHA and matching
environment; there is no resume flag. Valid completed batches are skipped; identity or
batch corruption stops execution. Preserve incompatible work directories outside the
registered path. Keep `HEAD` fixed until publication and across resume: **do not commit,
merge, rebase, pull, or checkout during execution**. The readiness record contains
amendment/implementation SHAs, GPU identity, verification evidence, and the complete
fresh-launch and batch contract. No official completed result is claimed.
