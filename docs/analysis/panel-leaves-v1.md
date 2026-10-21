---
type: results-report
date: 2026-10-20
experiment_line: hierarchical-panel
round: 1
purpose: phase-2-closeout
status: final
source_artifacts:
  - results/panel-leaves-v1/manifest.json
  - results/panel-leaves-v1/metrics.parquet
  - results/panel-leaves-v1/level_metrics.parquet
  - results/panel-leaves-v1/contrasts.parquet
  - results/panel-leaves-v1/summary.parquet
  - results/panel-leaves-v1/eligibility.parquet
  - results/panel-leaves-v1/intervals.parquet
---

# Phase 2 results: hierarchical reconciliation across leaves and forecasters (`panel-leaves-v1`)

**Summary.** The pre-registered leaf-level primary contrast is inconclusive. For
`COVIDhub-4_week_ensemble`, MinT(Shrink) before PI lowers mean leaf coverage deviation by
0.0043 relative to no reconciliation, but its 95% moving-block-bootstrap interval crosses zero
([−0.0088, 0.0022], unadjusted p = 0.115). Across the eight eligible forecasters, six point
estimates favor MinT under PI, but the descriptive mean is only −0.0005 [−0.0022, 0.0015].
There is therefore no supported H1 effect and no stable H2 leaf-level generalization.

The negative result is informative. Without PI, MinT(Shrink) worsens leaf-mean coverage
deviation for all eight forecasters (descriptive mean +0.0190 [0.0104, 0.0268]) and worsens
scaled interval score on average (+0.0530 [0.0159, 0.0946]). The same raw reconciliation has
the opposite descriptive pattern at the US node. `mint_median`, which changes location while
preserving the raw base spread, provides no general leaf-level benefit. PI sometimes reduces
coverage deviation, but increases scaled interval score for every forecaster under every
reconciler. Phase 2 does not support using hierarchical reconciliation as a general remedy for
leaf-level temporal miscoverage in this design.

Sources below use `R/` for `results/panel-leaves-v1/`. All numerical results come from its
manifest or parquet files. Negative treatment-minus-`none` estimates are improvements for
coverage deviation, longest miss run, scaled width and scaled interval score. The official run
used commit `b788582`, recorded `git_dirty = false`, and wrote the complete artifact set
atomically.

## 1. Question and pre-registered design

Phase 2 asks whether reconciling state-to-US forecasts before conformal PI improves temporal
coverage at the leaves, and whether the result generalizes across forecasters (D20–D27).

- **Data and panel.** Eight eligibility-selected forecasters from `data/deaths.csv`, four-week
  ahead, with each forecaster evaluated on its own post-warmup window (66–126 weeks). The
  hierarchy is the US plus 56 leaves; the endpoint excludes the five near-zero territories.
- **Arms.** `none`, `bottom_up`, quantile-wise `mint_shrink`, and location-only
  `mint_median`; each is evaluated raw and under PI at the pre-registered lr 0.1. The lr grid
  is descriptive. PID-Theta is excluded because its paper-core scorecast is 2·ahead stale
  (D15).
- **Primary endpoint.** Mean leaf `coverage_deviation`, where each leaf's value is mean
  |rolling coverage (w = 10) − 0.8|. H1 is `mint_shrink − none` under PI for
  `COVIDhub-4_week_ensemble`.
- **Inference.** Paired moving-block bootstrap over weeks, block 8 and B = 2000. Each draw is
  shared across nodes, preserving within-week cross-leaf dependence. The H1 p-value is
  unadjusted; other contrasts use Holm within (block, forecaster, level, controller) families.
  Blocks 4 and 12 are sensitivity analyses.
- **H2 summary.** Forecasters are the resampling unit (n = 8, B = 2000). D24 designates this
  summary as descriptive only; its percentile intervals are uncertainty summaries, not a new
  confirmatory family.

The manifest records `evaluation_end`, endpoint counts and unscored nodes for every forecaster.
Seven forecasters have 51 endpoint leaves. `UMass-MechBayes` has 50 because `ny` is unscored;
the other unscored nodes are territories and do not change the endpoint count.

## 2. H1: primary leaf-level result

| Forecaster | Contrast | Endpoint | Estimate | 95% CI | p |
|---|---|---|---:|---|---:|
| `COVIDhub-4_week_ensemble` | `mint_shrink − none`, PI, lr 0.1 | mean leaf coverage deviation | −0.0043 | [−0.0088, 0.0022] | 0.115 |

Source: `R/contrasts.parquet`, the unique row with `primary == True`. The control has leaf-mean
coverage deviation 0.1685 and MinT 0.1642 (`R/level_metrics.parquet`), a 2.6% relative decrease.
The interval includes both no effect and modest improvement, so H1 is not supported. The point
estimate is unchanged at blocks 4 and 12; their intervals remain across zero and their
unadjusted p-values are 0.106 and 0.125.

The mean hides leaf heterogeneity: 24 of 51 endpoint leaves improve, 15 worsen and 12 are
unchanged to numerical tolerance. The largest changes are −0.0602 for `ut` and +0.0388 for
`mn`. These are descriptive node-level decompositions, not separately tested findings.

[Figure 3](figures/panel-leaves-v1-leaf-contrasts-pi.pdf) plots every endpoint-leaf contrast
for the primary forecaster. Its purpose is to show why a small average cannot be interpreted
as a uniform effect: improvements and degradations coexist. This weakens a mechanism claim
based only on the mean and argues against promoting the point estimate as a general gain.

## 3. H2: per-forecaster results and descriptive summary

The block-8 PI forest has no forecaster whose `mint_shrink` coverage-deviation CI excludes
zero. Six of eight estimates are negative, but they are small; the two positive estimates are
`COVIDhub-baseline` and `UMass-MechBayes`. `bottom_up` is exactly zero at the leaf level by
construction (D28).

| Forecaster | `mint_shrink − none` | `mint_median − none` | `bottom_up − none` |
|---|---:|---:|---:|
| `COVIDhub-4_week_ensemble` | −0.0043 [−0.0088, 0.0022] | −0.0007 [−0.0048, 0.0030] | 0 [0, 0] |
| `CEID-Walk` | −0.0004 [−0.0024, 0.0025] | −0.0007 [−0.0014, 0.0001] | 0 [0, 0] |
| `COVIDhub-baseline` | +0.0018 [−0.0011, 0.0033] | −0.0002 [−0.0006, 0.0001] | 0 [0, 0] |
| `COVIDhub-ensemble` | −0.0025 [−0.0059, 0.0043] | −0.0021 [−0.0051, 0.0045] | 0 [0, 0] |
| `COVIDhub-trained_ensemble` | −0.0020 [−0.0081, 0.0062] | +0.0023 [−0.0027, 0.0060] | 0 [0, 0] |
| `JHUAPL-Bucky` | −0.0004 [−0.0012, 0.0009] | 0 [0, 0] | 0 [0, 0] |
| `RobertWalraven-ESG` | −0.0010 [−0.0063, 0.0051] | −0.0018 [−0.0052, 0.0041] | 0 [0, 0] |
| `UMass-MechBayes` | +0.0048 [−0.0048, 0.0090] | −0.0002 [−0.0054, 0.0051] | 0 [0, 0] |

Source: `R/contrasts.parquet`, leaf mean, PI, lr 0.1, block 8. Intervals are paired
moving-block-bootstrap 95% percentile intervals. [Figure 1](figures/panel-leaves-v1-forest-leaf-mean-pi.pdf)
shows the same rows. Its purpose is to compare direction and uncertainty without collapsing
forecasters; the key observation is that every MinT and location-only interval crosses zero.
The supported interpretation is heterogeneity around a near-zero leaf effect, so the project
decision is not to claim H2 generalization.

The D24 cross-forecaster summaries put leaf and US behavior side by side. Values are mean
contrasts [95% forecaster-resampling interval], followed by the number of eight forecasters
with a negative estimate.

| Arm | Treatment | Leaf coverage deviation | Leaf scaled interval score | US coverage deviation | US scaled interval score |
|---|---|---:|---:|---:|---:|
| raw | `mint_shrink` | +0.0190 [0.0104, 0.0268] (0/8) | +0.0530 [0.0159, 0.0946] (1/8) | −0.0336 [−0.0725, −0.0015] (5/8) | −0.1382 [−0.2718, −0.0165] (6/8) |
| raw | `bottom_up` | 0 [0, 0] (0/8) | 0 [0, 0] (0/8) | −0.0291 [−0.0641, −0.0000] (5/8) | −0.0766 [−0.2378, 0.0648] (4/8) |
| raw | `mint_median` | +0.0015 [−0.0002, 0.0034] (1/8) | +0.0143 [−0.0012, 0.0316] (4/8) | −0.0214 [−0.0589, 0.0079] (5/8) | −0.1061 [−0.2210, −0.0106] (5/8) |
| PI | `mint_shrink` | −0.0005 [−0.0022, 0.0015] (6/8) | +0.0162 [−0.0131, 0.0586] (5/8) | −0.0068 [−0.0182, 0.0053] (4/8) | −0.1558 [−0.3423, 0.0233] (5/8) |
| PI | `bottom_up` | 0 [0, 0] (0/8) | 0 [0, 0] (0/8) | −0.0035 [−0.0200, 0.0100] (3/8) | −0.1620 [−0.3620, 0.0364] (5/8) |
| PI | `mint_median` | −0.0004 [−0.0012, 0.0005] (6/8) | +0.0216 [0.0037, 0.0409] (2/8) | +0.0011 [−0.0024, 0.0047] (2/8) | −0.1016 [−0.2310, 0.0071] (4/8) |

These summaries are descriptive. They show a level interaction: raw quantile-wise MinT is
consistently unfavorable at the leaves but favorable on average at the US node. That pattern
is compatible with reconciliation trading local fit for aggregate coherence; it does not
establish a causal mechanism.

[Figure 2](figures/panel-leaves-v1-forest-leaf-mean-raw.pdf) shows the raw leaf-mean forest.
Its purpose is to separate reconciliation from PI. Unlike Figure 1, every `mint_shrink`
coverage-deviation estimate is positive. The corresponding descriptive summary interval is
also above zero. This changes the interpretation from “PI reveals a MinT gain” to “PI largely
attenuates a raw leaf-level penalty.”

## 4. `mint_median`: the location-only mechanism

`mint_median` applies the causal MinT median shift to all three quantiles and therefore keeps
each node's raw base spread unchanged (D26). For the primary forecaster's raw leaf mean, its
scaled width is exactly the same as `none` (1.2514), while coverage deviation changes only
from 0.1434 to 0.1410 and scaled interval score from 2.2353 to 2.2325. Across forecasters,
the raw location-only mean contrasts are +0.0015 for coverage deviation and +0.0143 for
scaled interval score, with both descriptive intervals spanning zero.

Quantile-wise `mint_shrink` behaves differently: on the same primary raw leaves it changes
scaled width from 1.2514 to 1.1525, coverage deviation to 0.1557, and scaled interval score
to 2.2532. The contrast between these arms supports a limited mechanism statement: the broad
raw leaf penalty is associated with changing interval spread, not merely with shifting the
median. Location-only reconciliation still does not produce a repeatable leaf benefit.

At the US node, location-only raw reconciliation has a favorable descriptive mean scaled-score
contrast (−0.1061 [−0.2210, −0.0106]). This reproduces Phase 1's conclusion that location and
scale effects differ by hierarchy level; it does not rescue H1, whose endpoint is the leaves.

## 5. PI versus raw

PI-versus-raw was not a registered inferential contrast, so this comparison is descriptive.
Under `none`, PI lowers the eight-forecaster mean leaf coverage deviation by 0.0062, but only
3/8 forecasters improve; the mean is driven by larger improvements for `JHUAPL-Bucky` and
`RobertWalraven-ESG`. For the primary forecaster, PI instead worsens coverage deviation from
0.1434 to 0.1685.

The interval-score trade-off is uniform. Under `none`, PI raises mean scaled interval score
from 3.0678 to 8.5687 (+5.5009), with 0/8 forecasters improving. The corresponding PI-minus-raw
increases are +5.4641 for `mint_shrink` and +5.5082 for `mint_median`, again with 0/8 improving.
PI can move temporal coverage deviation in the desired direction for some forecasters, but it
does so with much worse scaled interval scores in this panel.

## 6. D28 Holm sensitivity analysis (post hoc)

At leaf mean, every `bottom_up − none` row is identically zero, including its bootstrap
replicates, because bottom-up leaves are the base leaves. The registered D24 families retain
these five rows: 15 members normally and 14 adjusted secondaries in the primary family. This
is valid but conservative. As specified in D28, a post-hoc sensitivity recomputes Holm after
dropping the five `bottom_up` rows, leaving 10 informative members per leaf-mean family. It
does not replace the registered adjustment, and the primary remains reported unadjusted.

At block 8, the registered informative rows contain 33 adjusted p-values below 0.05; the
10-member sensitivity contains 36. Only three rows newly cross 0.05, all under the raw arm and
all unfavorable to MinT:

| Forecaster | Metric, `mint_shrink − none` | Estimate | Registered Holm | 10-member Holm |
|---|---|---:|---:|---:|
| `CEID-Walk` | longest miss run | +0.5098 | 0.0605 | 0.0330 |
| `CEID-Walk` | scaled interval score | +0.0491 | 0.0605 | 0.0330 |
| `RobertWalraven-ESG` | scaled interval score | +0.1810 | 0.0715 | 0.0390 |

Within the primary family, the exploratory 10-member adjustment sends the H1 p-value to 1.0;
this has no confirmatory role because H1 was pre-registered as unadjusted. The sensitivity
adds no evidence of benefit and modestly strengthens the evidence of raw leaf-level harm.

## 7. Limitations and D28 caveats

- **Limited pre-launch exposure.** Before the official run, the final reviewer executed the
  deterministic registered pipeline in memory and saw some summary rows, but did not inspect
  the primary row or record/pass on any number. D28 disclosed this before the run. Subsequent
  code changes were non-numeric.
- **Different windows and scored leaves.** H2 averages forecasters evaluated over different
  66–126 week windows. `UMass-MechBayes` lacks `ny` and therefore averages 50 rather than 51
  endpoint leaves. D25 unscored nodes reflect submission gaps; the same gaps are absent from
  MinT's complete-row error history.
- **Degenerate family members.** The zero `bottom_up` leaf rows make registered Holm
  conservative. Section 6 is explicitly a post-hoc sensitivity, not a family redefinition.
- **Controller and truth caveats.** PI inherits the paper core's non-causal learning-rate
  scale (D15), while every arm within a comparison inherits it equally. Evaluation uses final
  revised truth, not as-of vintages. With block 4 shorter than the rolling window, rolling/run
  metrics straddle block joins (D17).
- **Scope.** The panel covers one disease, horizon and selected set of eligible forecasters.
  Cross-forecaster intervals summarize these eight forecasters and are descriptive; they do
  not define a population-level generalization test.

## 8. What changed and what should happen next

Phase 1 found that quantile-wise reconciliation widened the US interval and did not improve
the primary US endpoint. Phase 2 shows that moving to leaves and multiple forecasters does not
reveal a hidden general benefit: H1 is inconclusive, H2 PI estimates are near zero, and the raw
leaf effect of MinT is consistently adverse. The level interaction is the durable result:
reconciliation may improve aggregate US behavior while degrading leaf behavior, and PI can
mask much of that leaf penalty only by paying a large interval-score cost.

The current branch should therefore close as a negative result rather than expand the same
arm grid. A future study would need a method that models cross-series quantile dependence or
joint probabilistic coherence, plus as-of truth if strict real-time claims are desired. The
four deferred engineering issues in the SDD ledger are useful hardening work but do not change
this scientific conclusion.

## 9. Artifact and reproducibility index

- Registered config: `configs/panel_leaves.toml`.
- Immutable run artifacts: `results/panel-leaves-v1/`; manifest SHA
  `b78858260a8a4228f84c4240b665170a02ce9317`, input SHA-256
  `cb4dccfd21d6c08b1247aa9a7c6a8d93f7c373871526cfcb4860df884b54ac06`.
- Main figures:
  `docs/analysis/figures/panel-leaves-v1-forest-leaf-mean-pi.pdf`,
  `docs/analysis/figures/panel-leaves-v1-forest-leaf-mean-raw.pdf`, and
  `docs/analysis/figures/panel-leaves-v1-leaf-contrasts-pi.pdf`.
- Run command: `uv run python -m hcp.run configs/panel_leaves.toml`.
- Statistical definitions and protocol: `docs/protocol-decisions.md`, D20–D28.
