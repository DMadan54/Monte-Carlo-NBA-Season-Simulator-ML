# Multi-cutoff standings-bias diagnosis

## Evidence

Source: `data/processed/multi_cutoff_runs/v1_20260911T110925Z_9a9b27/`, 28 fixed historical cutoffs from 2018-19 through 2024-25. Each row below is a team/cutoff observation from the player-aware hybrid with learned uncertainty.

- Mean standings MAE: 3.897 wins; rank correlation: 0.850.
- Learned uncertainty improved nominal-90% coverage from 70.1% to 80.4%, but it changes intervals rather than the central forecast.
- The team-linear control has slightly lower MAE (3.910 versus 3.897 is effectively tied) and the player-aware hybrid does not establish a standings advantage.

## Strength-decile pattern

| Actual-final-win decile | Mean simulated-minus-actual wins | Absolute error |
|---:|---:|---:|
| 1 (lowest) | +3.07 | 4.11 |
| 2 | +3.15 | 4.77 |
| 3 | +3.73 | 4.39 |
| 4 | +0.89 | 3.51 |
| 5 | -0.72 | 3.05 |
| 6 | -1.06 | 4.07 |
| 7 | -1.53 | 4.13 |
| 8 | -2.13 | 3.12 |
| 9 | -2.40 | 3.87 |
| 10 (highest) | -3.02 | 3.95 |

## Diagnosis

The verified pattern is **standings compression**: the central forecasts overestimate weak teams and underestimate strong teams. This is not explained by game-outcome sampling, because mean wins are calculated across many trials; sampling mainly affects intervals. Learned player uncertainty improves coverage but does not correct the mean-strength compression.

Likely contributors, not proven causal explanations:

1. The regularized game classifier shrinks extreme game probabilities toward 50%.
2. Last-observed roster estimates omit some roster changes and availability differences.
3. Player aggregate inputs may not add enough stable information beyond team history to change long-horizon central forecasts.

## Next experiment

Do not tune against this final aggregate. On pre-2024-25 validation cutoffs, compare a frozen probability-calibration layer (isotonic and Platt candidates) against the uncalibrated classifier. Promote only if validation improves calibration and does not materially worsen log loss or standings MAE; evaluate the selected method once on 2024-25 cutoffs.
