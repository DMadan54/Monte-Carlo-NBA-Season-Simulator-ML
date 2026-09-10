# Player-aware NBA forecasting — experimental model card

## What I developed

An independently trained, multi-stage NBA forecasting pipeline: causal player profiles, LightGBM playing-time and scoring forecasts, chronological out-of-fold feature generation, regularized game classifiers, age-aware next-season regressors, and seeded Monte Carlo roster scenarios. Standard libraries provide learning algorithms; the project work is in the data contracts, targets, feature design, model integration, evaluation, and leakage controls. This is not a claim to have invented LightGBM, RAPM, or a new state-of-the-art basketball metric.

Published DARKO/EPM/RAPM methods informed the design. No proprietary player ratings were used as inputs. Box-score profiles do not isolate defensive impact or context-free talent. Research links are in [PLAYER_MODEL_RESEARCH.md](PLAYER_MODEL_RESEARCH.md).

## Data and forecasts

- NBA team/player logs: 2015-16 to 2024-25, 11,979 games and 254,513 player appearances. Source caches were already local; their original retrieval timestamps are unknown. Source file hashes are saved.
- Static birth dates: retrieved from NBA endpoints with response caches and timestamps. All player-log IDs have coverage.
- Player inputs: past decayed counts per 36 minutes, shrunk shooting efficiency, historical minutes, participation gaps, and recent scoring change. The candidate roster uses prior appearances only, including a 15-team-game expiry rule.
- Learned next-game outputs: regulation-equivalent minutes for candidates (including zeros for absence), plus points per 36 conditional on >=5 actual minutes. The rate target is evaluated separately from minutes.
- Game inputs: both teams' historical form/Elo, roster summaries weighted by projected minutes, rest, and neutral-site context. IDs are join keys, not memorized predictors.
- Annual outputs: next-season points, rebounds, and assists per 36, plus total minutes. Rate evaluation conditions on >=300 future-season minutes and is therefore subject to survivor selection.

## Validation

The game variants were fixed before test evaluation. Validation is 2023-24; final testing is 2024-25. All 1,230 games in each season are retained. Downstream training begins in 2017-18 after player-model warmup. Player forecasts for each season are trained strictly on earlier seasons. Preprocessing is fit within the downstream training split. Forecast inputs exclude outcomes on the same date.

The validation-selected model is `hybrid_development_ml`: learned player minutes/scoring projections feeding a regularized logistic game classifier. Logistic regression is itself ML; nonlinear LightGBM game alternatives performed worse in this experiment.

Test results:

- Team-only logistic baseline: log loss 0.602622, Brier 0.208087, accuracy 67.48%.
- Selected player hybrid: log loss 0.601090, Brier 0.207277, accuracy 67.89%.
- Paired log-loss difference: -0.001532; approximate weekly-block 95% interval [-0.006087, 0.002808]. The improvement is not statistically conclusive under this check.
- Player minutes MAE: 7.489 baseline versus 5.827 ML, on 40,970 candidate rows, before rotation normalization. New/unobserved arrivals are not fully covered by this candidate-based target.
- Next-season scoring RMSE: 2.693 baseline versus 2.542 ML points per 36, weighted by minutes across 374 qualifying players. Rebounds/assists/season-minute results were mixed across validation/test.
- Adding roster-fit interactions did not improve validation. Those features are not in the selected model.

The standings objective is not solved: summing rolling game probabilities produced 3.926-win MAE for the hybrid versus 3.784 for the team baseline, and strong/weak-team compression persists. These are rolling retrospective aggregates, not preseason forecasts.

## Simulation and trades

A fixed-cutoff 2025-01-01 demonstration used only earlier performance, estimated rosters, and the final historical remaining schedule as a scenario input. In 2,000 trials over 746 games, MAE was 5.017 wins; nominal 90% outcome-only intervals covered 60% of teams. This undercoverage argues for better roster/injury/development uncertainty, not arbitrary probability stretching.

The simulator supports persistent per-player scoring perturbations, but their magnitude is an explicit sensitivity parameter, not a learned calibrated uncertainty estimate. A trade scenario moves a known player, reallocates minutes on both teams, and recalculates expected wins. It retains historical team context and has not demonstrated causal or reliable destination-specific player effects.

## Limitations and next evidence needed

- Last-observed rosters miss a transaction before the new player's first appearance; accurate dated transactions and availability are the highest-priority data improvement.
- Date-batched predictions deliberately ignore same-day news/results. Actual schedule rescheduling knowledge at historical cutoffs is not archived here.
- No independently estimated RAPM or lineup model yet: possession/lineup data are missing.
- Coaching input validation and an estimator exist, but historical effective assignments are missing, so no real coach estimates are claimed.
- Annual development models remain separate experiments, not yet a validated evolution process driving full-season forecasts.
- One final test season and one cutoff scenario are limited evidence. Keep the test results frozen; additional model choices require new validation evidence, not repeated tuning to 2024-25.
- Legacy caches and saved models were preserved. This model has not replaced the default production artifact.

## Reproduction and artifacts

Use `requirements-player.lock.txt` and the commands in [EXECUTION_LOG.md](EXECUTION_LOG.md). The local run is `data/processed/player_runs/v1_20260910T200443Z_6a86c4/`; it contains model bundles, per-game predictions, player forecasts, metrics, calibration bins, source hashes, and environment versions. Large caches/models are ignored by git; commit source code and documentation, not these local datasets.

A defensible portfolio description: “Developed a player-aware NBA forecasting pipeline with LightGBM player projections, chronological out-of-fold validation, and reproducible Monte Carlo roster scenarios; evaluated improvements and failure modes against team-only baselines.”
