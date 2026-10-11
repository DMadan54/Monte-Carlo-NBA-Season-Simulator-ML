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
- A dated age/age-squared variant of the in-season minutes and scoring models was also tested. It did not improve the held-out game model: development-hybrid log loss was 0.601159 with age versus 0.601090 without. The age input remains optional for future experiments, not selected.

The standings objective is not solved: summing rolling game probabilities produced 3.926-win MAE for the hybrid versus 3.784 for the team baseline, and strong/weak-team compression persists. These are rolling retrospective aggregates, not preseason forecasts.

## Simulation and trades

A fixed-cutoff 2025-01-01 demonstration used only earlier performance, estimated rosters, and the final historical remaining schedule as a scenario input. In matched 2,000-trial, 746-game scenarios, nominal 90% outcome-only intervals covered 60.0% of teams (MAE 5.024). A player-residual model trained on earlier out-of-fold scoring errors raised coverage to 86.7% (MAE 4.851). It samples one residual per player per trial and retains it over the remaining season. This is promising calibration evidence from one cutoff, not a production claim; it needs repeated predeclared cutoff evaluation.

An additional manually selected player standard deviation remains available only as a sensitivity analysis. A trade scenario moves a known player, reallocates minutes on both teams, and recalculates expected wins. It retains historical team context and has not demonstrated causal or reliable destination-specific player effects.

A multi-cutoff runner evaluated all December/January/February/March cutoffs in 2018-19 through 2024-25 as retrospective final-schedule scenarios. With 50 fixed trials per cutoff, hybrid learned residual uncertainty increased mean 90% coverage from 70.1% to 80.4% and improved standings MAE from 3.940 to 3.897 wins. This passes the predeclared uncertainty gate but remains experimental: coverage is below nominal, and the hybrid does not beat the team-linear uncertainty control on standings MAE (3.897 versus 3.910). Fixed-cutoff game Brier/log loss (0.3539/1.5506) are not comparable to the rolling next-game test and are materially weaker.

## Limitations and next evidence needed

- Last-observed rosters miss a transaction before the new player's first appearance; accurate dated transactions and availability are the highest-priority data improvement.
- Date-batched predictions deliberately ignore same-day news/results. Actual schedule rescheduling knowledge at historical cutoffs is not archived here.
- No independently estimated RAPM or lineup model yet: possession/lineup data are missing.
- Coaching input validation and an estimator exist, but historical effective assignments are missing, so no real coach estimates are claimed.
- Annual development models remain separate experiments, not yet a validated evolution process driving full-season forecasts.
- The preseason state-transition experiment improved 2024-25 minute MAE and scoring-rate RMSE, but substantially over-predicted >=300-minute participation. It remains separate from the simulator.
- Interpretable roster-compatibility interactions are available only for conditional scenarios. Their validation ablation did not improve the additive hybrid, so they are not used by the simulator.
- One final test season and one cutoff scenario are limited evidence. Keep the test results frozen; additional model choices require new validation evidence, not repeated tuning to 2024-25.
- Legacy caches and saved models were preserved. This model has not replaced the default production artifact.

## Reproduction and artifacts

Use `requirements-player.lock.txt` and the commands in [EXECUTION_LOG.md](EXECUTION_LOG.md). The local baseline run is `data/processed/player_runs/v1_20260910T200443Z_6a86c4/`; the learned-uncertainty rerun is `data/processed/player_runs/v1_20260910T215749Z_35f88a/`. They contain model bundles, per-game predictions, player forecasts, metrics, calibration bins, source hashes, and environment versions. Large caches/models are ignored by git; commit source code and documentation, not these local datasets.

The age-aware follow-up is `data/processed/player_runs/v1_20260910T215005Z_f257bf/`. Run it with `--birth-dates data/raw/player_metadata/birth_dates_20260910T200811Z.csv`; that locally cached source is intentionally ignored by git.

A defensible portfolio description: “Developed a player-aware NBA forecasting pipeline with LightGBM player projections, chronological out-of-fold validation, and reproducible Monte Carlo roster scenarios; evaluated improvements and failure modes against team-only baselines.”

## MVP voting experiment (2026-10-11)

This separate retrospective experiment matches a pinned secondary MVP voting table to cached NBA player IDs. It uses all 5,386 player-seasons with at least one logged regular-season game in 2015–16 through 2024–25, including 117 mapped vote recipients and 5,269 zero-vote candidates. All vote recipients matched; three Jimmy Butler rows use a reviewed suffix alias, and PHO/PHX and BRK/BKN team abbreviations are explicitly reconciled. The mapping, candidate table, source hashes, and coverage are saved in `data/processed/mvp_runs/v1_20261011T165232Z_cb62f5bd/`.

The target is each player's vote points divided by the season's maximum. Original work here is the identity/coverage contract, season-end feature assembly from cached game logs, and forward-only evaluation. The model is standard median imputation, standardization, and ridge regression; its penalty is selected using earlier-season validation folds when available. A fixed, transparent scoring baseline uses points per game, team win fraction, and games played. Holdouts are seasons ending 2018–2025; each model sees only earlier seasons' voting labels when fitting weights. Test-season box scores are season-end features, so this evaluates retrospective vote explanation, **not** preseason or in-season award forecasting.

Across eight held-out seasons, ridge selected the winner in 2/8 versus 3/8 for the scoring baseline. Mean within-season Spearman rank correlation was 0.257 versus 0.248 across all candidates; among vote recipients it was 0.582 versus 0.710. Vote-share MAE among vote recipients was 0.203 versus 0.380; across all candidates it was 0.0139 versus 0.1134, but that all-candidate metric is dominated by zero-vote players. The baseline's share scaling is a simple heuristic rather than a fitted share model. Ridge predictions never exceeded 0.25, while observed shares reached much higher values; pooled share-bin calibration shows overprediction in low bins and severe underprediction in the highest populated bin. It has not learned to represent top candidates' shares well. The chosen ridge penalty moved from 100 to 0.1 over folds, and correlated team wins/win fraction and games/minutes weights make individual coefficients unstable to interpret. Do not promote this model.

Limitations: the source is a secondary transcription; historical MVP eligibility rules are not encoded, so some zero-vote candidates were ineligible. The team-strength feature uses the team where a traded player logged the most minutes, not a weighted team context. Box-score plus/minus is noisy. There are only two training seasons for the first holdout and eight test winners overall. The voting data has no zero-vote player identities, so zero labels depend on complete game-log coverage and correct identity joins. A subsequent award-forecast model needs cutoff-frozen features and prospective evaluation.

### March 1 MVP forecast follow-up

The separate `data/processed/mvp_runs/v2_cutoff_20261011T170512Z_cb62f5bd/` run freezes player and team box scores before March 1 of each season, then forecasts the eventual MVP. It uses 2015–16 through 2024–25 cached logs. Every eventual vote recipient, including every winner, had appeared by that cutoff in all ten seasons. Trades use the player's most recently observed team as of the cutoff. The learned model is balanced logistic regression with median imputation and standardized features; its regularization is selected on earlier seasons only. The baseline combines points per game, team win fraction, and appearance fraction, with its probability scale fitted using training seasons only. The eight seasons ending 2018–2025 are held out one at a time.

The learned model ranked the eventual winner in its top three in **7/8** seasons versus **5/8** for the baseline; mean winner rank was 2.25 versus 3.75. It picked the exact winner in **2/8** versus **3/8**. Mean winner log loss was 1.413 versus 1.973, but multiclass Brier score was 0.719 versus 0.705 (lower is better). Among eventual vote recipients, mean Spearman rank correlation was 0.624 versus 0.543. Average top-pick probability was 0.307 for the model and 0.506 for the baseline, compared with hit rates of 0.250 and 0.375; eight outcomes are far too few for a firm calibration claim. Ranking has improved, but the model is not ready for promotion.

From 2023–24 onward, a cutoff-known flag records whether a player can still reach the normal 65-game path under the [NBA collective bargaining summary](https://cms.nba.com/wp-content/uploads/sites/4/2024/11/2024-25-CBA-101.pdf): games normally require at least 20 minutes, with up to two games of 15–19 minutes counting. It assumes 82 scheduled games and never uses future actual games. At the March 1 cutoffs, 399 candidates in 2023–24 and 398 in 2024–25 lacked a normal path; no eventual vote recipient did. Injury and grievance exceptions are not in the cache, so the flag does not hard-exclude anyone. Rounded minutes can misclassify threshold games. The first year with the rule offers no prior rule-era training examples, and its learned coefficient remains zero.

This is still a retrospective fixed-date test. March 1 represents a different fraction of the schedule in unusual seasons such as 2020–21. The model has no archived injury news, transactions before first appearance, live roster feed, or exceptional eligibility decisions. It predicts winner probabilities from observed player performances; it does not project individual future minutes or season-end box scores. Further cutoff dates and more seasons are needed before treating the probabilities as deployable.

### January–March cutoff validation

`data/processed/mvp_runs/v3_multi_cutoff_20261011T174353Z_cb62f5bd/` repeats the forward-only evaluation at January 1, February 1, and March 1, retraining and calibrating within earlier seasons separately at each date. Every eventual vote recipient and winner is present in the candidate set at all three dates. The same eight seasons (ending 2018–2025) are held out at each date; the 24 predictions represent **eight independent award outcomes**, not 24 independent seasons.

| Cutoff | Model winner | Baseline winner | Model top-three | Baseline top-three | Model log loss | Baseline log loss | Model Brier | Baseline Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| January 1 | 3/8 | 2/8 | 7/8 | 4/8 | 2.092 | 3.427 | 0.857 | 0.895 |
| February 1 | 3/8 | 3/8 | 7/8 | 5/8 | 1.400 | 1.976 | 0.676 | 0.728 |
| March 1 | 2/8 | 3/8 | 7/8 | 5/8 | 1.413 | 1.973 | 0.719 | 0.705 |

The model's paired winner log loss improved by 0.692 from January to February (season-block bootstrap 95% interval for February minus January: -0.987 to -0.401). February to March changed by +0.013 (interval -0.204 to +0.180), so there is no evidence of further improvement at the later date. Model-minus-baseline log-loss intervals include zero at all three cutoffs; superiority is not established. February is the strongest of these three dates on Brier, while March is worse than the baseline on that metric. This supports using the model as an experimental candidate ranker, not a calibrated winner forecast.

The delayed 2020–21 start makes calendar comparisons uneven: January 1 followed only about five games per team that year, versus roughly 32–37 in other seasons. A separate seven-season sensitivity table excluding 2020–21 is saved; the primary results retain it. The January model still has lower mean log loss than the baseline in that sensitivity (2.118 versus 2.602), but higher Brier (0.863 versus 0.795). More historical seasons and independent cutoff dates are needed before revising the model or claiming a stable advantage.
