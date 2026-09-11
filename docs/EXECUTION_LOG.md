# Execution log

## 2026-09-10

- Researched published methods, rewrote AGENTS.md, and wrote seven execution prompts.
- User authorized execution and emphasized independently trained ML as a portfolio objective.
- Existing user modifications include team features, simulator, a saved model, walk-forward scripts/results, .gitignore, and claude.txt; preserve them. Some legacy tests overwrite models and must not run unchanged.
- Cached player logs exist for 2015-16 through 2024-25; schema and coverage validation pending.
- Completed experimental run: `data/processed/player_runs/v1_20260910T200443Z_6a86c4/`.

## Prompt 1 — Baseline and leakage audit: executed

- Corrected season labels (including October 2020), offensive rebound denominators, neutral-site orientation/Elo advantage, chronological legacy training split, unique saved-model paths, and schedule/model feature-schema propagation.
- Found five 2024-25 games with both teams marked away. The initial strict one-game merge rejected these; neutral-site handling repaired that failure. Partial failed runs remain separate from the successful run.
- Existing raw/processed caches and saved legacy models were not rebuilt or overwritten. Corrected experimental features are built in memory. The legacy cached feature file now needs an explicit rebuild before legacy training; a clear schema error prevents using it silently.
- Historical season simulations consume each game's historical pre-game state. That is a rolling replay, not a preseason forecast. The new fixed-cutoff path is separate.
- Baselines trained on seasons before 2023-24 for validation, then before 2024-25 for final testing. Both test sets contain all 1,230 games, one row per game. No tuning used the final test outcomes.

## Prompt 2 — Data contract: executed

- Validated 254,513 player-game rows and 23,958 team-game rows across 2015-16 through 2024-25; no duplicate player-game rows and exact scoring reconciliation. Maximum rounded-minute discrepancy was four minutes per team-game.
- Cached static NBA birth-date metadata with source response files and retrieval timestamps; complete player-log birth-date coverage obtained.
- Original game-log retrieval times are unavailable and explicitly recorded as unknown. Coaching roster responses lack effective assignment dates. Transaction/injury archives and possession data are absent.

## Prompts 3–4 — Player and game ML: executed

- Built 374,683 causal candidate rows. Rosters are last-observed estimates, with missing new arrivals exposed as a limitation; actual current participants never construct pre-game rosters.
- Trained our own LightGBM minutes and scoring-rate models, generating annual out-of-fold predictions before downstream game training.
- Trained team-only, roster-only, and hybrid logistic/LightGBM models, plus separate minutes, scoring, and interaction ablations. Versioned estimators, features, cutoffs, source hashes, predictions, and environment lock are saved.
- On 2024-25 candidates, minutes MAE improved from 7.489 to 5.827 before team-minute normalization. Conditional scoring RMSE improved from 7.484 to 7.423 points per 36.
- Validation selected `hybrid_development_ml`. Test log loss was 0.601090 versus team-linear 0.602622. Weekly block-bootstrap interval for the paired difference was [-0.006087, 0.002808], including no improvement. Keep experimental.
- Rolling expected-win MAE worsened slightly: 3.926 versus team baseline 3.784. Strong/weak-team compression remains; improved game probabilities did not fix standings bias.

## Prompt 5 — Development and fit: executed as experiments

- Trained independent age-aware next-season points/rebounds/assists per-36 and season-minute regressors. Sources cover every age row. Rate targets condition on >=300 next-season minutes; exits retain zero minutes but undefined skill targets.
- Scoring RMSE improved on both validation and test; test was 2.542 versus 2.693 for last-season rates. Other targets were mixed. Annual models are saved separately and are not yet a validated seasonal transition model inside the simulator.
- Roster-profile interaction ablation did not improve validation over the selected hybrid. No learned lineup-level chemistry or causal trade effect is claimed.
- Trade scenario function/CLI reallocates both teams' minutes and reports conditional expected-win changes. This is an experimental what-if interface, not a backtested trade recommendation system.

## Prompt 6 — Coaching: data gate blocked

- Implemented dated assignment validation, known-at filtering, and a regularized roster-conditioned coaching estimator; date handling is tested.
- Public NBA roster metadata was successfully retrieved, but has no assignment effective dates. A search also located season-level coaching datasets; those do not establish game-date assignments. No coach model was fit to invented or undated assignments.
- Required next input: verified dated head-coach assignments for the training/evaluation seasons, with source/availability evidence. RAPM likewise remains contingent on validated possession/lineup data.

## Prompt 7 — Fixed-cutoff integration: experimental demonstration executed

- Added a seeded simulator using frozen cutoff profiles, supplied schedule, current wins, and optional persistent per-player scoring uncertainty. Nonzero uncertainty is explicitly a sensitivity assumption, not calibrated.
- A 2025-01-01 scenario simulated 746 remaining games in 2,000 trials. Standings MAE was 5.017 wins; nominal 90% outcome-only intervals covered 60% of teams. No future actual stats or transactions entered the snapshot.
- Final historical schedule dates were supplied as scenario inputs; this is not a verified archive of the schedule published at the cutoff. One scenario does not establish preseason validity. Do not promote to production.

## Commands and checks

Use the project-local Python 3.12 environment; `requirements-player.lock.txt` records the installed versions separately from legacy requirements.

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m scripts.run_player_experiments
.venv/Scripts/python.exe -m scripts.pull_player_birth_dates
.venv/Scripts/python.exe -m scripts.run_development_experiment --run-dir data/processed/player_runs/v1_20260910T200443Z_6a86c4 --birth-dates data/raw/player_metadata/birth_dates_20260910T200811Z.csv
.venv/Scripts/python.exe -m scripts.validate_player_results --run-dir data/processed/player_runs/v1_20260910T200443Z_6a86c4
.venv/Scripts/python.exe -m scripts.run_cutoff_scenario --run-dir data/processed/player_runs/v1_20260910T200443Z_6a86c4 --cutoff 2025-01-01 --n-sims 2000
```

Fourteen focused tests passed at this point, including future-mutation, unusual seasons, neutral games, dated context, trade membership, rotation feasibility, persistent uncertainty, reproducibility, and league-win conservation. The unchanged legacy suite was not run because it retrains/overwrites saved models and assumes obsolete random-split behavior.

Final verification: source compilation passed, all 14 focused tests passed, and whitespace checks passed. A second real-data scenario run after adding source-date validation reproduced the same seeded standings metrics in `scenario_20260910T201459Z/`. README and MODEL_CARD.md now distinguish implemented ML from experimental and data-blocked work. No legacy trained model was overwritten during this execution.

## Follow-up — age-aware in-season player forecasts: executed

- Added strict static birth-date joins to the causal candidate data. `AGE` and `AGE_SQUARED` are available only when a validated birth-date contract is supplied; missing coverage fails rather than defaulting an unknown player to an average age.
- The no-age baseline re-ran unchanged, confirming a matched comparison. In the age-aware 2024-25 run, minutes MAE was 5.825 versus 5.827 without age and conditional scoring RMSE was 7.423 in both cases. Game-level results were not better: the development hybrid test log loss was 0.601159 versus 0.601090 without age.
- The paired player-versus-team improvement remains inconclusive: -0.001463 log loss with approximate weekly-block interval [-0.006118, 0.002934]. Age is retained as an optional, documented experiment, not selected or promoted.
- `15` focused tests passed after this change, including static-age coverage and implausible/missing-age validation.

## Follow-up — learned player-residual uncertainty: executed

- Added a separate scoring-residual regressor trained only on earlier out-of-fold player forecasts. It predicts each active player's absolute points-per-36 error and converts it to a normal residual standard deviation; no final-test targets enter the cutoff forecast bundle.
- The fixed-cutoff simulator now draws one residual per player per trial and retains it across that player's remaining games. This represents persistent player-level forecasting uncertainty, rather than repeatedly adding independent game noise.
- Matched 2,000-trial scenarios at the 2025-01-01 cutoff used the same 746-game supplied schedule and seed. Outcome-only intervals had 60.0% empirical coverage for nominal 90% team-win intervals (MAE 5.024). Learned player-residual intervals had 86.7% coverage (MAE 4.851); for example, OKC's simulated standard deviation widened from 2.94 to 4.08 wins.
- This is one historical cutoff, so the result is encouraging calibration evidence rather than a production promotion. The next evaluation should repeat predeclared cutoffs without tuning the residual model to any of them.
- `16` focused tests passed, including the learned-uncertainty source contract and interval-widening behavior.

## Prompt 1 — multi-cutoff calibration backtest: implementation committed; evaluation pending

- Added `scripts/run_multi_cutoff_backtest.py`, a fixed-cutoff runner covering the predeclared December 1, January 1, February 1, and March 1 cutoffs for 2018-19 through 2024-25. It saves per-cutoff standings, game probabilities, calibration bins, source hashes, seeds, and protocol metadata in an isolated run directory.
- The runner refits cutoff-eligible player/game models and validates the chronological OOF upstream artifact against strictly earlier candidate source dates. Historical remaining schedules are labelled retrospective scenario inputs.
- Added seeded 50/80/90% standings interval quantiles plus an explicit team-only learned-uncertainty control. Focused simulator and player pipeline checks passed (10 tests).
- The first complete 28-cutoff evaluation has not yet finished: causal roster reconstruction and the learned-uncertainty batches are computationally expensive locally. No metrics, model-card promotion decision, or claim of generalized coverage has been made. Partial run directories are retained as failed/incomplete artifacts and are not inputs to any conclusion.
