# Monte Carlo NBA Season ML Simulator

A Python portfolio project developing independently trained player and game models for NBA forecasting. The pipeline combines causal player profiles, learned minutes/scoring forecasts, chronological validation, and reproducible Monte Carlo roster scenarios.

## Implemented

- Cached NBA team/player logs and validated player-game data contracts.
- Team rolling form, Four Factors, and Elo with explicit season labels and neutral-site handling.
- LightGBM player minutes and scoring-rate forecasts, generated out-of-fold by season.
- Team-only, roster-only, and hybrid game classifiers with matched chronological evaluation.
- Age-aware next-season player production regressors.
- Fixed-cutoff season scenarios with learned persistent player-residual uncertainty, plus explicit trade scenarios.
- Tests for leakage, dated context, roster changes, feasible minutes, and simulation conservation.

Coaching input validation and an estimator exist, but dated coaching assignments are missing, so no real coaching estimates have been trained. Possession-level RAPM, detailed lineup compatibility, calibrated roster/injury uncertainty, awards prediction, and the dashboard remain future work.

## Research and results

Read [research decisions](docs/PLAYER_MODEL_RESEARCH.md), [ordered implementation prompts](docs/IMPLEMENTATION_PROMPTS.md), [execution record](docs/EXECUTION_LOG.md), and the [model card](docs/MODEL_CARD.md).

The first player-aware experiment used 254,513 player appearances across ten seasons. On the held-out 2024-25 season, the validation-selected hybrid achieved log loss 0.6011 versus 0.6026 for the team-only logistic baseline. This small improvement was not statistically conclusive in an approximate weekly-block check. Rolling standings error did not improve. The age-aware next-season scoring model improved its held-out error, while other development targets had mixed results.

These are independently trained models using standard ML libraries, not imported proprietary player ratings. The project does not claim novel learning algorithms or state-of-the-art NBA accuracy.

## Run the experimental pipeline

Use Python 3.12 and a project-local virtual environment. The current workspace already has `.venv`; `requirements-player.lock.txt` captures its tested dependencies separately from the legacy `requirements.txt`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-player.lock.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m scripts.run_player_experiments
```

The runner requires cached team and player parquet logs in `data/raw/`. It validates them, rebuilds corrected team features in memory, and creates a unique directory under `data/processed/player_runs/`. It never replaces a legacy saved model. The original game-log cache retrieval timestamps are unknown and recorded as such.

Pass `--birth-dates BIRTH_DATE_CSV` to test static age and age-squared features in the player minutes/scoring models. This optional variant was evaluated but did not improve the first held-out game test, so it is not the selected model.

For age-based forecasts, retrieve static birth dates, then supply the printed CSV and experiment directory:

```powershell
.venv/Scripts/python.exe -m scripts.pull_player_birth_dates
.venv/Scripts/python.exe -m scripts.run_development_experiment --run-dir RUN_DIRECTORY --birth-dates BIRTH_DATE_CSV
.venv/Scripts/python.exe -m scripts.validate_player_results --run-dir RUN_DIRECTORY
.venv/Scripts/python.exe -m scripts.run_cutoff_scenario --run-dir RUN_DIRECTORY --cutoff 2025-01-01 --season 2024-25 --n-sims 2000
```

`RUN_DIRECTORY` and `BIRTH_DATE_CSV` are placeholders for actual output paths. Add `--learned-player-uncertainty` to propagate residual uncertainty learned from earlier player forecasts, or `--player-points-sd` for a separate sensitivity assumption. A scenario can add `--trade PLAYER_ID DESTINATION_TEAM_ID`, repeated for multiple moves. IDs must exist in the cutoff snapshot. This is a conditional what-if calculation, not a validated trade recommendation.

## Forecast interpretation

Rolling historical evaluation uses each game's past information as it becomes available. It is not a preseason forecast. Fixed-cutoff scenarios freeze player/team profiles and use a supplied remaining schedule; the historical demonstration uses the final played calendar, not an archived as-published schedule. No future actual performance or transactions enter its cutoff snapshot.

The baseline January 2025 scenario's nominal 90% outcome-only intervals covered only 60% of teams. Learned persistent player-residual uncertainty raised coverage to 86.7% in the same single-cutoff experiment. Repeat-cutoff evaluation, better roster data, and injury inputs are still required before making confident season-level claims.

## Scenario dashboard

With the local versioned artifact available, run the read-only dashboard:

```powershell
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

It loads a selected scenario directory and displays simulated standings, win intervals, actual wins, assumptions, and limitations. It never retrains or changes model artifacts.

## Structure

- `src/ingest/`: source clients, caches, and validated contracts.
- `src/features/`: team features and causal player profiles.
- `src/models/`: player forecasts, game models, annual/coaching support, and simulation.
- `scripts/`: experiment, metadata, development, validation, and scenario entry points.
- `tests/`: focused automated regression checks.
- `docs/`: evidence, prompts, execution record, and model card.
- `data/`: local caches and versioned artifacts; large new caches/models are ignored by git.

Legacy pipeline tests can retrain and overwrite existing models and assume old validation behavior. Use the focused `tests/` suite for this work. Legacy cached team features need an explicit rebuild before legacy training after the schema corrections.
