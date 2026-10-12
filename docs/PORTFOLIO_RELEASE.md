# Portfolio release: evidence and CV wording

## Scope and status

This is a completed **experimental portfolio project** demonstrating independently trained player and game models, chronological evaluation, Monte Carlo regular-season standings, and conditional roster-change scenarios. It is not a validated live/preseason prediction service. The player-aware model has not established a reliable standings advantage over the team-only baseline. Coaching, injury/availability, and dated transaction inputs are incomplete; no causal trade or player impact estimate is claimed. The 2026–27 full-season forecast remains blocked by 30 unassigned schedule games in the saved official schedule snapshot.

The dashboard includes a **2025-01-01 historical fixed-cutoff standings scenario** and a **2025–26 MVP awards holdout**. The standings schedule is the final played calendar supplied after the season, not an archived schedule known at the cutoff. The standings report and rows are byte-for-byte copies of the source run; all bundled MVP rows and source hashes are checked by `scripts.verify_portfolio_evidence`. A reviewer can inspect both sections without downloading the training caches.

## Numbers independently checked from saved artifacts

| Claim | Verified value | Source and interpretation |
| --- | ---: | --- |
| Historical training/evaluation cache | 10 seasons, 11,979 games, 254,513 player appearances | `data/processed/player_runs/v1_20260910T200443Z_6a86c4/data_audit.json`; original cache retrieval dates are unknown. |
| Held-out 2024–25 game log loss | 0.601090 player-aware vs 0.602622 team-only | 1,230 game predictions per model; the difference is small and its approximate weekly-block interval includes zero. |
| Held-out 2024–25 player minutes MAE | 5.827 vs 7.489 minutes for the simple baseline | 40,970 candidate player-game rows; about 22% lower MAE, before rotation normalization. It excludes players absent from the candidate roster. |
| Historical fixed-cutoff example | 746 remaining games, 2,000 seeded trials, 4.851-win standings MAE | 2025-01-01 retrospective scenario; nominal 90% intervals covered 26/30 teams (86.7%). One cutoff only. |
| Multi-cutoff standings study | 28 cutoffs across seven seasons; 3.897-win MAE | Player-aware learned-uncertainty model; team-only control was 3.910 wins, effectively tied. Nominal 90% coverage was 80.4%, below target. Only 50 trials per cutoff. |
| New 2025–26 MVP test | Learned model picked 0/3 cutoffs; baseline 3/3 | Three cutoff dates share one award outcome. This is a documented failed independent holdout, not three independent wins for the baseline. |

Run `.venv/Scripts/python.exe -m scripts.verify_portfolio_evidence` from the repository root to recheck the table against the local versioned experiment runs. The verifier also checks the bundled example, game prediction rows, scenario win conservation, multi-cutoff aggregates, and MVP coverage. The large training caches and experiment runs are local and ignored by Git; a fresh clone can run the bundled dashboard but cannot rerun training without obtaining those inputs. Earlier repository history contains some tracked data/model files; this release adds only the small example, code, tests, and documentation.

## CV wording

**Recommended two bullets:**

- Built an independently trained NBA forecasting pipeline across 10 seasons (11,979 games; 254,513 player appearances), using player minutes/scoring forecasts, chronological validation, and Monte Carlo standings simulations.
- Reduced held-out player minutes MAE from 7.489 to 5.827 minutes (22%) on 40,970 player-game candidates; evaluated a 1,230-game test season and 28 historical standings cutoffs against team-only baselines, documenting mixed downstream results.

**If space allows, add:**

- Developed a reproducible 2,000-trial historical standings scenario and a read-only dashboard; the scenario achieved 4.851-win MAE and 26/30 coverage for nominal 90% intervals at one January 2025 cutoff.

For an interview, explain that the 0.601090 versus 0.602622 game log-loss difference was inconclusive and the seven-season standings MAE was effectively tied with the team-only control. The 22% minutes improvement is the clearest measured player-model gain. Do not describe the fixed-cutoff scenario as a validated preseason forecast.

## Reviewer walkthrough

1. Install `requirements-player.lock.txt` in Python 3.12.
2. Run `.venv/Scripts/python.exe -m streamlit run dashboard/app.py`. The bundled standings and MVP examples load by default; the sidebar accepts another versioned artifact directory for either tab.
3. Read [MODEL_CARD.md](MODEL_CARD.md) for methods and limitations, and [EXECUTION_LOG.md](EXECUTION_LOG.md) for chronological experiment history.
4. If the local ignored runs are present, run `.venv/Scripts/python.exe -m scripts.verify_portfolio_evidence` and the focused `tests/test_portfolio_release.py` tests.
