# Reproduce the awards workflow

Use the project Python environment with `requirements-player.lock.txt` installed.
The commands below use `python`; activate that environment first. Source collection
also requires `pip install -r requirements-awards.txt`. Do not install into the
system Python. The historical raw logs and trained bundles are local caches and
are not committed; pass their directories explicitly.

## Inspect the included release (offline)

```powershell
python -m unittest discover -s tests -v
python -m streamlit run dashboard/app.py
```

The default website shows the preserved 2026-27 inputs and the incomplete awards
release, with historical development evidence. Select Historical demo to view the
existing versioned scenario if present, or Custom version and its local path.
Awards and Teams / players keep the same selected source version and timestamp.
A missing historical cache has an explicit error instead of invented results.

## Rebuild an awards status artifact from the shared source

Choose a new output filename; commands reject existing output artifacts.

```powershell
python -m scripts.run_awards --source data/raw/live_preseason/v1_20260922T113549Z_c79af6 --history data/processed/awards/history_v4_20261006 --output results/awards/reproduction/awards.json
```

This verifies upstream file hashes and historical evidence, keeps the source
cutoff, and produces the same status and numerical evidence as the included
release. The generation timestamp changes. It does not run an invented season.
Select that file in the website's Awards artifact field to inspect it.

## Rebuild the historical dataset from official sources

`RAW_LOG_DIRECTORY` must contain NBA player and team LeagueGameLog parquet files
for 2020-21 through 2024-25. These are the existing project's cache filenames.

```powershell
python -m scripts.collect_award_votes --output data/raw/awards/reproduction
python -m scripts.build_award_history --raw-logs RAW_LOG_DIRECTORY --votes data/raw/awards/reproduction --output data/processed/awards/reproduction
```

The collector uses 15 source URLs resolved from NBA Communications, stores
retrieval timestamps and SHA-256 hashes, and rejects partial/duplicate ballots
and inconsistent point totals. Partial failed directories must not be reused;
choose a new directory after resolving the error. PDF downloads are ignored by
Git; their manifests and extracted voting records are included. Derived history
is included so evaluation can be reproduced offline without the full raw cache.

For exact numerical baseline reproduction from the included derived history:

```powershell
python -m unittest discover -s tests -p test_awards.py -v
```

That test reruns chronological evaluation and checks equality with every saved
per-season evaluation record. Reretrieving a source can legitimately change its
hash; do not silently replace the preserved evidence.

## Generate season and awards artifacts in one run

`RUN_DIRECTORY` is an existing trained player run, and `RAW_LOG_DIRECTORY` is its
compatible validated cache. Both may be read-only. A new output directory is made
for each invocation:

```powershell
python -m scripts.run_cutoff_scenario --run-dir RUN_DIRECTORY --raw-dir RAW_LOG_DIRECTORY --output-root data/processed/player_runs/awards_checks --awards-history data/processed/awards/history_v4_20261006 --cutoff 2025-01-01 --season 2024-25 --n-sims 20 --learned-player-uncertainty
```

This runs the unchanged historical scenario model, saves standings and identical
joint player/team draws, then generates `awards.json` in that same version. Twenty
trials are an integration smoke test, not a forecast-quality run. For full
historical scenarios use the project's normal trial settings. Do not pass
2026-27 to the historical runner and claim it is an upcoming-season simulation;
it constructs its remaining schedule from historical played games.

The 2026-27 full-season simulation remains blocked by its preserved 1,200-game
schedule contract, among other player-state gaps. No award model is approved for
probabilities. See `AWARDS_MODEL_CARDS.md` for exact requirements per award.
