# Monte Carlo NBA Season ML Simulator

Simulates the rest of an NBA season thousands of times using a machine-learned
game-outcome model, producing predicted final standings, playoff seeding
probabilities, and bracket odds — alongside a separate ML model that predicts
the MVP/ DPOY/ awards race from player-season stats.

## What this does

- **Game outcome model**: predicts single-game win probability from
  pre-game team form (rolling win %, point differential, rest days,
  home/away)
- **Season simulator**: runs the remaining schedule thousands of times
  (Monte Carlo) using the game model as the engine, aggregating into
  standings and playoff probabilities
- **Awards predictor**: separate model trained on historical MVP/DPOY
  voting share, predicting the current season's award race from
  player stats
- **Backtesting**: both models are validated against real past seasons
  they were not trained on

## Tech stack

- Python, pandas, `nba_api` for data
- LightGBM for the predictive models
- Streamlit for the interactive dashboard

## Project structure

```
data/
├── raw/              # cached raw pulls from nba_api (not committed — see .gitignore)
└── processed/         # engineered feature tables

src/
├── ingest/            # pulling and caching data
├── features/           # feature engineering (rolling stats, etc.)
├── models/             # game model, season simulator, awards model
└── backtest/            # validating against past seasons

notebooks/              # exploration
app/                     # Streamlit dashboard
```

## Getting started

```bash
pip install -r requirements.txt

# Pull team game logs for a range of seasons (caches locally as Parquet)
python src/ingest/pull_team_game_logs.py --start_season 2018 --end_season 2024

# Build the rolling-feature table used to train the game model
python src/features/build_team_features.py
```

## Roadmap

- [X] Data ingestion (team game logs)
- [X] Feature engineering (rolling team form)
- [X] Train game-outcome model (LightGBM) — `src/models/game_model.py`
- [X] Monte Carlo season simulator scaffolding — `src/models/season_sim.py`
  (needs a real remaining-schedule pull wired in — see TODO in file)
- [ ] Pull historical awards voting data
- [ ] Train awards-race model
- [ ] Backtest both models against real past seasons
- [ ] Streamlit dashboard
- [ ] Deploy live demo

## Results

_(to be added once the models are trained and backtested)_
