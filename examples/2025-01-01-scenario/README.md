# Bundled historical scenario

This 2025-01-01 fixed-cutoff retrospective scenario is copied from the local versioned run `data/processed/player_runs/v1_20260910T215749Z_35f88a/scenario_20260910T220652Z/`. The source run used 2,000 trials and the final played remaining schedule as a scenario input. The schedule was not known in full at the cutoff; rosters were estimated from prior appearances. The data is for a portfolio dashboard demonstration, not a live or preseason forecast.

`scripts.verify_portfolio_evidence` checks that `report.json` and `standings.csv` match the saved source byte-for-byte when the local run is available. The dashboard validates 30 distinct teams, interval order, league-win conservation, reported MAE, and coverage before displaying them.
