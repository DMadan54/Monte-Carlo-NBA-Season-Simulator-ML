# MVP inputs and historical evidence

The MVP input module adds four groups for eventual award-model training: production
(points, rebounds and assists per appearance), efficiency (true shooting and its
difference from league true shooting), availability/role (games, minutes and games
share), and team success (team win percentage). These are candidate predictors,
not hand-selected model weights. They describe statistical associations, not what
causes voters to vote for a player.

`src/models/mvp_inputs.py` computes the inputs from sufficient statistics and
exposes `preseason_inputs` to carry forward the previous season without looking at
target-season players or performance. Missing shooting efficiency remains missing;
there is no implicit zero. The full observed player population is preserved.
Team context uses a player's final team; stint-weighted traded-player context
remains a limitation. Games share is a descriptive availability proxy, not an
eligibility decision. This version's complete seasons all have 82 team games.

## Evidence version mvp_inputs_v1_20261009

The saved artifact contains 2,285 player-seasons from 2021–22 through 2024–25.
Official MVP voting covers 2022–23, 2023–24 and 2024–25. It reuses the existing
hash-verified `history_v4_20261006` dataset and preserves NBA source URLs, retrieval
timestamps and cache hashes. Original game-log retrieval timestamps are unknown;
the output does not invent them. The partial 2020–21 history is explicitly excluded.
No original artifact is overwritten.

| Winner | Season | PPG | RPG | APG | TS% | Games | Team wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joel Embiid | 2022–23 | 33.08 | 10.15 | 4.15 | 65.47 | 66 | 54 |
| Nikola Jokić | 2023–24 | 26.39 | 12.35 | 8.96 | 65.01 | 79 | 57 |
| Shai Gilgeous-Alexander | 2024–25 | 32.68 | 4.99 | 6.39 | 63.68 | 76 | 68 |

These different profiles motivate multiple production dimensions. They do not
establish that a particular metric determines the award. The website also exposes
within-season Spearman correlations between each feature and recorded vote points.
Those correlations cover only listed recipients, are subject to selection bias,
and must not be read as correlations over the entire player population. Unlisted
players have no imputed vote total.

## Chronological preseason checks

Each 2022–23 through 2024–25 check uses only the previous season's metrics, with
an October 1 cutoff and a strict source-date check. Target outcomes are used only
for evaluation. The field retains players who subsequently leave and excludes
incoming players; coverage exclusions are reported. There are 30 checks: ten
single-feature baselines across three seasons. No weights are fitted.

Prior-year PPG ranks the eventual winners 1st, 25th and 4th; prior-year rebounds
rank them 6th, 3rd and 81st, while assists rank them 68th, 4th and 23rd. This is
evidence against relying on any one production statistic across these examples.
Efficiency-only rankings include low-volume players, another reason to evaluate
efficiency jointly with production and role. Every examined season remains
development evidence. No model probabilities, log loss, Brier score or calibration
claim is produced by these rankings.

## Reproduce

From the repository root, with the project's Python environment:

```powershell
python -m scripts.build_mvp_inputs --history data/processed/awards/history_v4_20261006 --output data/processed/awards/mvp_inputs_local
python -m unittest discover -s tests -p test_mvp_inputs.py
python -m unittest discover -s tests -p test_awards_dashboard.py
```

The output directory must be new. Feature tables/checks are deterministic for the
same source files; generation timestamps vary. `dashboard/app.py` pins the saved
evidence version explicitly and checks the input-table fingerprint. The historical
research panel is labelled separately from the selected season forecast.

## Remaining model work

Expand historical voting and full populations beyond three outcomes, then train
regularized ranking/choice models with chronological evaluation and feature-group
ablations. Verify eligibility and historical preseason populations. Extend the
shared joint simulation beyond scoring rates to rebounds, assists, shooting
volume/efficiency, availability and complete minutes. Evaluate the full preseason
pipeline separately from a model trained on actual season statistics. Only then
consider publishing calibrated MVP probabilities. The current release adds inputs
and evidence; the 2026–27 forecast remains incomplete.
