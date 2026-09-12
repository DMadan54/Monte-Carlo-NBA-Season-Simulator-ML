# Fixed-cutoff probability-calibration experiment

## Protocol

Used the saved player-aware fixed-cutoff game probabilities. Platt and isotonic calibrators were fit on 2018-19 through 2022-23 cutoffs, selected once on 2023-24 validation log loss, then evaluated once on 2024-25.

## Result

| Method | Validation log loss | Test log loss | Test Brier | Test ECE |
|---|---:|---:|---:|---:|
| Raw | 1.478 | 1.527 | 0.359 | 0.306 |
| Selected isotonic | 0.691 | 0.687 | 0.247 | 0.012 |

## Decision

Isotonic calibration clearly improves the saved fixed-cutoff probability output. However, the raw-versus-calibrated gap is unusually large relative to rolling next-game results. Before integration into the simulator, audit schedule orientation, game-target alignment, and cutoff feature construction. Do not claim that calibration alone fixes standings compression until calibrated probabilities are re-simulated and compared by cutoff.
