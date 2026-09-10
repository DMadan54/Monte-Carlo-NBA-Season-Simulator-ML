# Implementation prompts

Execute in order. Read AGENTS.md and PLAYER_MODEL_RESEARCH.md. Preserve unrelated changes and existing artifacts. Record commands, results, limitations, and status in EXECUTION_LOG.md. ML trained in this repository is central; transparent baselines establish whether it adds value. Data gates block dependent estimation, not independent code/tests. Never claim synthetic tests establish real predictive quality.

## Prompt 1: Baseline and leakage audit

Inspect data, features, training, simulation, and existing tests. Verify authoritative season labels, Four Factors denominators, chronological game grouping, feature schemas, and forecast horizon. Add regression tests for verified defects and repair them without overwriting historical artifacts. Include future-mutation tests and reproduce a chronological baseline in a unique directory. Distinguish rolling replay from fixed-cutoff forecasts. Do not add player features yet.

## Prompt 2: Player data contract

Inventory cached player logs first. Implement validated loading with stable IDs, dates, season labels, minutes, counts, and source/cutoff metadata. For missing necessary data, use bounded cached retrieval after verifying a sample. Check duplicates, nonnegative counts, game/team linkage, overtime-aware minute totals, and trades. Define fail-fast contracts for dated rosters, birth dates, injuries, and coaches; do not infer advance knowledge from later participation. Record coverage and missing data.

## Prompt 3: Player profiles and learned minutes

Build strictly past-only exposure-weighted decayed player profiles with shrinkage, plus simple and ML minutes forecasts trained chronologically. Separate activity from conditional minutes where data support it. Support explicit roster scenarios, enforce 240-minute regulation rotations, redistribute displaced minutes, and label last-observed historical rosters as estimates. Test unseen players, zero exposure, trades, cutoff invariance, and minute conservation. Evaluate on real future games where possible.

## Prompt 4: Our player-aware game model

Train team-only, roster-only, and hybrid ML models on matching chronological folds, with both opponents represented and one probability per game. Use past-only or chronological out-of-fold upstream forecasts. Box-score profiles are not RAPM; fit independent RAPM only with validated possession data. Save versioned models, schemas, seeds, cutoffs, and hashes. Report log loss/Brier/calibration and coverage/error slices. Preserve the existing default model and document our original modelling contributions.

## Prompt 5: Development and destination fit

Compare simple player forecasts against learned future-performance forecasts. Add age-based trajectories only with verified birth dates; distinguish recent-history prediction from long-term aging. Test a small regularized set of transferable roster-profile interactions against the additive baseline, including held-out seasons, roster changes, and unseen combinations. Avoid pair-ID memorization and causal claims. Use explicit scenarios when transaction history is absent, and record blocked components honestly.

## Prompt 6: Coaching

Validate dated coaching assignments. Fit continuity and a strongly regularized coach term conditional on roster/team/opponent strength, comparing with the identical no-coach model on chronological data and coaching changes. Check identifiability and pool unseen coaches to the prior. If assignments cannot be sourced, implement and test the data contract and record that estimation remains blocked. Do not invent coaching ratings.

## Prompt 7: Explicit-cutoff simulation and review

Integrate the available experimental player model into a new explicit-cutoff simulator without replacing saved legacy models. Accept dated rosters, a remaining schedule, current wins, fixed seed, and simulation count. Hold player/season uncertainty persistent within each trial and separate game noise. Forbid future actual stats or trades. Test winner/win conservation, reproducibility, expected wins versus Monte Carlo means, and cutoff invariance. Run real-data evaluation for supported modes, distinguishing scenario forecasts and rolling retrospective replay from preseason validation. Write a model card and update execution status with what helped, failed, or remains data-blocked.
