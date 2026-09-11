# Next-phase implementation prompts

Execute these prompts in order. Read `AGENTS.md`, `docs/MODEL_CARD.md`, `docs/PLAYER_MODEL_RESEARCH.md`, and `docs/EXECUTION_LOG.md` before beginning. This is an independently trained NBA forecasting portfolio project, so every experiment must be reproducible, leak-free, and allowed to fail.

## Common rules

- Every prediction date is an information cutoff. Features, transforms, roster membership, transaction records, schedules, and fitted models must come from strictly before that cutoff. Use `shift(1)` or an equivalent date-safe operation for derived game features.
- Do not overwrite legacy models, raw caches, or previous experiment directories. Save each run under a unique versioned directory with command, seed, feature schema, source hashes, cutoffs, and metrics.
- Use chronological train/validation/test splits. Select once on validation, freeze configuration, and evaluate later data. Do not tune against a final test season or a single cutoff.
- Keep matched team-only baselines. Complexity earns promotion only by meeting predeclared evaluation gates.
- Update the model card and execution log with commands, results, limitations, and status: promoted, experimental, rejected, or blocked.
- Before any commit, inspect `git status --short` and stage only files changed for the prompt. Do not sweep unrelated files into the commit.

## Prompt 1 — Multi-cutoff calibration backtest

Build a chronological fixed-cutoff backtest for the team-only baseline and player-aware hybrid, both with and without learned player-residual uncertainty. The purpose is to determine whether the January 2025 coverage result generalizes rather than treating one cutoff as proof.

Use predeclared cutoffs of December 1, January 1, February 1, and March 1 for every completed supported season from 2018-19 through 2024-25. At each cutoff, create team and player snapshots only from prior records. Use fixed seeds and documented trial counts. The final historical remaining schedule may be used only as a clearly labelled retrospective scenario input; never present it as the schedule known at the cutoff. Future appearances, transactions, results, or residual labels must not affect an earlier snapshot or forecast.

Save, for every model and cutoff, projected team wins, p05/p50/p95, simulation standard deviation, actual final wins, feature/model versions, and all source dates. Aggregate standings MAE and RMSE, bias by final-win decile, rank correlation, nominal 50/80/90% interval coverage, average interval width, Brier score, log loss, and calibration bins. Report results overall, by horizon, and for top-five/bottom-five teams. Use paired comparisons grouped by season/cutoff, not independent team rows.

Add mutation tests proving data after a cutoff cannot change its snapshot or output. Test reproducibility, league-win conservation, expected wins versus Monte Carlo means, and that learned uncertainty is trained only on earlier out-of-fold residuals. Predefine the promotion gate: learned uncertainty should bring average coverage closer to nominal without a material average MAE regression. Define “material” before examining the aggregate result.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Add multi-cutoff calibration backtest"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 2 — Dated roster, transaction, and availability contract

Design a source-backed contract for roster membership, transactions, and player availability. This is the prerequisite for reliable fixed-cutoff rosters, trade scenarios, and player contribution estimates. Inventory candidate sources first: effective dates, known-at dates, coverage, IDs, retrieval times, licensing constraints, and whether the data can support historical backtests and live forecasts. Prefer primary sources. Do not substitute first appearance for transaction date unless it is labelled an estimate.

Create validated, versioned ingestion tables with player/team IDs, event type, effective date, known-at date, source identifier, retrieval timestamp, and confidence/status. Define semantics for trades, waivers, signings, inactive periods, injuries, returns, and unknowns. Keep known-at distinct from effective-on. Fail closed when an event would make a pre-cutoff roster unknowable.

Implement roster reconstruction at a cutoff with an auditable trail explaining every player’s inclusion or exclusion. Test trades before a cutoff, announcements after a cutoff, players not yet appearing for a destination, missing/duplicate events, and inactive-return sequences. Run only a small diagnostic scenario set to measure the effect of corrected roster membership; do not tune the game model in this phase.

Document the remaining gap separately for historical evaluation and live use. If no reliable dated source passes the contract, retain the ingestion validation and tests, mark roster reconstruction blocked, and keep last-observed rosters experimental.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Add dated roster data contract"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 3 — Preseason player state-transition model

Turn the development experiment into an independently evaluated preseason player state-transition model. Forecast next-season participation, total minutes, and rate outcomes such as points, rebounds, and assists per 36. Do not integrate it into the simulator until it earns that step.

Create one row per player-season using only prior-season information and date-safe static attributes such as verified birth date. Keep targets separate: participation, total minutes, and rate statistics conditional on a documented future-minute threshold. Preserve exits as zero-minute observations in participation/minutes evaluation; do not allow survivor-only rate metrics to become full-population claims.

Compare last-season-rate, age-bucket, and shrinkage baselines with regularized linear and gradient-boosting approaches. Use chronological season folds, select a frozen configuration on validation seasons, and reserve later seasons for final evaluation. Report minutes MAE, participation calibration, minute-weighted rate MAE/RMSE, and errors by age/experience/role/team-change slice. Add strict static-age and future-mutation tests.

Save a versioned model artifact containing target definitions, inputs, cutoff, seed, predictions, and uncertainty. Only pass the component into a later preseason simulator phase if it improves predeclared baselines on frozen evaluation. Otherwise document it as research evidence.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Evaluate preseason player state transitions"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 4 — Interpretable roster compatibility model

Build a compact, interpretable compatibility layer that estimates model fit between a player and destination roster. Do not claim it estimates a causal trade effect. Use transferable aggregate components rather than player-pair identifiers: creation, shooting/spacing proxies, rebounding, turnover burden, role/size balance, age/experience mix, and defensively relevant proxies available in past-only data.

Define compatibility as interactions between an incoming player profile and the destination roster excluding that player. Prevent future performance, player/team/season IDs, and pair-specific memorization from entering the learner. Use the additive roster model as baseline. Predeclare a small regularized interaction set and use chronological out-of-fold player forecasts upstream.

Evaluate first on validation and then frozen later-season tests. Report game log loss, Brier score, calibration, fixed-cutoff standings MAE, and player-minute errors. Include ablations by compatibility family. Slice by team changes and new combinations only with reported sample sizes. Reject the component if it does not consistently beat the additive player model on validation.

Provide a scenario interface using the phrase “estimated model contribution under stated roster assumptions,” with uncertainty and limits. Never call it a causal trade estimate without a separate causal design and data.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Evaluate roster compatibility features"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 5 — Coach contribution data gate and model

First obtain and validate dated head-coach assignments. Each row needs team/coach IDs, effective start/end dates, known-at or defensible availability metadata, and a source reference. Do not assign a season-level coach label to every game if an in-season replacement may have occurred. If the source cannot meet this standard, implement/retain the contract and mark real coach estimation blocked.

If the gate passes, fit coach contribution only as a strongly regularized residual after roster strength, team history, opponents, rest, home context, and season effects. Include continuity/tenure features and partial pooling so limited-sample coaches shrink heavily toward zero. Raw team winning percentage is not coach quality.

Compare chronological no-coach and coach models on matching folds, coaching changes, and later seasons. Report log loss, Brier, calibration, standings MAE, residual distributions, uncertainty, and shrinkage. Test future-assignment mutation, known-at handling, unknown/unseen coaches, and midseason transitions. Promote only for a stable held-out gain beyond roster strength.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Add coach contribution evaluation"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 6 — Preseason simulator integration and final backtest

Integrate only components that passed their own gates: player state transitions, dated roster reconstruction, compatibility features, coach effects, and calibrated residual uncertainty. Create a new versioned preseason simulator rather than changing the legacy rolling replay. Inputs must be a date-safe preseason roster snapshot, published schedule snapshot when available, selected model bundle, seed, and trial count.

For each trial, sample persistent player state and availability once for the season, construct feasible rotations, and sample game outcomes from the frozen game model. Keep player-state uncertainty separate from game-outcome noise. Enforce 240 regulation minutes, league-win conservation, fixed seeds, and output standings, seeds, playoff qualification probabilities, and team/player contribution summaries. Do not add playoff brackets until regular-season validation passes.

Backtest chronologically across seasons with a genuine preseason schedule/roster snapshot. If inputs are reconstructed, label the result a historical scenario. Compare against team-only preseason and prior-season-win baselines; add external baselines only when they are legal and date-safe. Report standings MAE, rank correlation, playoff Brier score, interval coverage, strong/weak-team bias, and calibration by season.

Make a promotion decision in the model card. The project’s leading simulator requires frozen configuration, repeatable scripts, versioned artifacts, passing leakage tests, and a documented multi-season advantage—or an honest record that it has not surpassed the baseline.

Update the model card and execution log. Run focused tests, stage only this prompt’s files, commit with `git commit -m "Integrate validated preseason simulator"`, run `git push origin main`, then verify `git status --short` is clean.

## Prompt 7 — Portfolio dashboard and reproducibility release

Build the Streamlit dashboard only after the selected simulator and its limitations are frozen. Load a versioned artifact by explicit path and display artifact version, training cutoff, data coverage, simulation seed, trial count, and whether results are preseason, fixed-cutoff scenario, or rolling replay.

Provide standings distributions, team p05/p50/p95, playoff odds, player minutes/contributions, trade scenarios, and calibration/backtest evidence. Trade pages require user-supplied assumptions and must show uncertainty and non-causal status. Do not display unavailable coach, injury, RAPM, or chemistry measures as if they were real estimates.

Add a reproducible forecast command and a smoke test that loads the selected artifact, runs a small seeded simulation, and verifies output schema and win conservation. Include a model-card view linking sources, protocol, failure modes, and a portfolio-safe description of the independently developed ML work. Test missing artifacts, invalid IDs, empty data, and unsupported dates with clear errors.

Update the README with local setup and demo commands. Keep raw data, large artifacts, secrets, and user-specific paths out of Git. Confirm the app never mutates an artifact during normal use.

Update the model card and execution log. Run focused tests and dashboard smoke tests, stage only this prompt’s files, commit with `git commit -m "Add reproducible forecasting dashboard"`, run `git push origin main`, then verify `git status --short` is clean.
