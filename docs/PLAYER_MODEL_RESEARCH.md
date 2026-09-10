# Research informing our player-aware models

Researched 2026-09-10. This note supports implementation; it is not a benchmark run in this repository.

## Decision

Develop our own modular ML pipeline: past-only player profiles -> learned minutes/performance forecasts -> projected rotations -> learned game probabilities -> Monte Carlo standings. Retain team-only and simple player baselines. Research supports this architecture, but does not identify one universally best algorithm for our data and horizon. Fit development, compatibility, and coaching as separate experiments.

Use standard libraries to train our own estimators and document our feature design, targets, leakage controls, ablations, and errors. Do not merely import proprietary player ratings and present the result as our own player model. Exact RAPM requires possession data; with box scores, train profile-based models and label their limitations.

## Evidence and limitations

### Regularized player impact

Joseph Sill's 2010 paper introduces ridge regularization for adjusted plus-minus and evaluates predictions on held-out games. It supports regularized impact estimates and future testing instead of raw plus-minus rankings; it does not establish a fix for this project's standings bias. [Original paper, hosted copy](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf).

EPM's creator describes projected skills feeding a statistical plus-minus prior for RAPM, then predicted-minute aggregation, game predictions, and Monte Carlo simulations. Its published game comparison uses random 10-fold cross-validation, not our required chronological preseason test. Historical daily values also rely on upstream fitted models, whose cutoffs require auditing. Adopt the architecture as inspiration, not its published error as an expected result here. [EPM methodology](https://dunksandthrees.com/about/epm).

### Player development

DARKO describes stat-specific exponential decay, Bayesian updating/modified Kalman filtering, aging curves, and boosted combinations. It adjusts to team changes and increases uncertainty after a move; its documentation also identifies playing time as difficult to predict. Our starting point is exposure-weighted decayed rates, shrinkage, and learned next-game/next-period forecasts; compare development against a no-change baseline. This is an independent model, not an exact DARKO implementation. [Creator methodology](https://www.darko.app/about).

### Destination fit

Petridis and Pelechrinis (2026), L-RAPM, shrink lineup estimates toward player-rating sums and control opponents. They use 2022-23 player priors and expanding weekly evaluation in 2023-24. Unseen lineups fall back to player-based priors: that is not learned pair-specific chemistry for combinations never observed. The preprint compares with raw lineup ratings, not a comprehensive set of modern forecasters. [Full paper](https://arxiv.org/html/2601.15000v1).

Martonosi, Gonzalez, and Oshiro use player order statistics to classify previously unobserved lineups. Their conservative classifier targets precision in identifying positive-plus-minus lineups, not continuous team strength or season wins. This supports testing transferable skill summaries instead of only memorizing lineup IDs. [Authors' abstract](https://arxiv.org/abs/2303.04963).

A PLOS ONE study of ESPN RPM finds teammate complementarity remaining after adjustment. An impact rating therefore should not automatically be treated as fully portable across rosters. The paper does not supply a universal trade-effect formula. [Research article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237920).

### Coaches

Cannon, Fisher, Fellingham, and Page (2026) describe roster-adjusted coach win-probability curves using monotone Bayesian additive regression trees. Only the author/publisher-supplied abstract indexed by RePEc was accessible; full methods/results were not verified. Treat this as a research lead, not a validated effect size to copy. [Paper DOI](https://doi.org/10.1515/jqas-2025-0025), [accessible abstract](https://ideas.repec.org/a/bpj/jqsprt/v22y2026i2p155-168n1003.html).

Our proposed simpler experiment uses dated assignments and a strongly regularized coach term conditional on team/player/opponent strength. This is our design choice, not that paper's exact model. Keep it only if future-game predictions improve; coaching changes are not randomized.

### Data

pbpstats documents possession parsing and on-floor lineups. It is a candidate tool, not proof of historical endpoint availability or correct reconstruction. Validate scores, minutes, and lineups before fitting RAPM. [Official documentation](https://pbpstats.readthedocs.io/en/latest/).

## Evaluation commitments

- Freeze a baseline and repair verified chronology/feature defects before measuring new features.
- Past participation is evidence only after that game, not an advance active-roster announcement. Last-observed rosters are estimates; expose that limitation.
- Separate rate/skill development from increased minutes. Use actual birth dates for age effects and explicit priors for unseen players.
- Aggregate generic player profiles with minutes/240; additive per-100-possession impact coefficients use minutes/48. State units; do not add arbitrary scores to Elo.
- Compare team-only, player-only, and hybrid models on matching chronological folds. Upstream learned forecasts must be out-of-fold for downstream training.
- Add development, fit, and coaching separately. Test roster changes and unseen combinations. Treat historical trades as observational, not randomized interventions.
- Separate rolling next-game replay from preseason or fixed-cutoff forecasts. Unknown future trades are scenarios, not available historical knowledge.
- Report log loss, Brier, calibration, coverage, standings MAE/bias/interval coverage, and uncertainty. Do not tune until one famous team's wins look right.
