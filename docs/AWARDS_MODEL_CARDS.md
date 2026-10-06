# Awards release: evidence and limitations

Release date: 2026-10-06. **No 2026-27 award probabilities or candidate rankings
are approved.** The release supplies ingestion, historical baseline evidence,
eligibility contracts, shared-draw export, accounting checks and website status.
It is not a completed award forecasting product. All-NBA is deferred.

## Shared season contract

The preserved upcoming-season source is
`data/raw/live_preseason/v1_20260922T113549Z_c79af6`. Its authoritative roster
timestamp is `2026-09-22T11:35:49.701781+00:00`, not the requested September 21.
There are 573 roster rows and 1,200 named schedule games. Its contract requires
1,230 games, and no season simulation has been produced. This release neither
guesses the 30 NBA Cup/TBD games nor changes those original artifacts.

`simulate_cutoff(..., joint_draws={})` optionally retains the existing simulation's
draws. The default return value and RNG streams are unchanged. New runs save
`joint_draws.npz` beside `report.json`, `standings.csv`, and player/team snapshots.
The NPZ has no pickle objects:

| Field | Shape / meaning |
| --- | --- |
| `schema_version`, `cutoff`, `seed` | Scalar identity |
| `team_ids`, `team_wins` | T identifiers, S by T integer outcomes |
| `player_ids`, `player_team_ids` | P identifiers, exact scenario roster membership |
| `player_pts36` | S by P persistent scoring-rate draws used in that same trial |
| `player_minutes` | P frozen per-game rotation allocations; each team sums to 240 |
| `semantics` | Explicitly not a complete simulated player season |

There are no simulated games played, starts, minute-by-minute availability,
efficiency draws or defensive impact trajectories. A scoring rate must not be
multiplied into an invented healthy 82-game award season. Team-only classifiers
do not respond to scoring residuals; exporting them does not create an effect.
The JSON awards artifact binds to source file hashes, version and cutoff. It
records status, empty candidate lists, null probabilities, exclusions, eligibility,
specific missing requirements and optional development evidence.

## Sources, population and leakage

Official NBA [2022-23](https://pr.nba.com/voting-results-2022-23-nba-regular-season-awards/),
[2023-24](https://pr.nba.com/voting-results-2023-24-nba-regular-season-awards/) and
[2024-25](https://pr.nba.com/voting-results-2024-25-nba-regular-season-awards/)
indexes identify the downloaded evidence. The collector reconciles all ballot
positions to 100, 99 and 100 voters respectively; weighted totals also reconcile.
The first two seasons use voting-total PDFs, the third uses individual ballots.
Raw PDF downloads remain local, with URLs, retrieval timestamps, hashes and
compact extracted records tracked under `data/raw/awards/v1_20261006`.

There are **163 vote-recipient records** across 15 award-seasons, joined by
checked name normalization and explicit aliases to NBA player IDs. All observed
player appearances are included in the historical population, including records
whose minutes rounded to zero. The result has **8,400 player-season-award rows**:
539, 572 and 569 players in the three target seasons, each expanded to five awards.
Unlisted players retain null vote points with `unlisted_unknown`; observed zero
and unknown are distinct. This is a broad candidate superset, **not a verified
eligible population**. Starts, rookie designations and exception decisions are
missing. No model is trained on a winners-only or vote-recipients-only population.

Player statistics and team results reuse cached NBA LeagueGameLog data. Player
points reconcile to complete team results. Trade-season player totals aggregate
across teams; `last_team_wins` is context for the last observed team, not an
all-season attribution. Original cache retrieval timestamps are unknown, recorded
as null, and must not be replaced by the time of this build. Retrospective stat
corrections cannot be ruled out. File hashes preserve the exact cache inputs.

Preseason baseline inputs use only prior seasons with a source date strictly
before October 1 of the target season. The population is prior-season participants,
retaining players who later leave the NBA; target participation never selects the
forecast field. New entrants are excluded and counted in the evaluation's coverage
audit. Final voting and winner labels are separate, conservatively marked
available on the complete index publication date, and used only after ranking.

## Eligibility contract

The [current NBA-NBPA CBA](https://www.nbpa.com/cba/) took effect in July 2023.
Article XXIX section 6 governs MVP, DPOY and MIP: 65 qualifying games, normally at
least 20 minutes each, with two games of at least 15 minutes allowed. The 62-game
season-ending-injury route additionally requires 85% participation before injury
and the specified medical determination. Successful eligibility grievances or
extraordinary-circumstances challenges are separate routes. The implementation
requires an externally verified approval; it never infers one from an injury.
Missing decisions remain unknown. Earlier season rules are not backfilled.

Sixth Man requires strictly more bench appearances than starts, per the
[official 2025-26 release](https://pr.nba.com/keldon-johnson-2025-26-kia-nba-sixth-man-of-the-year/).
ROY and Sixth Man are not covered by this CBA's 65-game rule. A verified rookie
designation is required; lack of logs is not proof of rookie status. Season-specific
2026-27 rookie and Sixth Man instructions/designations are still a release gate.
Rule verification timestamps and document references are in the award artifact.
Rounded LeagueGameLog minutes cannot resolve the precise 20-minute boundary.

## MVP card

Status: incomplete. Transparent preseason baselines rank the prior year's points
per appearance or total points, without claiming these are adequate award scores.
Winner ranks in 2022-23, 2023-24 and 2024-25 are **1, 25, 4** for PPG and
**3, 17, 2** for total points. Both place two of three winners in the top five.
Candidate counts are 605, 539 and 572. The future award model needs joint production,
efficiency, usage/role, availability and team outcomes plus an eligible full field.

## Most Improved Player card

Status: incomplete. The growth baseline carries forward the previous year's PPG
change; the comparator is previous-year PPG. Both use the same two-year-history
field (450, 433, 450 players), excluding 155, 106 and 122 prior players without
the second history year. Growth winner ranks are **144, 83, 78**; PPG ranks are
**90, 50, 271**. Neither baseline puts any winner in the top five. These results
do not support publishing a 2026-27 MIP ranking. A role-change/development model
and explicit limited-history treatment are required.

## Defensive Player of the Year card

Status: incomplete; no ranking evaluated or issued. Steals and blocks are available
in the logs but do not identify total defensive impact. Possession/lineup context,
opponent quality, team defense and stable defensive measurements must be assessed
and projected before a model is fit. Availability/eligibility is also missing.

## Rookie of the Year card

Status: incomplete; no ranking evaluated or issued. Verified rookie populations
and historical preseason rookie projections are absent. Returning-player averages
would systematically omit the target field; league-average priors are not an
evaluated rookie forecasting model. College/international/other evidence and
explicit uncertainty for limited-history players require separate validation.

## Sixth Man of the Year card

Status: incomplete; no ranking evaluated or issued. The player logs lack starts,
and the season simulator has no starts/bench-role draws. Projected average minutes
cannot establish the strict bench-majority rule. A dated role model with observed
historical starts and season-specific eligibility checks is required.

## Evaluation and promotion

All three target seasons have now been examined and are **development evidence**.
These are chronological, fixed preseason statistical baselines, not an evaluation
of the shared simulation-to-voting pipeline. No end-of-season model is represented
as preseason evidence. Three independent outcomes per award and unresolved fields
are insufficient for fitting and selecting a credible learned choice model.

No probabilities were issued, so winner log loss, Brier score and calibration are
null, not zero or passed. Rank ties use average rank; an absent winner is a miss.
Full per-season ranks, reciprocal ranks, top-five indicators and coverage are in
`data/processed/awards/history_v4_20261006/evaluation.json` and shown on the website.

Before approving probabilities: expand the sourced historical panel; reconstruct
dated full candidate populations and preseason projections; fit transparent and
learned alternatives using only earlier seasons; tune on development seasons;
evaluate untouched later seasons through the entire simulation pipeline. Report
winner ranking, log loss, multiclass Brier and calibration by season with
season-level uncertainty. Calibrate conditional voting preferences separately
from season uncertainty; never label a hand score or arbitrary softmax calibrated.
Only then integrate conditional choice probabilities over the identical season
draw IDs. Each trial and aggregated full field must sum to one. The tested
`leading_probabilities` helper retains omitted mass as Other. It is not exposed
as a forecast in this release because no choice model is approved.

## Verification and remaining work

All 47 tests in the full `tests/` suite passed. The focused suite covers vote parsing, null semantics, qualification boundaries,
exceptions, chronological leakage, coverage, full-field mass, Other, all simulator
export paths, source tampering, version/cutoff mismatch and dashboard missing states.
A real 20-trial historical integration run passed checks for 30 teams and 478
players; it is only an engineering smoke test. Existing historical results and
the original preseason input run were not overwritten.

Desktop and 390-by-844 mobile layouts were inspected in the browser. Saved
screenshots and the integration audit are beside the release awards artifact.
The website is local, not deployed. Remaining blockers: the upstream complete
season scenario; complete player-season/role/availability draws; verified eligible
fields; adequate award-specific predictors; and untouched full-pipeline evaluation.
