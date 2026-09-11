# 2024-25 fixed-cutoff simulation results

## Scenario

This is a retrospective scenario from the 2025-01-01 cutoff. The simulator uses frozen past-only team and player profiles, 2,000 seeded trials, and the final historical remaining schedule as an input. That schedule was **not** treated as information known on 2025-01-01. Roster membership is last-observed and no future transactions or injuries are assumed known.

Model: `hybrid_development_ml` with learned persistent player scoring-residual uncertainty. Source artifact: `v1_20260910T215749Z_35f88a/scenario_20260910T220652Z` (local, ignored from Git).

## Headline results

- 746 remaining games; 2,000 trials; seed 42.
- Final-standings MAE: 4.851 wins.
- Empirical coverage of nominal 90% intervals: 86.7% (26 of 30 teams).
- This is one retrospective cutoff, not evidence of a deployable live or preseason forecast.

## Example projected standings

| Team | Mean wins | 90% interval | Actual wins |
|---|---:|---:|---:|
| Oklahoma City | 64.6 | 58–71 | 68 |
| Cleveland | 64.3 | 57–71 | 64 |
| Boston | 56.8 | 49–64 | 61 |
| New York | 55.3 | 48–62 | 51 |
| Memphis | 52.5 | 45–60 | 48 |
| Denver | 50.0 | 42–57 | 50 |
| Milwaukee | 48.0 | 40–56 | 48 |
| Houston | 48.0 | 41–55 | 52 |

## Interpretation

The uncertainty component produces wider, more realistic season ranges than outcome-only simulation. It does not solve standings bias, establish causal player value, or provide verified current-roster/injury knowledge. Use this as a reproducible portfolio demonstration of the project’s model, uncertainty treatment, and limitations.
