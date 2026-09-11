# Portfolio finalization prompts

Run these in order after the completed fixed-cutoff scenario. Each task must preserve versioned artifacts, avoid overwriting models, and end with focused tests, documentation, a scoped commit, and a push.

## Prompt A — Scenario-results dashboard

Build a small Streamlit dashboard around `reports/FIXED_CUTOFF_2025_RESULTS.md` and its source artifact. Require an explicit artifact path; never train, mutate, or overwrite an artifact during use. Show the cutoff date, seed, trials, schedule disclaimer, standings table, p05/p50/p95, actual wins, MAE, and interval coverage. Label every result “historical fixed-cutoff retrospective scenario.” Add graceful errors for missing artifacts and a smoke test for schema and league-win conservation.

## Prompt B — Current-season data readiness audit

Inventory the requirements for a current-season forecast: published schedule snapshot, official transaction events, roster status, injuries/availability, and retrieval timestamps. Record effective date separately from known-at date. Prefer NBA/team primary sources; do not infer historical knowledge from a page retrieved today. Produce a source matrix and a fail-closed ingestion plan. If historical known-at data cannot be established, retain live-only support and mark historical backtesting blocked.

## Prompt C — Standings-bias diagnosis

Diagnose the fixed-cutoff model’s standings compression without changing the selected model. Compare errors by actual-win decile, horizon, conference, and top/bottom-five teams. Separate probability calibration from schedule, roster, and simulation effects. Use grouped season/cutoff comparisons and report uncertainty. Propose changes only as new experiments; do not tune against the 2024-25 scenario.

## Prompt D — Portfolio release checklist

Write a concise README walkthrough: setup, one reproducible scenario command, artifact locations, dashboard command, and limitations. Include a portfolio-safe project description distinguishing independently trained ML, experimental components, and blocked data. Confirm raw data, generated artifacts, and user-specific paths remain outside Git. Run the dashboard smoke test and source compilation before release.
