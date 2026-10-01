# 2026-27 Preseason Scenario - not a mid-season forecast

## Status: BLOCKED - no standings projection has been generated

The current-data ingestion contract completed successfully on 2026-09-22 UTC,
but the requested full-season simulation cannot be run without inventing part
of the regular-season schedule. The official NBA schedule PDF dated 2026-08-13
contains 1,200 named games (80 appearances for each of the 30 teams), not the
1,230 games required for an 82-game season. The remaining 30 games are
unassigned NBA Cup/TBD slots. They are not simulated, imputed, or treated as
neutral games.

## Ingested source contract

- Versioned input run: `data/raw/live_preseason/v1_20260922T113549Z_c79af6`
- Final 2025-26 team logs: 2,460 rows / 1,230 games; 2025-10-21 through 2026-04-12.
- Final 2025-26 player logs: 26,651 rows / 1,230 games; 2025-10-21 through 2026-04-12.
- Roster snapshot: 573 player rows across all 30 teams, retrieved 2026-09-22 UTC
  from the NBA `CommonTeamRoster` endpoint for season 2026-27.
- Published schedule: 1,200 named games, 2026-10-20 through 2027-04-11, parsed
  from the NBA's schedule-by-date PDF. It validates to exactly 80 named games
  per team.

The user-requested reference date was 2026-09-21. The roster endpoint is a
retrieval-time snapshot and has no historical-as-of guarantee; the preserved
retrieval timestamp (2026-09-22 UTC) is the authoritative timestamp for this
artifact. It must not be relabelled as a verified September 21 snapshot.

## Model and scenario contract, pending schedule completion

- Probability model: raw, uncalibrated hybrid probabilities. Calibration is
  deliberately not applied. The stated matched 2024-25 justification is
  2.940 top-5 MAE and 4.343 bottom-5 MAE for raw versus 8.640 and 11.055 for
  isotonic.
- Team state: final 2025-26 rolling form, Four Factors, and Elo, reconstructed
  only from the completed 2025-26 logs.
- Roster strength: returning players use final-2025-26 minutes share. Players
  new to their current team, including free agents, trades, and rookies, will
  receive a league-average bench-minutes weight rather than an assumed starter
  role. New players without 2025-26 NBA logs require league-average rate priors.
- Availability: no injuries, restrictions, or absences are assumed; no injury
  data was supplied.
- Trials: neither the 1,000-trial smoke test nor the 5,000-trial run was
  started, because both would produce incomplete 80-game standings.

## Required next input

Obtain an authoritative 1,230-game 2026-27 schedule with the 30 currently
unassigned NBA Cup/TBD games resolved. Then re-run the versioned ingestion and
the 1,000-then-5,000 raw-hybrid simulation. Do not append guessed fixtures to
the preserved input run.
