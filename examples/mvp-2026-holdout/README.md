# Bundled MVP awards evaluation

This is a read-only extract of `data/processed/mvp_runs/v4_2026_holdout_20261012T105053Z_78977174/`. `scripts.build_awards_example` generated it from the versioned run. The manifest records source hashes and the official voting source; `scripts.verify_portfolio_evidence` compares every bundled row with the local run when it is available.

The 2025–26 rows are retrospective January 1, February 1, and March 1 forecasts of the eventual MVP. Each date uses only games before that date and a model trained on seasons ending no later than 2025. The eventual winner column is an evaluation label, never a predictor. The three dates are **one independent award outcome**. The learned model missed the winner at all three dates while the simple baseline picked him. This is an experimental model comparison, not a live or preseason forecast.
