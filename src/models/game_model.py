"""
Train a LightGBM classifier to predict single-game win probability from
pre-game team form features.

Usage:
    python src/models/game_model.py
"""

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"

FEATURES = [
    "ROLL_WIN_PCT",
    "ROLL_POINT_DIFF",
    "IS_HOME",
    "DAYS_SINCE_LAST_GAME",
    "ROLL_EFG",
    "ROLL_TOV_PCT",
    "ROLL_ORB_PCT",
    "ROLL_FTR",
    "ROLL_EFG_OPP",
    "ROLL_TOV_PCT_OPP",
    "ROLL_ORB_PCT_OPP",
    "ROLL_FTR_OPP",
    "ROLL_EFG_DIFF",
    "SEASON_POINT_DIFF",
    "EMA_POINT_DIFF",
    "ELO_RATING",
    "OPP_ELO_RATING",
]
TARGET = "WIN"
TANKING_FLAGS_PATH = PROCESSED_DATA_DIR / "backtest_tanking_flags.parquet"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/features/build_team_features.py first."
        )
    return pd.read_parquet(path)


def add_tanking_review_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Join the separately generated tanking-review flag to feature rows."""
    if not TANKING_FLAGS_PATH.exists():
        raise FileNotFoundError(
            f"{TANKING_FLAGS_PATH} not found. Run src/backtest/flag_tanked_games.py first."
        )
    flags = pd.read_parquet(TANKING_FLAGS_PATH)
    join_columns = ["SEASON", "GAME_ID", "TEAM_ID"]
    required = set(join_columns + ["LIKELY_TANKING_REVIEW_FLAG"])
    if not required.issubset(flags.columns):
        raise ValueError("Tanking flag file does not contain the expected join columns.")
    flags = flags[join_columns + ["LIKELY_TANKING_REVIEW_FLAG"]].drop_duplicates(join_columns)
    merged = df.merge(flags, on=join_columns, how="left", validate="one_to_one")
    merged["LIKELY_TANKING_REVIEW_FLAG"] = merged["LIKELY_TANKING_REVIEW_FLAG"].fillna(False)
    return merged


def build_model() -> lgb.LGBMClassifier:
    """Return the shared deterministic model configuration."""
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def train_model(exclude_tanking_flags: bool = False, model_output: Path | None = None):
    """Fit the game model, optionally excluding review-flagged training rows.

    The held-out metric split is created before filtering. This makes the
    evaluation set representative of real games, including flagged games;
    only the fit rows are removed from the training sample.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    if exclude_tanking_flags:
        df = add_tanking_review_flag(df)
    df = df.dropna(subset=FEATURES + [TARGET])

    train_rows, test_rows = train_test_split(
        df, test_size=0.2, random_state=42
    )
    removed_count = 0
    if exclude_tanking_flags:
        removed_count = int(train_rows["LIKELY_TANKING_REVIEW_FLAG"].sum())
        train_rows = train_rows.loc[~train_rows["LIKELY_TANKING_REVIEW_FLAG"]].copy()

    X_train, y_train = train_rows[FEATURES], train_rows[TARGET]
    X_test, y_test = test_rows[FEATURES], test_rows[TARGET]

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"Training rows:   {len(train_rows):,}")
    if exclude_tanking_flags:
        print(f"Removed flagged training rows: {removed_count:,}")
    print(f"Test rows (unfiltered): {len(test_rows):,}")
    print(f"Test accuracy:   {acc:.4f}")
    print(f"Log loss:        {ll:.4f}")
    print(f"Brier score:      {brier:.4f}")
    print("\nFeature importances:")
    for feat, imp in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp}")

    model_path = model_output or MODELS_DIR / "game_model_four_factors.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-tanking-flags",
        action="store_true",
        help="Exclude flagged team-games from fit rows only; test rows remain unfiltered.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Optional path for the saved model (defaults to the primary model path).",
    )
    args = parser.parse_args()
    train_model(args.exclude_tanking_flags, args.model_output)


if __name__ == "__main__":
    main()
