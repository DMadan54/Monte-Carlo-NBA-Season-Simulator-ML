"""
Train a LightGBM classifier to predict single-game win probability from
pre-game team form features.

Usage:
    python src/models/game_model.py
"""

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
]
TARGET = "WIN"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "team_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/features/build_team_features.py first."
        )
    return pd.read_parquet(path)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_features()
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Time-aware-ish split: shuffle split is fine for a first version,
    # but note in your README that a proper version should split by season
    # (train on earlier seasons, test on the most recent one) to avoid
    # any subtle leakage between games in the same season.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"Test accuracy:   {acc:.4f}")
    print(f"Log loss:        {ll:.4f}")
    print(f"Brier score:      {brier:.4f}")
    print("\nFeature importances:")
    for feat, imp in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp}")

    model_path = MODELS_DIR / "game_model_four_factors.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()
