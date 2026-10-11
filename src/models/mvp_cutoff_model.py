"""Fixed-March-1 MVP winner forecasts using only earlier game logs."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["games", "games_fraction", "minutes_per_game", "pts_per_game",
            "reb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
            "true_shooting", "plus_minus_per_game", "team_win_pct",
            "team_margin_per_game", "team_count", "normal_path_open"]
ALPHAS = (0.03, 0.3, 3.0)
BASELINE_BETAS = (0.5, 1.0, 2.0, 4.0, 8.0)


def cutoff_for_season(season):
    """March 1 of the season end year; observations on that date are excluded."""
    return pd.Timestamp(year=int(season[:4]) + 1, month=3, day=1)


def build_cutoff_candidates(players, teams, mapped):
    p = players.copy()
    t = teams.copy()
    p["GAME_DATE"] = pd.to_datetime(p.GAME_DATE).dt.normalize()
    t["GAME_DATE"] = pd.to_datetime(t.GAME_DATE).dt.normalize()
    p["PLAYER_ID"] = p.PLAYER_ID.astype(str)
    p["TEAM_ID"] = p.TEAM_ID.astype(str)
    t["TEAM_ID"] = t.TEAM_ID.astype(str)
    p["cutoff"] = p.SEASON.map(cutoff_for_season)
    t["cutoff"] = t.SEASON.map(cutoff_for_season)
    p = p[p.GAME_DATE < p.cutoff].copy()
    t = t[t.GAME_DATE < t.cutoff].copy()
    if p.duplicated(["SEASON", "GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate player game before cutoff")
    team = t.groupby(["SEASON", "TEAM_ID"]).agg(
        team_games=("GAME_ID", "nunique"), team_wins=("WL", lambda x: int((x == "W").sum())),
        team_margin_per_game=("PLUS_MINUS", "mean")).reset_index()
    team["team_win_pct"] = team.team_wins / team.team_games
    # The latest observed team is known at the cutoff, including players traded midseason.
    affiliation = p.groupby(["SEASON", "PLAYER_ID", "TEAM_ID"]).agg(
        first_seen=("GAME_DATE", "min"), last_seen=("GAME_DATE", "max"),
        affiliation_minutes=("MIN", "sum")).reset_index()
    current = affiliation.sort_values(["SEASON", "PLAYER_ID", "last_seen", "affiliation_minutes", "TEAM_ID"],
        ascending=[True, True, False, False, True]).drop_duplicates(["SEASON", "PLAYER_ID"])
    team_count = affiliation.groupby(["SEASON", "PLAYER_ID"]).TEAM_ID.nunique().rename("team_count").reset_index()
    totals = p.groupby(["SEASON", "PLAYER_ID"]).agg(
        player_name=("PLAYER_NAME", "first"), games=("GAME_ID", "nunique"),
        minutes=("MIN", "sum"), pts=("PTS", "sum"), reb=("REB", "sum"),
        ast=("AST", "sum"), stl=("STL", "sum"), blk=("BLK", "sum"),
        fga=("FGA", "sum"), fta=("FTA", "sum"), plus_minus=("PLUS_MINUS", "sum"),
        qualifying_20=("MIN", lambda x: int((x >= 20).sum())),
        qualifying_15_19=("MIN", lambda x: int(((x >= 15) & (x < 20)).sum()))).reset_index()
    for stat in ["minutes", "pts", "reb", "ast", "stl", "blk", "plus_minus"]:
        totals[f"{stat}_per_game"] = totals[stat] / totals.games
    totals["true_shooting"] = totals.pts / (2 * (totals.fga + 0.44 * totals.fta)).replace(0, np.nan)
    c = totals.merge(current[["SEASON", "PLAYER_ID", "TEAM_ID"]], on=["SEASON", "PLAYER_ID"], validate="one_to_one")
    c = c.merge(team_count, on=["SEASON", "PLAYER_ID"], validate="one_to_one")
    c = c.merge(team, on=["SEASON", "TEAM_ID"], validate="many_to_one")
    c["games_fraction"] = c.games / c.team_games
    c["season_end_year"] = c.SEASON.str[:4].astype(int) + 1
    c["qualifying_games_so_far"] = c.qualifying_20 + np.minimum(2, c.qualifying_15_19)
    c["remaining_team_games_assumed"] = np.maximum(0, 82 - c.team_games)
    c["normal_path_open"] = ((c.season_end_year < 2024) |
        (c.qualifying_games_so_far + c.remaining_team_games_assumed >= 65)).astype(int)
    c = c.rename(columns={"SEASON": "season", "PLAYER_ID": "player_id", "TEAM_ID": "current_team_id"})
    c["cutoff"] = c.season.map(cutoff_for_season)
    c = c.merge(mapped[["season", "player_id", "vote_share", "is_winner"]],
                on=["season", "player_id"], how="left", validate="one_to_one")
    c["vote_share"] = c.vote_share.fillna(0.0)
    c["is_winner"] = c.is_winner.fillna(False).astype(bool)
    return c


def baseline_score(frame):
    return (frame.pts_per_game.clip(lower=0) * frame.team_win_pct.clip(lower=0) *
            frame.games_fraction.clip(lower=0, upper=1))


def fit_model(train, c):
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=0))
    model.fit(train[FEATURES], train.is_winner.astype(int))
    return model


def normalized_probability(raw):
    raw = np.asarray(raw, dtype=float)
    return raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw))


def baseline_probability(frame, beta):
    score = baseline_score(frame).to_numpy(dtype=float)
    z = (score - score.mean()) / max(score.std(), 1e-9)
    logits = beta * z
    shifted = np.exp(logits - logits.max())
    return normalized_probability(shifted)


def fit_baseline_beta(train):
    losses = {}
    for beta in BASELINE_BETAS:
        losses[beta] = np.mean([
            -np.log(max(baseline_probability(season, beta)[season.is_winner.to_numpy()][0], 1e-12))
            for _, season in train.groupby("season")])
    return min(BASELINE_BETAS, key=lambda beta: (losses[beta], beta))


def evaluate(candidates, first_test=2018):
    predictions, metrics, weights = [], [], []
    for year in range(first_test, 2026):
        train = candidates[(candidates.season_end_year >= 2016) & (candidates.season_end_year < year)].copy()
        test = candidates[candidates.season_end_year == year].copy()
        if test.empty or train.season_end_year.nunique() < 2 or test.is_winner.sum() != 1:
            raise ValueError(f"Incomplete chronological fold {year}")
        inner_years = sorted(train.season_end_year.unique())[2:]
        if inner_years:
            losses = {}
            for c in ALPHAS:
                fold_losses = []
                for validation_year in inner_years:
                    inner_train = train[train.season_end_year < validation_year]
                    inner_val = train[train.season_end_year == validation_year]
                    model = fit_model(inner_train, c)
                    probs = normalized_probability(model.predict_proba(inner_val[FEATURES])[:, 1])
                    fold_losses.append(-np.log(max(probs[inner_val.is_winner.to_numpy()][0], 1e-12)))
                losses[c] = float(np.mean(fold_losses))
            selected_c = min(ALPHAS, key=lambda c: (losses[c], c))
        else:
            selected_c = 0.3
        model = fit_model(train, selected_c)
        baseline_beta = fit_baseline_beta(train)
        test["model_probability"] = normalized_probability(model.predict_proba(test[FEATURES])[:, 1])
        test["baseline_score"] = baseline_score(test)
        test["baseline_probability"] = baseline_probability(test, baseline_beta)
        test["model_rank"] = test.model_probability.rank(ascending=False, method="min")
        test["baseline_rank"] = test.baseline_probability.rank(ascending=False, method="min")
        for method, score in [("model", "model_probability"), ("baseline", "baseline_probability")]:
            order = test.sort_values(score, ascending=False)
            winner_probability = float(test.loc[test.is_winner, score].iloc[0])
            recipients = test[test.vote_share > 0]
            corr = spearmanr(recipients[score], recipients.vote_share).statistic
            metrics.append({"test_year": year, "method": method, "train_through": year - 1,
                            "cutoff": str(test.cutoff.iloc[0].date()),
                            "selected_c": selected_c if method == "model" else np.nan,
                            "baseline_beta": baseline_beta if method == "baseline" else np.nan,
                            "winner_correct": bool(order.is_winner.iloc[0]),
                            "winner_rank": int(test.loc[test.is_winner, score.replace("probability", "rank")].iloc[0]),
                            "winner_top3": bool(order.is_winner.iloc[:3].any()),
                            "recipient_recall_at_10": float((order.vote_share.iloc[:10] > 0).sum() / (test.vote_share > 0).sum()),
                            "winner_log_loss": -np.log(max(winner_probability, 1e-12)),
                            "multiclass_brier": float(np.sum((test[score] - test.is_winner.astype(int)) ** 2)),
                            "recipient_spearman": float(corr) if np.isfinite(corr) else np.nan,
                            "top_probability": float(order[score].iloc[0])})
        weights.append({"test_year": year, "selected_c": selected_c,
                        **dict(zip(FEATURES, model[-1].coef_[0]))})
        predictions.append(test[["season", "season_end_year", "cutoff", "player_id", "player_name",
                                 "current_team_id", "is_winner", "vote_share", "model_probability",
                                 "baseline_probability", "model_rank", "baseline_rank",
                                 "qualifying_games_so_far", "remaining_team_games_assumed", *FEATURES]])
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(weights)
