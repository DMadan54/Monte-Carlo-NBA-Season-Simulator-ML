"""Retrospective season-end MVP vote-share experiment."""
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["games", "minutes", "pts_per_game", "reb_per_game", "ast_per_game",
            "stl_per_game", "blk_per_game", "true_shooting", "plus_minus_per_game",
            "team_win_pct", "team_wins"]
TEAM_ABBR_ALIASES = {"PHO": "PHX", "BRK": "BKN"}


def name_key(name):
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", ascii_name)


def resolve_votes(votes, players, aliases):
    """Use season/name only when unique; require explicit ID for exceptions."""
    identity = players[["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]].drop_duplicates().copy()
    identity["name_key"] = identity.PLAYER_NAME.map(name_key)
    alias_ids = dict(zip(aliases.vote_name, aliases.player_id.astype(str)))
    rows = []
    issues = []
    for vote in votes.itertuples(index=False):
        choices = identity[(identity.SEASON == vote.season) &
                           (identity.name_key == name_key(vote.player_name))]
        source = "unique_normalized_name"
        if vote.player_name in alias_ids:
            choices = identity[(identity.SEASON == vote.season) &
                               (identity.PLAYER_ID.astype(str) == alias_ids[vote.player_name])]
            source = "reviewed_alias"
        ids = choices.PLAYER_ID.astype(str).unique()
        if len(ids) != 1:
            issues.append({"season": vote.season, "vote_name": vote.player_name,
                           "candidate_ids": "|".join(ids), "reason": "unmatched" if not len(ids) else "ambiguous"})
            continue
        log_teams = sorted(choices.TEAM_ABBREVIATION.dropna().unique())
        canonical_vote_team = TEAM_ABBR_ALIASES.get(vote.team_abbr, vote.team_abbr)
        if canonical_vote_team not in log_teams:
            issues.append({"season": vote.season, "vote_name": vote.player_name,
                           "candidate_ids": ids[0], "reason": "team_mismatch"})
            continue
        rows.append({"season": vote.season, "vote_name": vote.player_name,
                     "player_id": ids[0], "log_name": choices.PLAYER_NAME.iloc[0],
                     "match_method": source, "vote_team": vote.team_abbr,
                     "log_teams": "|".join(log_teams),
                     "vote_share": vote.vote_share, "is_winner": vote.is_winner})
    mapped = pd.DataFrame(rows)
    if not mapped.empty and mapped.duplicated(["season", "player_id"]).any():
        raise ValueError("Multiple vote rows mapped to one player-season")
    return mapped, pd.DataFrame(issues, columns=["season", "vote_name", "candidate_ids", "reason"])


def build_candidates(players, teams, mapped):
    p = players.copy()
    p["PLAYER_ID"] = p.PLAYER_ID.astype(str)
    p["TEAM_ID"] = p.TEAM_ID.astype(str)
    t = teams.copy()
    t["TEAM_ID"] = t.TEAM_ID.astype(str)
    if p.duplicated(["SEASON", "GAME_ID", "PLAYER_ID"]).any():
        raise ValueError("Duplicate player-game rows")
    team_seasons = t.groupby(["SEASON", "TEAM_ID"]).agg(
        team_wins=("WL", lambda x: int((x == "W").sum())),
        team_games=("GAME_ID", "nunique")).reset_index()
    team_seasons["team_win_pct"] = team_seasons.team_wins / team_seasons.team_games
    affiliation = p.groupby(["SEASON", "PLAYER_ID", "TEAM_ID"], as_index=False).MIN.sum()
    primary = affiliation.sort_values(["SEASON", "PLAYER_ID", "MIN", "TEAM_ID"],
                                      ascending=[True, True, False, True]).drop_duplicates(["SEASON", "PLAYER_ID"])
    totals = p.groupby(["SEASON", "PLAYER_ID"]).agg(
        player_name=("PLAYER_NAME", "first"), games=("GAME_ID", "nunique"), minutes=("MIN", "sum"),
        pts=("PTS", "sum"), reb=("REB", "sum"), ast=("AST", "sum"),
        stl=("STL", "sum"), blk=("BLK", "sum"), fga=("FGA", "sum"),
        fta=("FTA", "sum"), plus_minus=("PLUS_MINUS", "sum")).reset_index()
    for stat in ["pts", "reb", "ast", "stl", "blk", "plus_minus"]:
        totals[f"{stat}_per_game"] = totals[stat] / totals.games
    totals["true_shooting"] = totals.pts / (2 * (totals.fga + 0.44 * totals.fta)).replace(0, np.nan)
    c = totals.merge(primary[["SEASON", "PLAYER_ID", "TEAM_ID"]], on=["SEASON", "PLAYER_ID"], validate="one_to_one")
    c = c.merge(team_seasons, on=["SEASON", "TEAM_ID"], validate="many_to_one")
    c = c.rename(columns={"SEASON": "season", "PLAYER_ID": "player_id", "TEAM_ID": "primary_team_id"})
    c = c.merge(mapped[["season", "player_id", "vote_share", "is_winner"]],
                on=["season", "player_id"], how="left", validate="one_to_one")
    c["vote_share"] = c.vote_share.fillna(0.0)
    c["is_winner"] = c.is_winner.fillna(False).astype(bool)
    c["season_end_year"] = c.season.str[:4].astype(int) + 1
    return c


def baseline_score(frame):
    # Fixed transparent rule, not fitted on test years.
    return frame.pts_per_game * frame.team_win_pct * (frame.games / frame.groupby("season").games.transform("max"))


def evaluate(candidates, alphas=(0.1, 1.0, 10.0, 100.0), first_test=2018):
    predictions, metrics, coefficients = [], [], []
    for year in range(first_test, 2026):
        train = candidates[(candidates.season_end_year >= 2016) & (candidates.season_end_year < year)].copy()
        test = candidates[candidates.season_end_year == year].copy()
        if test.empty or train.season_end_year.nunique() < 2:
            raise ValueError(f"Insufficient chronological data for {year}")
        # Select penalty using only earlier seasons, where an inner fold is available.
        inner_years = sorted(train.season_end_year.unique())[2:]
        alpha_losses = {}
        for alpha in alphas:
            losses = []
            for inner_year in inner_years:
                fit = train[train.season_end_year < inner_year]
                val = train[train.season_end_year == inner_year]
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
                m.fit(fit[FEATURES], fit.vote_share)
                losses.append(np.mean((np.clip(m.predict(val[FEATURES]), 0, 1) - val.vote_share) ** 2))
            alpha_losses[alpha] = float(np.mean(losses)) if losses else np.nan
        alpha = min(alphas, key=lambda a: (np.inf if np.isnan(alpha_losses[a]) else alpha_losses[a], a)) if inner_years else 10.0
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[FEATURES], train.vote_share)
        test["pred_share"] = np.clip(model.predict(test[FEATURES]), 0, 1)
        test["baseline_score"] = baseline_score(test)
        test["baseline_share"] = np.clip(test.baseline_score / train.assign(b=baseline_score(train)).b.max(), 0, 1)
        test["model_rank"] = test.pred_share.rank(ascending=False, method="min")
        test["baseline_rank"] = test.baseline_score.rank(ascending=False, method="min")
        winner_id = test.loc[test.is_winner, "player_id"].iloc[0]
        for method, score, share in [("ridge", "pred_share", "pred_share"),
                                     ("baseline", "baseline_score", "baseline_share")]:
            rank_corr = spearmanr(test[score], test.vote_share).statistic
            recipients = test[test.vote_share > 0]
            recipient_corr = spearmanr(recipients[score], recipients.vote_share).statistic
            metrics.append({"test_year": year, "method": method, "train_through": year - 1,
                            "train_seasons": train.season_end_year.nunique(), "alpha": alpha if method == "ridge" else np.nan,
                            "winner_correct": bool(test.loc[test[score].idxmax(), "player_id"] == winner_id),
                            "winner_rank": int(test[score].rank(ascending=False, method="min")[test.is_winner].iloc[0]),
                            "spearman": float(rank_corr) if np.isfinite(rank_corr) else np.nan,
                            "recipient_spearman": float(recipient_corr) if np.isfinite(recipient_corr) else np.nan,
                            "share_mae_all": float(np.mean(abs(test[share] - test.vote_share))),
                            "share_mae_recipients": float(np.mean(abs(test.loc[test.vote_share > 0, share] - test.loc[test.vote_share > 0, "vote_share"])))})
        coefficients.append({"test_year": year, "alpha": alpha, **dict(zip(FEATURES, model[-1].coef_))})
        predictions.append(test[["season", "season_end_year", "player_id", "player_name", "is_winner", "vote_share",
                                 "pred_share", "baseline_share", "model_rank", "baseline_rank", *FEATURES]])
    return pd.concat(predictions), pd.DataFrame(metrics), pd.DataFrame(coefficients)
