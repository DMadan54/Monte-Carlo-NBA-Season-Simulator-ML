import numpy as np
import pandas as pd

# Hard‑coded conference map for 2024‑25 season
CONFERENCE_MAP = {
    # Eastern Conference
    "ATL": "East", "BKN": "East", "BOS": "East", "CHA": "East", "CHI": "East",
    "CLE": "East", "DET": "East", "IND": "East", "MIL": "East", "MIA": "East",
    "NYK": "East", "ORL": "East", "PHI": "East", "TOR": "East", "WAS": "East",
    # Western Conference
    "DAL": "West", "DEN": "West", "GSW": "West", "HOU": "West", "LAC": "West",
    "LAL": "West", "MEM": "West", "MIN": "West", "NOP": "West", "PHX": "West",
    "POR": "West", "SAC": "West", "SAS": "West", "UTA": "West", "OKC": "West",
}

def get_latest_features(team_abbr: str, team_feat: pd.DataFrame) -> pd.Series:
    """Return the most recent rolling feature row for *team_abbr* (as a Series)."""
    df = team_feat[team_feat["TEAM_ABBREVIATION"] == team_abbr]
    if df.empty:
        raise ValueError(f"No feature rows for team {team_abbr}")
    return df.sort_values("GAME_DATE").iloc[-1]

def simulate_series(home: str, away: str, model, team_feat: pd.DataFrame) -> (str, int):
    """Simulate a best‑of‑7 series between *home* and *away*.
    Returns a tuple (winning_team_abbr, games_played)."""
    home_feat = get_latest_features(home, team_feat).copy()
    away_feat = get_latest_features(away, team_feat).copy()
    home_feat["IS_HOME"] = 1
    away_feat["IS_HOME"] = 0
    FEATURES = [
        "ROLL_WIN_PCT", "ROLL_POINT_DIFF", "IS_HOME", "DAYS_SINCE_LAST_GAME",
        "ROLL_EFG", "ROLL_TOV_PCT", "ROLL_ORB_PCT", "ROLL_FTR",
        "ROLL_EFG_OPP", "ROLL_TOV_PCT_OPP", "ROLL_ORB_PCT_OPP", "ROLL_FTR_OPP",
        "ROLL_EFG_DIFF",
    ]
    prob_home_win = model.predict_proba([home_feat[FEATURES]])[0, 1]
    wins_home = wins_away = games = 0
    while True:
        games += 1
        if np.random.random() < prob_home_win:
            wins_home += 1
        else:
            wins_away += 1
        if wins_home == 4 or wins_away == 4:
            break
        if games >= 7:
            break
    winner = home if wins_home == 4 else away
    return winner, games

def simulate_playoffs(final_wins: dict, model, team_feat: pd.DataFrame) -> (str, dict):
    """Simulate the full NBA playoffs (including play‑in) and return the champion.
    Also returns a dict with per‑conference playoff qualifiers (set of team abbreviations).
    """
    conf_teams = {"East": [], "West": []}
    for team, wins in final_wins.items():
        conf = CONFERENCE_MAP[team]
        conf_teams[conf].append((team, wins))
    seeds = {}
    for conf, lst in conf_teams.items():
        sorted_lst = sorted(lst, key=lambda x: (-x[1], x[0]))
        seeds[conf] = [team for team, _ in sorted_lst]

    qualifiers = {"East": set(), "West": set()}
    def play_in(conf: str):
        s7, s8, s9, s10 = seeds[conf][6:10]
        win78, _ = simulate_series(s7, s8, model, team_feat)
        lose78 = s7 if win78 == s8 else s8
        win910, _ = simulate_series(s9, s10, model, team_feat)
        win8, _ = simulate_series(lose78, win910, model, team_feat)
        final_7 = win78
        final_8 = win8
        new_list = seeds[conf][:6] + [final_7, final_8]
        seeds[conf] = new_list
        qualifiers[conf].update([final_7, final_8])
    play_in("East")
    play_in("West")
    for conf in ("East", "West"):
        qualifiers[conf].update(seeds[conf][:6])

    def bracket_round(current_seeds: list) -> list:
        pairings = [(current_seeds[i], current_seeds[-1 - i]) for i in range(4)]
        winners = []
        for home, away in pairings:
            win, _ = simulate_series(home, away, model, team_feat)
            winners.append(win)
        return winners

    def conference_champion(conf: str) -> str:
        qf_winners = bracket_round(seeds[conf])
        sf_winners = bracket_round(qf_winners)
        champ, _ = simulate_series(sf_winners[0], sf_winners[1], model, team_feat)
        return champ

    east_champ = conference_champion("East")
    west_champ = conference_champion("West")
    champion, _ = simulate_series(east_champ, west_champ, model, team_feat)
    return champion, qualifiers

def run_full_simulation(n_sims: int = 200, seed: int = 42):
    """Run *n_sims* complete season + playoff simulations.
    Returns a DataFrame with mean regular‑season wins, playoff qualification rate,
    and title percentage for each team.
    """
    import sys
    sys.path.append(r"C:/Users/dhruv/OneDrive/Desktop/Monte Carlo NBA")
    from src.models.season_sim import load_remaining_schedule, load_model
    from src.models.season_sim import PROCESSED_DATA_DIR
    np.random.seed(seed)
    model = load_model()
    schedule = load_remaining_schedule()
    team_feat = pd.read_parquet(PROCESSED_DATA_DIR / "team_features.parquet")
    win_sum = {team: 0 for team in CONFERENCE_MAP}
    playoff_qual = {team: 0 for team in CONFERENCE_MAP}
    titles = {team: 0 for team in CONFERENCE_MAP}
    for _ in range(n_sims):
        from src.models.season_sim import simulate_one_season
        final_wins = simulate_one_season(schedule, model, {team: 0 for team in CONFERENCE_MAP})
        for t, w in final_wins.items():
            win_sum[t] += w
        champ, qualifiers = simulate_playoffs(final_wins, model, team_feat)
        titles[champ] += 1
        for conf in ("East", "West"):
            for t in qualifiers[conf]:
                playoff_qual[t] += 1
    df = pd.DataFrame({
        "team": list(CONFERENCE_MAP.keys()),
        "mean_wins": [win_sum[t] / n_sims for t in CONFERENCE_MAP],
        "playoff_rate": [playoff_qual[t] / n_sims for t in CONFERENCE_MAP],
        "title_pct": [titles[t] / n_sims * 100 for t in CONFERENCE_MAP],
    })
    return df.sort_values("mean_wins", ascending=False).reset_index(drop=True)
