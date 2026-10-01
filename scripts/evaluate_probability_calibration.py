"""Frozen chronological Platt/isotonic calibration and standings evaluation."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scripts.audit_fixed_cutoff_artifact import validate_calibration_inputs
from src.ingest.player_data import load_cached
from src.features.build_team_features import add_basic_fields

def score(y, p):
    p=np.clip(p,1e-6,1-1e-6); bins=np.minimum((p*10).astype(int),9)
    ece=sum((bins==b).sum()*abs(p[bins==b].mean()-y[bins==b].mean()) for b in range(10) if (bins==b).any())/len(y)
    return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),'ece':float(ece)}


def simulate_2024_standings(rows, probabilities, raw_dir, n_sims, seed):
    """Matched outcome-only standings simulation for frozen 2024-25 probabilities."""
    raw, _, _ = load_cached(raw_dir)
    team = add_basic_fields(raw)
    team['TEAM_ID'] = pd.to_numeric(team.TEAM_ID)
    output = []
    rows = rows.assign(calibrated_probability=np.asarray(probabilities))
    rows['home_team'] = pd.to_numeric(rows.home_team)
    rows['away_team'] = pd.to_numeric(rows.away_team)
    for index, ((season, cutoff), games) in enumerate(rows.groupby(['season', 'cutoff'], sort=False)):
        cutoff = pd.Timestamp(cutoff)
        season_rows = team[team.SEASON == season]
        current = season_rows[season_rows.GAME_DATE < cutoff].groupby('TEAM_ID').WIN.sum()
        actual = season_rows.groupby('TEAM_ID').WIN.sum()
        teams = sorted(set(games.home_team).union(games.away_team).union(current.index))
        positions = {team_id: position for position, team_id in enumerate(teams)}
        wins = np.tile(current.reindex(teams, fill_value=0).to_numpy(), (n_sims, 1)).astype(float)
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[index])
        draws = rng.random((n_sims, len(games))) < games.calibrated_probability.to_numpy()
        for game_index, game in enumerate(games.itertuples(index=False)):
            wins[:, positions[game.home_team]] += draws[:, game_index]
            wins[:, positions[game.away_team]] += ~draws[:, game_index]
        mean = wins.mean(axis=0)
        p05, p95 = np.quantile(wins, [.05, .95], axis=0)
        frame = pd.DataFrame({'TEAM_ID': teams, 'mean_wins': mean, 'p05': p05, 'p95': p95})
        frame['actual_wins'] = frame.TEAM_ID.map(actual)
        output.append(dict(
            season=season, cutoff=str(cutoff.date()), n_sims=n_sims,
            standings_mae=float((frame.mean_wins - frame.actual_wins).abs().mean()),
            top_five_mae=float(frame.nlargest(5, 'actual_wins').eval('mean_wins - actual_wins').abs().mean()),
            bottom_five_mae=float(frame.nsmallest(5, 'actual_wins').eval('mean_wins - actual_wins').abs().mean()),
            coverage_90=float(((frame.actual_wins >= frame.p05) & (frame.actual_wins <= frame.p95)).mean()),
        ))
    return pd.DataFrame(output)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--raw-dir',type=Path,default=Path('data/raw')); ap.add_argument('--n-sims',type=int,default=50)
    ap.add_argument('--seed',type=int,default=20260911); args=ap.parse_args()
    data=pd.read_parquet(args.run_dir/'game_probabilities.parquet')
    validate_calibration_inputs(args.run_dir, data)
    data=data[(data.model=='hybrid_development_ml') & (data.learned_uncertainty==False)].copy()
    data['year']=data.season.str[:4].astype(int)
    train=data[data.year<=2022]; val=data[data.year==2023]; test=data[data.year==2024]
    methods={'raw':lambda p:p}
    platt=LogisticRegression(C=1,max_iter=1000).fit(train[['probability']],train.target)
    iso=IsotonicRegression(out_of_bounds='clip').fit(train.probability,train.target)
    methods['platt']=lambda p:platt.predict_proba(np.asarray(p).reshape(-1,1))[:,1]
    methods['isotonic']=lambda p:iso.predict(np.asarray(p))
    validation={name:score(val.target.to_numpy(),fn(val.probability.to_numpy())) for name,fn in methods.items()}
    selected=min(('platt', 'isotonic'), key=lambda name: validation[name]['log_loss'])
    calibrated_test = methods[selected](test.probability.to_numpy())
    calibrated_standings = simulate_2024_standings(test, calibrated_test, args.raw_dir, args.n_sims, args.seed)
    raw_standings = pd.read_csv(args.run_dir/'per_cutoff_metrics.csv')
    raw_standings = raw_standings[(raw_standings.season == '2024-25') &
                                  (raw_standings.model == 'hybrid_development_ml') &
                                  (raw_standings.learned_uncertainty == False)].copy()
    standings_columns = ['standings_mae', 'top_five_mae', 'bottom_five_mae', 'coverage_90']
    result={'protocol':'Fit on 2018-19 to 2022-23 cutoffs; compare/select on held-out 2023-24; evaluate selected method once on 2024-25.',
            'selection_metric':'validation log loss (lower is better)',
            'validation':validation,'selected':selected,
            'final_test':{'raw':score(test.target.to_numpy(),test.probability.to_numpy()),
                          'selected':score(test.target.to_numpy(),calibrated_test)},
            'standings_protocol':'50-trial matched outcome-only simulation, averaged over four 2024-25 cutoffs.',
            'standings_raw_2024_25':raw_standings[['cutoff', *standings_columns]].to_dict(orient='records'),
            'standings_selected_2024_25':calibrated_standings.to_dict(orient='records'),
            'standings_raw_2024_25_mean':raw_standings[standings_columns].mean().to_dict(),
            'standings_selected_2024_25_mean':calibrated_standings[standings_columns].mean().to_dict()}
    (args.run_dir/'calibration_evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
