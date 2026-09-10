"""Paired block uncertainty and error slices for an already-frozen experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.player_game_model import metrics
from src.ingest.player_data import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    selection = json.loads((run / 'selection.json').read_text())
    name = selection['experimental_player_selected']
    base = pd.read_csv(run / 'team_linear_test_predictions.csv', dtype={'GAME_ID': str})
    selected = pd.read_csv(run / f'{name}_test_predictions.csv', dtype={'GAME_ID': str})
    merged = base.merge(selected[['GAME_ID', 'probability']], on='GAME_ID', suffixes=('_base', '_player'), validate='one_to_one')
    y = merged.TARGET.to_numpy()
    def losses(p):
        p = np.clip(p, 1e-8, 1-1e-8)
        return -(y * np.log(p) + (1-y) * np.log(1-p))
    merged['loss_difference'] = losses(merged.probability_player.to_numpy()) - losses(merged.probability_base.to_numpy())
    merged['week'] = pd.to_datetime(merged.GAME_DATE).dt.to_period('W').astype(str)
    blocks = merged.groupby('week').loss_difference.agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(blocks), size=(2000, len(blocks)))
    samples = blocks[indices].sum(axis=1)
    differences = samples[:, 0] / samples[:, 1]
    forecasts = pd.read_parquet(run / 'player_forecasts.parquet')
    coverage = forecasts.groupby('GAME_ID').TARGET_MIN.sum() / 480
    merged['captured_minutes'] = merged.GAME_ID.map(coverage).fillna(0)
    slices = {}
    for label, mask in [('roster_estimate_under_90pct', merged.captured_minutes < .9),
                        ('roster_estimate_at_least_90pct', merged.captured_minutes >= .9)]:
        if mask.any():
            slices[label] = metrics(merged.loc[mask, 'TARGET'], merged.loc[mask, 'probability_player'])
    report = dict(selected_model=name, paired_log_loss_difference=float(merged.loss_difference.mean()),
        weekly_block_bootstrap_95pct=np.quantile(differences, [.025, .975]).tolist(),
        bootstrap_replicates=2000, seed=42,
        uncertainty_limit='One test season; weekly blocks are an approximation and do not remove all team/season dependence',
        roster_slices=slices,
        slice_limit='Captured actual minutes are retrospective diagnostics only, never prediction inputs')
    output = run / 'validation_audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    paths = [*Path('src').rglob('*.py'), *Path('tests').glob('*.py'), *Path('scripts').glob('*player*.py'),
             Path('scripts/run_development_experiment.py'), Path('scripts/run_cutoff_scenario.py')]
    (run / 'source_manifest.json').write_text(json.dumps({str(p): file_hash(p) for p in paths}, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
