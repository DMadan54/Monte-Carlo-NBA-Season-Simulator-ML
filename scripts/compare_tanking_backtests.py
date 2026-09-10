import pandas as pd
from pathlib import Path

PROCESSED_DATA_DIR = Path("data/processed")

def main():
    # Load the two results
    df_all = pd.read_csv(PROCESSED_DATA_DIR / "walk_forward_backtest_results.csv")
    df_no_tank = pd.read_csv(PROCESSED_DATA_DIR / "walk_forward_backtest_results_no_tanking.csv")

    # Rename columns to merge
    df_all = df_all.rename(columns={
        "SIMULATED_MEAN_WINS": "SIM_ALL_WINS",
        "ABSOLUTE_ERROR": "ERR_ALL"
    })
    df_no_tank = df_no_tank.rename(columns={
        "SIMULATED_MEAN_WINS": "SIM_NOTANK_WINS",
        "ABSOLUTE_ERROR": "ERR_NOTANK"
    })

    # Merge
    merged = df_all.merge(df_no_tank[["SEASON", "TEAM", "ACTUAL_WINS", "SIM_NOTANK_WINS", "ERR_NOTANK"]], 
                          on=["SEASON", "TEAM"], 
                          suffixes=("", "_notank"))

    # Confirm actual-win totals are identical
    diff_actual = (merged["ACTUAL_WINS"] != merged["ACTUAL_WINS_notank"]).sum()
    print(f"Mismatch in actual wins: {diff_actual}")
    assert diff_actual == 0, "Actual wins mismatch between runs!"
    
    # Drop redundant column
    merged = merged.drop(columns=["ACTUAL_WINS_notank"])

    # Calculate overall MAE
    mae_all = merged["ERR_ALL"].mean()
    mae_notank = merged["ERR_NOTANK"].mean()

    # Calculate top-5 and bottom-5 MAE per season, then average across seasons
    top5_all_list = []
    top5_notank_list = []
    bot5_all_list = []
    bot5_notank_list = []

    for season, grp in merged.groupby("SEASON"):
        grp_sorted = grp.sort_values("ACTUAL_WINS", ascending=False)
        top5 = grp_sorted.head(5)
        bot5 = grp_sorted.tail(5)
        
        top5_all_list.append(top5["ERR_ALL"].mean())
        top5_notank_list.append(top5["ERR_NOTANK"].mean())
        bot5_all_list.append(bot5["ERR_ALL"].mean())
        bot5_notank_list.append(bot5["ERR_NOTANK"].mean())

    top5_all_mae = pd.Series(top5_all_list).mean()
    top5_notank_mae = pd.Series(top5_notank_list).mean()
    bot5_all_mae = pd.Series(bot5_all_list).mean()
    bot5_notank_mae = pd.Series(bot5_notank_list).mean()

    print("\n--- REPORT ---")
    print(f"Overall MAE:      All = {mae_all:.3f} | NoTank = {mae_notank:.3f} | Delta = {mae_notank - mae_all:+.3f}")
    print(f"Top-5 Teams MAE:  All = {top5_all_mae:.3f} | NoTank = {top5_notank_mae:.3f} | Delta = {top5_notank_mae - top5_all_mae:+.3f}")
    print(f"Bottom-5 Teams MAE: All = {bot5_all_mae:.3f} | NoTank = {bot5_notank_mae:.3f} | Delta = {bot5_notank_mae - bot5_all_mae:+.3f}")
    
    # Save comparison CSV
    out_path = PROCESSED_DATA_DIR / "walk_forward_comparison.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved comparison CSV to {out_path}")

if __name__ == "__main__":
    main()
