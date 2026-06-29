"""
End-to-end pipeline orchestrator.

Runs every stage in order and writes CSV outputs into ../outputs/ that
the Streamlit dashboard reads. The trained spoilage model is pickled
to ../models/ so the dashboard can re-score without retraining.

USAGE
    cd code/
    python run_pipeline.py

ETA on a typical laptop: 5-15 minutes (Prophet fits per category dominate).

If you want to skip Prophet (it's the slowest), pass --skip-prophet.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_loading
import features
import demand_forecast
import elasticity
import spoilage
import optimizer
import simulation
import sensitivity


def banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {msg}\n{bar}")


def main(skip_prophet: bool = False, skip_dunn: bool = False) -> None:
    config.ensure_dirs()
    t_total = time.time()

    # ─────────────────────────────────────────────────────────────────────
    # Stage 0 — Load + split
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 0 — Load perishable + 3-way split")
    perishable = data_loading.load_perishable()
    print(f"  Cleaned rows: {len(perishable):,} | "
          f"Categories: {sorted(perishable['category'].unique())}")
    ts_pivot, price_daily = data_loading.build_daily_aggregates(perishable)
    train_dates, val_dates, holdout_dates = data_loading.make_three_way_split(ts_pivot)
    print(f"  Train: {train_dates[0].date()} → {train_dates[-1].date()} "
          f"({len(train_dates)} days)")
    print(f"  Val  : {val_dates[0].date()} → {val_dates[-1].date()} "
          f"({len(val_dates)} days)")
    print(f"  Hold : {holdout_dates[0].date()} → {holdout_dates[-1].date()} "
          f"({len(holdout_dates)} days)  ← SACRED")

    # ─────────────────────────────────────────────────────────────────────
    # Stage 1 — Demand forecasting (3-way comparison + selection)
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 1 — Demand forecasting")
    feats = features.build_demand_feature_set(
        ts_pivot, price_daily, train_dates, val_dates, holdout_dates,
    )

    print("[LightGBM] training + 5-fold CV + validation eval...")
    lgb_res = demand_forecast.evaluate_lightgbm(
        feats["X_train"], feats["y_train"],
        feats["X_val"], feats["y_val"], feats["feat_val"],
    )
    print(f"  LGB val_RMSE={lgb_res['val_rmse']}  dir_acc={lgb_res['dir_acc']}%")

    ts_train = ts_pivot.loc[train_dates]

    if not skip_prophet:
        print("[Prophet] per-category fits (slow)...")
        pro_res = demand_forecast.evaluate_prophet(
            ts_train, val_dates, holdout_dates, feats["feat_val"],
        )
        print(f"  Prophet val_RMSE={pro_res['val_rmse']}  dir_acc={pro_res['dir_acc']}%")
    else:
        # Skip path: stub Prophet so it cannot win
        pro_res = {"name": "Prophet", "val_rmse": 9e9, "val_mae": 9e9, "dir_acc": 0.0,
                   "cat_rmse": pd.Series(dtype=float)}
        print("  [skipped]")

    print("[Holt-Winters] per-category fits...")
    hw_res = demand_forecast.evaluate_holtwinters(ts_train, val_dates, feats["feat_val"])
    print(f"  HW val_RMSE={hw_res['val_rmse']}  dir_acc={hw_res['dir_acc']}%")

    winner_name, comparison_df = demand_forecast.select_winner(lgb_res, pro_res, hw_res)
    print(f"  Winner: {winner_name}")
    comparison_df.to_csv(config.OUTPUTS_DIR / "demand_model_comparison.csv", index=False)

    print(f"[{winner_name}] retraining on train+val → forecasting holdout...")
    cat_forecasts = demand_forecast.forecast_holdout(
        winner_name, ts_pivot,
        feats["feat_train"], feats["feat_val"], feats["feat_holdout"], holdout_dates,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Stage 2 — Elasticity (Dunnhumby OOS validation)
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 2 — Elasticity (Dunnhumby OOS validation)")
    if skip_dunn or not config.DUNN_DIR.exists():
        print("  [Dunnhumby skipped — using DEFAULT_ELASTICITY_MAP]")
        elasticity_map = dict(config.DEFAULT_ELASTICITY_MAP)
        oos_df = pd.DataFrame([
            {"Department": d, "Elasticity": v,
             "N_Train": np.nan, "N_Test": np.nan,
             "InSample_R2": np.nan, "OOS_R2": np.nan}
            for d, v in elasticity_map.items()
        ])
    else:
        print("  Loading transactions + product table...")
        elas_df = elasticity.load_dunnhumby_elasticity_table()
        oos_df = elasticity.run_elasticity_validation(elas_df)
        elasticity_map = elasticity.elasticity_map_from_validation(oos_df)
        print(oos_df.to_string(index=False))
    oos_df.to_csv(config.OUTPUTS_DIR / "elasticity_estimates.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 3 — Spoilage classifier
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 3 — Spoilage classifier (4-way comparison)")
    df_opt, df_pre, df_hold = spoilage.prepare_spoilage_dataset(perishable, holdout_dates)
    print(f"  Pre-holdout rows: {len(df_pre):,} | Holdout rows: {len(df_hold):,}")
    spoil_model, train_cols, spoil_cmp_df, spoil_winner = spoilage.train_and_select(df_pre)
    print(spoil_cmp_df.to_string(index=False))
    print(f"  Winner: {spoil_winner}")
    spoil_cmp_df.to_csv(config.OUTPUTS_DIR / "spoilage_model_comparison.csv", index=False)

    df_hold, df_opt = spoilage.score_holdout(spoil_model, train_cols, df_hold, df_opt)
    waste_frac = spoilage.empirical_waste_fraction(df_pre)
    print(f"  Empirical waste fraction (pre-holdout): {waste_frac:.4f}")

    # Persist artefacts so the dashboard does not have to retrain anything
    with open(config.MODELS_DIR / "spoilage_model.pkl", "wb") as f:
        pickle.dump({"model": spoil_model, "train_cols": train_cols,
                     "winner": spoil_winner, "waste_frac": waste_frac}, f)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 4 — Scale forecasts → Run 60-day simulation
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 4 — 60-day OOS simulation (3 strategies)")
    sku_forecasts = optimizer.scale_forecasts_to_sku(cat_forecasts, df_hold)
    print(f"  Scaled forecasts: {len(sku_forecasts)} (date, category) entries")
    print(f"  Mean SKU-day forecast: {np.mean(list(sku_forecasts.values())):.1f}")
    print(f"  Actual holdout units_sold mean: {df_hold['units_sold'].mean():.1f}")

    sim_df = df_hold.copy().reset_index(drop=True)

    # Save the holdout rows (with all features needed by batch_simulate) so the
    # dashboard's what-if section can re-run the optimizer on a sample without
    # touching the raw datasets. We also serialise the SKU-level forecasts.
    whatif_cols = list(dict.fromkeys(
        config.SPOILAGE_FEATURES + [
            "transaction_date", "category", "product_id",
            "base_price", "cost_price", "units_sold", "initial_quantity",
        ]
    ))
    whatif_sample = sim_df[whatif_cols].sample(
        n=min(500, len(sim_df)), random_state=config.RANDOM_STATE,
    ).reset_index(drop=True)
    whatif_sample.to_csv(config.OUTPUTS_DIR / "whatif_sample.csv", index=False)

    fc_rows = [{"date": d, "category": c, "forecast": v}
               for (d, c), v in sku_forecasts.items()]
    pd.DataFrame(fc_rows).to_csv(
        config.OUTPUTS_DIR / "sku_demand_forecasts.csv", index=False,
    )

    sim_kwargs = dict(spoilage_model=spoil_model, train_cols=train_cols,
                      elasticity_map=elasticity_map, waste_frac=waste_frac,
                      demand_forecasts=sku_forecasts)
    sim_results = simulation.run_full_simulation(sim_df, **sim_kwargs)
    sim_results.to_csv(config.OUTPUTS_DIR / "sim_results.csv", index=False)
    print(f"  Simulation rows: {len(sim_results):,}")

    kpi = simulation.kpi_summary(sim_results)
    kpi.to_csv(config.OUTPUTS_DIR / "strategy_kpi.csv")
    delta = simulation.delta_table(kpi, winner_name, spoil_winner)
    delta.to_csv(config.OUTPUTS_DIR / "strategy_delta_table.csv", index=False)
    print(delta.to_string(index=False))

    cat_breakdown = simulation.category_profit_breakdown(sim_results)
    cat_breakdown.to_csv(config.OUTPUTS_DIR / "category_profit.csv", index=False)
    bucket_tbl = simulation.near_expiry_buckets(sim_results)
    bucket_tbl.to_csv(config.OUTPUTS_DIR / "near_expiry_buckets.csv", index=False)
    daily_tbl = simulation.daily_profit_trend(sim_results)
    daily_tbl.to_csv(config.OUTPUTS_DIR / "daily_profit_trend.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────
    # Stage 5 — Sensitivity
    # ─────────────────────────────────────────────────────────────────────
    banner("Stage 5 — Sensitivity")
    hs_kpi, hs_n = sensitivity.high_spoilage_kpi(sim_df, **sim_kwargs)
    hs_kpi.to_csv(config.OUTPUTS_DIR / "high_spoilage_kpi.csv")
    print(f"  High-spoilage subset: {hs_n} rows")
    print(hs_kpi.to_string())

    elas_sens = sensitivity.elasticity_sensitivity(
        sim_df, elasticity_map=elasticity_map, **{k: v for k, v in sim_kwargs.items() if k != "elasticity_map"})
    elas_sens.to_csv(config.OUTPUTS_DIR / "elasticity_sensitivity.csv", index=False)
    print(elas_sens.to_string(index=False))

    opt_df = sensitivity.optimizer_stability_sample(
        df_pre, spoilage_model=spoil_model, train_cols=train_cols,
        elasticity_map=elasticity_map, waste_frac=waste_frac, sample_n=1000)
    opt_df.to_csv(config.OUTPUTS_DIR / "optimizer_stability_sample.csv", index=False)
    sensitivity.discount_distribution(opt_df).to_csv(
        config.OUTPUTS_DIR / "discount_distribution.csv", index=False)
    sensitivity.category_mean_discount(opt_df).to_csv(
        config.OUTPUTS_DIR / "category_mean_discount.csv", index=False)

    # ─────────────────────────────────────────────────────────────────────
    # Manifest
    # ─────────────────────────────────────────────────────────────────────
    manifest = {
        "run_completed_at": pd.Timestamp.now().isoformat(),
        "winner_demand_model": winner_name,
        "winner_spoilage_model": spoil_winner,
        "elasticity_map": elasticity_map,
        "empirical_waste_fraction": waste_frac,
        "n_holdout_dates": len(holdout_dates),
        "n_holdout_rows": len(df_hold),
        "skip_prophet": skip_prophet,
        "skip_dunn": skip_dunn,
        "n_categories": int(perishable["category"].nunique()),
    }
    (config.OUTPUTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote manifest → {config.OUTPUTS_DIR / 'manifest.json'}")

    print(f"\n  TOTAL TIME: {(time.time() - t_total) / 60:.1f} min")
    print(f"  All outputs written to: {config.OUTPUTS_DIR}")
    print(f"  Spoilage model pickled to: {config.MODELS_DIR / 'spoilage_model.pkl'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-prophet", action="store_true",
                   help="Skip Prophet fits (much faster; LightGBM still wins).")
    p.add_argument("--skip-dunn", action="store_true",
                   help="Skip Dunnhumby load; use default elasticities from config.")
    args = p.parse_args()
    main(skip_prophet=args.skip_prophet, skip_dunn=args.skip_dunn)
