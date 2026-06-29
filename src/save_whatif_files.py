"""
One-shot script to produce the two extra files the dashboard's What-If
section needs:

    outputs/whatif_sample.csv         (500-row sample of holdout features)
    outputs/sku_demand_forecasts.csv  (per-SKU per-day demand forecasts)

Run this AFTER run_pipeline.py / pipeline_notebook.ipynb has produced the
main outputs. It just re-loads the perishable dataset, re-creates the
holdout slice, and saves the two extras. Takes ~5 seconds.

    cd code/
    python save_whatif_files.py
"""
from __future__ import annotations

import json
import pickle

import pandas as pd

import config
import data_loading
import demand_forecast
import features
import optimizer
import spoilage


def main() -> None:
    config.ensure_dirs()

    # Recreate the holdout slice exactly as the pipeline does
    peri = data_loading.load_perishable()
    ts_pivot, price_daily = data_loading.build_daily_aggregates(peri)
    train_dates, val_dates, holdout_dates = data_loading.make_three_way_split(ts_pivot)
    df_opt, df_pre, df_hold = spoilage.prepare_spoilage_dataset(peri, holdout_dates)

    # ── 1. Save the 500-row whatif sample ────────────────────────────────
    whatif_cols = list(dict.fromkeys(
        config.SPOILAGE_FEATURES + [
            "transaction_date", "category", "product_id",
            "base_price", "cost_price", "units_sold", "initial_quantity",
        ]
    ))
    sim_df = df_hold.copy().reset_index(drop=True)
    sample = sim_df[whatif_cols].sample(
        n=min(500, len(sim_df)), random_state=config.RANDOM_STATE,
    ).reset_index(drop=True)
    sample.to_csv(config.OUTPUTS_DIR / "whatif_sample.csv", index=False)
    print(f"Wrote outputs/whatif_sample.csv  ({len(sample)} rows)")

    # ── 2. Reproduce the SKU-level forecasts ─────────────────────────────
    # We need to re-run the demand forecaster to get the holdout forecasts.
    # Load the manifest to know which model won.
    mani_path = config.OUTPUTS_DIR / "manifest.json"
    if mani_path.exists():
        manifest = json.loads(mani_path.read_text())
        winner = manifest.get("winner_demand_model", "LightGBM")
    else:
        winner = "LightGBM"
    print(f"Forecasting holdout with {winner} (this takes ~30 seconds)...")

    feats = features.build_demand_feature_set(
        ts_pivot, price_daily, train_dates, val_dates, holdout_dates,
    )
    cat_forecasts = demand_forecast.forecast_holdout(
        winner, ts_pivot,
        feats["feat_train"], feats["feat_val"], feats["feat_holdout"], holdout_dates,
    )
    sku_forecasts = optimizer.scale_forecasts_to_sku(cat_forecasts, df_hold)

    fc_rows = [{"date": d, "category": c, "forecast": v}
               for (d, c), v in sku_forecasts.items()]
    pd.DataFrame(fc_rows).to_csv(
        config.OUTPUTS_DIR / "sku_demand_forecasts.csv", index=False,
    )
    print(f"Wrote outputs/sku_demand_forecasts.csv  ({len(fc_rows)} rows)")

    # ── 3. Confirm the spoilage model pickle exists ──────────────────────
    pkl = config.MODELS_DIR / "spoilage_model.pkl"
    if pkl.exists():
        print(f"models/spoilage_model.pkl already present ({pkl.stat().st_size / 1024:.0f} KB) — leaving as-is")
    else:
        print("models/spoilage_model.pkl is MISSING — re-run run_pipeline.py to regenerate it")


if __name__ == "__main__":
    main()
