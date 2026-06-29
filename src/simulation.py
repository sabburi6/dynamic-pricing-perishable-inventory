"""
Stage 4 (cont.) — KPI roll-ups and category breakdowns from the simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer import batch_simulate

STRATEGIES = ["no_discount", "fixed_20", "dynamic"]
LABELS = {"no_discount": "No Discount", "fixed_20": "Fixed 20%", "dynamic": "Dynamic"}


def run_full_simulation(sim_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run all three strategies and concat the per-row results."""
    parts = [batch_simulate(sim_df, s, **kwargs) for s in STRATEGIES]
    return pd.concat(parts, ignore_index=True)


def kpi_summary(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue / profit / waste / sell-through per strategy."""
    kpi = (
        sim_results.groupby("strategy")
        .agg(
            Total_Revenue=("revenue", "sum"),
            Total_Profit=("profit", "sum"),
            Total_Waste=("exp_waste", "sum"),
            Avg_SellThru=("sell_through", "mean"),
            Avg_Discount=("discount", "mean"),
        )
        .reindex(STRATEGIES).round(2)
    )
    init_sum = sim_results.groupby("strategy")["init_qty"].sum().reindex(STRATEGIES)
    kpi["Waste_Pct"] = (kpi["Total_Waste"] / init_sum * 100).round(2)
    kpi["Strategy_Label"] = [LABELS[s] for s in kpi.index]
    return kpi


def delta_table(kpi: pd.DataFrame, demand_model: str, spoilage_model: str) -> pd.DataFrame:
    """Side-by-side delta vs no-discount baseline."""
    base = kpi.loc["no_discount"]
    return pd.DataFrame({
        "Strategy":               [LABELS[s] for s in STRATEGIES],
        "Revenue_USD":            [round(kpi.loc[s, "Total_Revenue"], 2) for s in STRATEGIES],
        "Profit_USD":             [round(kpi.loc[s, "Total_Profit"], 2) for s in STRATEGIES],
        "Units_Wasted":           [round(kpi.loc[s, "Total_Waste"], 1) for s in STRATEGIES],
        "Sell_Through_Pct":       [round(kpi.loc[s, "Avg_SellThru"] * 100, 1) for s in STRATEGIES],
        "Waste_Pct":              [kpi.loc[s, "Waste_Pct"] for s in STRATEGIES],
        "Delta_Profit_vs_Base":   [round(kpi.loc[s, "Total_Profit"] - base["Total_Profit"], 2) for s in STRATEGIES],
        "Forecasted_Demand_Used": ["No", "No", "Yes"],
        "Demand_Model":           ["—", "—", demand_model],
        "Spoilage_Model":         ["—", "—", spoilage_model],
    })


def category_profit_breakdown(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Long-form table: strategy × category → profit."""
    return (
        sim_results.groupby(["strategy", "category"])["profit"]
        .sum()
        .unstack("strategy")
        .reindex(columns=STRATEGIES)
        .rename(columns=LABELS)
        .round(2)
        .reset_index()
    )


def near_expiry_buckets(sim_results: pd.DataFrame) -> pd.DataFrame:
    """For the dynamic strategy, summarise discount + profit by days-to-expiry."""
    bucket_order = ["1-2 days", "3-5 days", "6-10 days", "10+ days"]

    def bucket(d):
        if d <= 2:
            return "1-2 days"
        if d <= 5:
            return "3-5 days"
        if d <= 10:
            return "6-10 days"
        return "10+ days"

    dyn = sim_results[sim_results["strategy"] == "dynamic"].copy()
    dyn["expiry_bucket"] = dyn["days_until_expiry"].apply(bucket)
    dyn["profit_per_unit"] = dyn["profit"] / dyn["init_qty"].clip(lower=1)
    dyn["discount_pct"] = dyn["discount"] * 100
    tbl = (
        dyn.groupby("expiry_bucket")
        .agg(
            n_SKUs=("profit", "count"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_profit_per_unit=("profit_per_unit", "mean"),
            pct_discounted=("discount", lambda x: (x > 0).mean() * 100),
        )
        .reindex(bucket_order)
        .round(2)
        .reset_index()
    )
    return tbl


def daily_profit_trend(sim_results: pd.DataFrame) -> pd.DataFrame:
    """Daily profit per strategy — for the time-series chart."""
    return (
        sim_results.groupby(["date", "strategy"])["profit"]
        .sum().unstack("strategy")
        .reindex(columns=STRATEGIES)
        .rename(columns=LABELS)
        .reset_index()
    )
