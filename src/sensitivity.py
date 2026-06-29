"""
Stage 5 — Sensitivity analyses.

- High-spoilage subset (days_until_expiry <= 3): re-run all three strategies
- Elasticity sensitivity (× 0.5 / × 1.0 / × 1.5): does the optimizer's
  recommendation move when elasticity assumptions change? IMPORTANT: this
  is dominated by the near-expiry override (-2.0). The non-near-expiry
  variant is what tests whether the regular elasticity actually drives
  decisions.
- Optimizer stability check: discount distribution on a 1k-row pre-holdout
  sample.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

import config
from simulation import run_full_simulation, kpi_summary
from optimizer import batch_simulate, optimize_row


def high_spoilage_kpi(sim_df: pd.DataFrame, *, threshold: int = 3,
                      **batch_kwargs) -> Tuple[pd.DataFrame, int]:
    """Re-run all 3 strategies on the (days_until_expiry <= threshold) subset."""
    subset = sim_df[sim_df["days_until_expiry"] <= threshold].copy()
    if len(subset) < 10:
        # Widen the net to keep the analysis meaningful
        subset = sim_df[sim_df["days_until_expiry"] <= 7].copy()
    results = run_full_simulation(subset, **batch_kwargs)
    return kpi_summary(results), len(subset)


def elasticity_sensitivity(sim_df: pd.DataFrame, *,
                            elasticity_map: dict, sample_n: int = 200,
                            include_non_near_expiry_only: bool = True,
                            **batch_kwargs) -> pd.DataFrame:
    """
    Run the optimizer with elasticity scaled to {0.5×, 1.0×, 1.5×}.

    By default we ALSO produce a non-near-expiry-only variant — because
    when 42% of rows hit the -2.0 override, the regular-elasticity
    multiplier has nothing to do, and the test looks like a non-result.
    Restricting to non-near-expiry rows actually tests whether the
    elasticity coefficient is pulling weight in the optimization.
    """
    rows = []

    def _run(label_suffix, df_subset):
        for mult, scenario in [(0.5, "Elast x 0.5"), (1.0, "Baseline"),
                                (1.5, "Elast x 1.5")]:
            scaled_map = {k: v * mult for k, v in elasticity_map.items()}
            sample = df_subset.sample(min(sample_n, len(df_subset)),
                                       random_state=99).copy()
            res = batch_simulate(sample, "dynamic",
                                  elasticity_map=scaled_map, **batch_kwargs)
            for d, count in res["discount"].value_counts().items():
                rows.append({
                    "subset":   label_suffix,
                    "scenario": scenario,
                    "discount": float(d),
                    "n_rows":   int(count),
                    "mean_profit": float(res.loc[res["discount"] == d, "profit"].mean()),
                })
            rows.append({
                "subset":      label_suffix,
                "scenario":    scenario,
                "discount":    -1.0,  # marker row for OVERALL mean profit
                "n_rows":      int(len(res)),
                "mean_profit": float(res["profit"].mean()),
            })

    _run("ALL_ROWS", sim_df)
    if include_non_near_expiry_only:
        non_near = sim_df[sim_df["days_until_expiry"] > config.NEAR_EXPIRY_DAYS]
        if len(non_near) >= 30:
            _run("NON_NEAR_EXPIRY_ONLY", non_near)

    return pd.DataFrame(rows)


def optimizer_stability_sample(df_pre: pd.DataFrame, *,
                                spoilage_model, train_cols: list,
                                elasticity_map: dict, waste_frac: float,
                                sample_n: int = 1000) -> pd.DataFrame:
    """Run optimize_row on a 1k pre-holdout sample → discount distribution."""
    sample = df_pre.sample(min(sample_n, len(df_pre)),
                            random_state=config.RANDOM_STATE).copy()
    rows = sample.apply(
        lambda r: optimize_row(r, spoilage_model=spoilage_model,
                               train_cols=train_cols,
                               elasticity_map=elasticity_map,
                               waste_frac=waste_frac),
        axis=1,
    )
    opt_df = pd.DataFrame(list(rows))
    opt_df["category"] = sample["category"].values
    return opt_df


def discount_distribution(opt_df: pd.DataFrame) -> pd.DataFrame:
    dist = opt_df["discount"].value_counts(normalize=True).sort_index() * 100
    return pd.DataFrame({
        "discount": dist.index,
        "pct_of_skus": dist.values.round(2),
    })


def category_mean_discount(opt_df: pd.DataFrame) -> pd.DataFrame:
    return (opt_df.groupby("category")["discount"].mean() * 100).round(2).reset_index()
