"""
Stage 3 — Spoilage classifier (4-way comparison).

Trains LogisticRegression / RandomForest / XGBoost / LightGBM on the binary
target (units_wasted > 0) using only pre-holdout rows. Selects winner by
ROC-AUC, with Brier Score as tiebreaker (within 0.005 AUC).

Also computes the empirical waste fraction used to calibrate the
spoilage probability into expected wasted units.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBClassifier
import lightgbm as lgb_cls

import config


# Columns the spoilage stage needs from the perishable dataset (features +
# context cols used downstream by the optimizer/simulation)
OPT_COLS = list(dict.fromkeys(
    config.SPOILAGE_FEATURES + [
        "was_spoiled", "category", "product_id", "store_id",
        "cost_price", "base_price", "units_sold", "units_wasted",
        "waste_cost", "transaction_date", "initial_quantity",
    ]
))


def prepare_spoilage_dataset(perishable_clean: pd.DataFrame, holdout_dates):
    """Build df_opt (full) and split into pre-holdout vs holdout."""
    df_opt = perishable_clean[OPT_COLS].dropna().copy()
    df_opt["spoilage_flag"] = (df_opt["units_wasted"] > 0).astype(int)
    df_pre = df_opt[df_opt["transaction_date"] < holdout_dates[0]].copy()
    df_hold = df_opt[df_opt["transaction_date"].isin(holdout_dates)].copy()
    return df_opt, df_pre, df_hold


def _encode(df_subset: pd.DataFrame) -> pd.DataFrame:
    return (pd.get_dummies(df_subset[config.SPOILAGE_FEATURES],
                           columns=["quality_grade"], drop_first=True)
              .astype(float))


def train_and_select(df_pre: pd.DataFrame) -> Tuple[object, list, pd.DataFrame, str]:
    """
    Fit four classifiers on an 80/20 stratified split of df_pre,
    return (winner_model, train_columns, comparison_df, winner_name).
    """
    X_full = _encode(df_pre)
    y_full = df_pre["spoilage_flag"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y_full, test_size=0.2,
        random_state=config.RANDOM_STATE, stratify=y_full,
    )
    train_cols = X_tr.columns.tolist()
    pos_weight = float((y_tr == 0).sum()) / float((y_tr == 1).sum())

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced",
            random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            min_samples_leaf=10, n_jobs=-1,
            random_state=config.RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=config.RANDOM_STATE, verbosity=0),
        "LightGBM": lgb_cls.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            class_weight="balanced", n_jobs=-1,
            random_state=config.RANDOM_STATE, verbose=-1),
    }

    rows = []
    trained = {}
    for name, clf in candidates.items():
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model":         name,
            "ROC_AUC":       round(float(roc_auc_score(y_te, proba)), 4),
            "Brier_Score":   round(float(brier_score_loss(y_te, proba)), 4),
            "F1_Spoiled":    round(float(f1_score(y_te, pred, pos_label=1, zero_division=0)), 4),
        })
        trained[name] = clf

    cmp_df = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

    # Winner: best AUC; if top two within 0.005, tiebreak on Brier
    if len(cmp_df) > 1 and (cmp_df.loc[0, "ROC_AUC"] - cmp_df.loc[1, "ROC_AUC"]) <= 0.005:
        top2 = cmp_df.head(2)
        winner_name = top2.loc[top2["Brier_Score"].idxmin(), "Model"]
    else:
        winner_name = cmp_df.loc[0, "Model"]

    return trained[winner_name], train_cols, cmp_df, winner_name


def score_holdout(model, train_cols: list, df_hold: pd.DataFrame,
                  df_opt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add spoilage_risk column to both df_hold and df_opt."""
    X_hold = (_encode(df_hold).reindex(columns=train_cols, fill_value=0))
    df_hold = df_hold.copy()
    df_hold["spoilage_risk"] = model.predict_proba(X_hold)[:, 1]

    X_opt = (_encode(df_opt).reindex(columns=train_cols, fill_value=0))
    df_opt = df_opt.copy()
    df_opt["spoilage_risk"] = model.predict_proba(X_opt)[:, 1]
    return df_hold, df_opt


def empirical_waste_fraction(df_pre: pd.DataFrame) -> float:
    """E[units_wasted / initial_quantity] over pre-holdout rows."""
    return float((df_pre["units_wasted"] / df_pre["initial_quantity"].clip(lower=1)).mean())
