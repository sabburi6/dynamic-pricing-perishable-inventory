# Dynamic Pricing for Perishable Grocery Inventory

An end-to-end machine learning pipeline that optimizes markdown pricing for perishable grocery products. The system forecasts daily demand, estimates price elasticity, classifies spoilage risk, and recommends per-SKU discount levels to maximize profit while minimizing food waste.

## Problem

Grocery retailers lose 10–15% of perishable stock annually to spoilage caused by mismatched pricing and demand timing. Static or uniform discount strategies either erode margins on items that would sell at full price, or act too late to prevent waste. This project addresses the problem by jointly optimizing demand signals and spoilage risk to make selective, data-driven markdown decisions.

## Pipeline Overview

The system is built as a five-stage pipeline:

**Stage 1 — Demand Forecasting:** Predicts daily category-level unit sales using LightGBM with lag features, rolling statistics, calendar encodings, and price aggregates. Compared against Holt-Winters as a classical baseline.

**Stage 2 — Price Elasticity Estimation:** Estimates how sensitive demand is to price changes using log-log panel regression with store and week fixed effects on the Dunnhumby retail panel dataset (102 weeks, 2,500 households).

**Stage 3 — Spoilage Classification:** Binary classification of whether a SKU-day will result in spoilage, using features like days until expiry, storage temperature, quality grade, and demand variability. Four classifiers compared (LightGBM, XGBoost, Random Forest, Logistic Regression).

**Stage 4 — Markdown Optimization:** Grid search over five discount levels (0%, 10%, 20%, 30%, 40%) to select the discount that maximizes expected profit per SKU-day, subject to a 5% minimum margin floor. Uses outputs from Stages 1–3.

**Stage 5 — Sensitivity Analysis:** Tests robustness through high-spoilage subsets, elasticity scaling experiments, and stability sampling.

## Datasets

- **Perishable Goods Management Dataset** (Kaggle) — 100,000 transaction records across 8 product categories (Bakery, Beverages, Dairy, Deli, Meat, Produce, Ready-to-Eat, Seafood), 50 stores, 5 regions
- **Dunnhumby Complete Journey** — Retail panel data covering 2,500 households across 102 weeks, used for elasticity estimation

Both datasets are publicly available. 
The two datasets have no shared product identifiers, so they cannot be merged directly. Instead, price elasticity values estimated from the Dunnhumby panel are applied to matching product categories in the perishable dataset (e.g., Dunnhumby's PRODUCE department maps to the Produce category). This allows the optimizer to use real-world price sensitivity data without requiring a row-level join.

## Key Results

**Demand Forecasting**
| Model | RMSE | MAE | Directional Accuracy |
|-------|------|-----|---------------------|
| LightGBM | 858.4 | 662.6 | 62.3% |
| Holt-Winters | 923.2 | 741.7 | 47.3% |

**Spoilage Classification**
| Model | ROC-AUC | Brier Score | F1 |
|-------|---------|-------------|-----|
| LightGBM | 0.853 | 0.154 | 0.783 |
| XGBoost | 0.852 | 0.154 | 0.785 |
| Random Forest | 0.828 | 0.171 | 0.767 |
| Logistic Regression | 0.722 | 0.216 | 0.707 |

**60-Day Simulation — Strategy Comparison**
| Strategy | Revenue | Profit | Waste Rate | Sell-Through |
|----------|---------|--------|------------|-------------|
| No Discount | $10.27M | $419K | 33.5% | 66.0% |
| Fixed 20% | $10.22M | $349K | 22.9% | 75.0% |
| Dynamic | $11.85M | $1.99M | 16.8% | 79.0% |

Dynamic pricing delivered a **376% profit gain** over the no-discount baseline while cutting waste from 33.5% to 16.8%.

**Category Highlights:**
- Bakery: reversed a $95K loss into a $95K gain
- Seafood: reversed a $279K loss into a $362K gain
- Near-expiry items (1–2 days): average 23% discount, moved most inventory before spoilage

## Tech Stack

- **Language:** Python 3.11
- **ML Models:** LightGBM, XGBoost, Scikit-learn, Statsmodels
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Evaluation:** SHAP, ROC-AUC, RMSE, Brier Score
