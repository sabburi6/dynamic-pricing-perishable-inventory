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

