"""Run once (and again whenever you add newer data) to rebuild the model files and the accuracy table.
    python train_models.py
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import read_excel, clean, make_features, FEATURES, BACKTEST_CUTOFF

DATA_FILE = "data/DAM_Market_Snapshot_Apr2022-Oct2026.xlsx"   # change to your newest file when retraining
df = clean(read_excel(DATA_FILE))
data = make_features(df).join(df["mcp"].rename("target")).dropna()


def cat():
    return CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function="RMSE",
                             random_seed=42, verbose=0)


# 1) model used for the real next-day forecast: trained on ALL data
cat().fit(data[FEATURES], data["target"]).save_model("models/catboost_full.cbm")
print("Saved full model    | rows:", len(data), "|", data.index.min(), "→", data.index.max())

# 2) backtest model: never sees the last 180 days (used for the fair 'Check a past date' tab)
train, test = data[data.index <= BACKTEST_CUTOFF], data[data.index > BACKTEST_CUTOFF]
bt = cat().fit(train[FEATURES], train["target"])
bt.save_model("models/catboost_backtest.cbm")
print("Saved backtest model| train rows:", len(train), "| test rows:", len(test))

# 3) accuracy table on the 180-day hold-out
preds = {"Naive (t-24)": test["mcp_lag24"].values}
preds["Linear Regression"] = make_pipeline(StandardScaler(), LinearRegression()).fit(
    train[FEATURES], train["target"]).predict(test[FEATURES])
preds["Random Forest"] = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=-1, random_state=42).fit(
    train[FEATURES], train["target"]).predict(test[FEATURES])
preds["CatBoost"] = bt.predict(test[FEATURES])

peak_hours = train.groupby(train.index.hour).target.mean().nlargest(6).index.tolist()
is_peak = test.index.hour.isin(peak_hours)
y = test["target"].values


def scores(a, p):
    return {"MAE": float(mean_absolute_error(a, p)), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "MAPE": float(np.mean(np.abs((a - p) / a)) * 100)}


res = {"period": [str(df.index.min().date()), str(df.index.max().date())], "n_hours": int(len(df)),
       "test_range": [str(test.index.min().date()), str(test.index.max().date())], "test_hours": int(len(test)),
       "peak_hours": sorted(int(h) + 1 for h in peak_hours),
       "test": [{"Model": n, **scores(y, p)} for n, p in preds.items()],
       "peak": [{"Model": n, **scores(y[is_peak], p[is_peak])} for n, p in preds.items()]}
res["mae_catboost"] = next(r["MAE"] for r in res["test"] if r["Model"] == "CatBoost")
json.dump(res, open("data/results.json", "w"), indent=2)
for r in res["test"]:
    print(f'{r["Model"]:18s} MAE {r["MAE"]:8.1f}  RMSE {r["RMSE"]:8.1f}  MAPE {r["MAPE"]:5.1f}%')
