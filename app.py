import hmac, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
from storage import read_extra, combine, extra_csv, github_put, EXTRA_NAME
from features import (read_excel, clean, make_features, forecast_next_day, history_ok,
                      FEATURES, NEEDED, BACKTEST_CUTOFF)

BASE = Path(__file__).parent
DATA_FILE = BASE / "data" / "DAM_Market_Snapshot_Apr2022-Oct2026.xlsx"

try:
    RES = json.load(open(BASE / "data" / "results.json"))
except Exception:
    RES = {}


def _fmt(d):
    return pd.Timestamp(d).strftime("%d %b %Y")

st.set_page_config(page_title="IEX Day-Ahead Price Forecast", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model(name: str) -> CatBoostRegressor:
    m = CatBoostRegressor()
    m.load_model(str(BASE / "models" / name))
    return m


@st.cache_resource(show_spinner="Newer data found – retraining the model (about a minute)…")
def get_full_model(key: str, _df: pd.DataFrame) -> CatBoostRegressor:
    """Use the saved model if it already covers all the data; otherwise retrain automatically."""
    if RES and RES.get("period", [None, None])[1] == key[:10]:
        return load_model("catboost_full.cbm")
    data = make_features(_df).join(_df["mcp"].rename("target")).dropna()
    m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, loss_function="RMSE",
                          random_seed=42, verbose=0)
    return m.fit(data[FEATURES], data["target"])


@st.cache_data
def load_excel() -> pd.DataFrame:
    return read_excel(DATA_FILE)


@st.cache_data
def load_extra() -> pd.DataFrame:
    return read_extra(BASE / "data" / EXTRA_NAME)


def secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


@st.cache_data
def load_upload(content: bytes) -> pd.DataFrame:
    return read_excel(io.BytesIO(content))


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Data")
excel_raw = load_excel()
base_all = combine(excel_raw, load_extra())
st.sidebar.write("To forecast for a later day, upload an Excel file with the newest days "
                 "(same columns as the original file).")
up = st.sidebar.file_uploader("Newer data (.xlsx) – raw IEX downloads or the cleaned file; you can select several files",
                              type=["xlsx"], accept_multiple_files=True)

raw, upload_raw = base_all, None
if up:
    try:
        parts = [load_upload(f.getvalue()) for f in up]
        upload_raw = parts[0]
        for p_ in parts[1:]:
            upload_raw = combine(upload_raw, p_)
        raw = combine(base_all, upload_raw)
        st.sidebar.success(f"Read {len(up)} file(s), {len(upload_raw):,} rows "
                           f"({upload_raw['date'].min():%d %b %Y} – {upload_raw['date'].max():%d %b %Y}).")
    except Exception as e:
        st.sidebar.error(f"Could not read that file: {e}")

df = clean(raw)
last_ts = df.index.max()
last_day = last_ts.normalize()
today = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()

if upload_raw is not None:
    with st.sidebar.expander("Save this data permanently", expanded=True):
        st.caption("Until you save, the new data is only used on your screen. Saving makes it the "
                   "default for everyone.")
        if not history_ok(df):
            st.error("The newest 8 days must be complete (24 hours each, no missing hours). "
                     "Please fix the file first.")
        else:
            sec = secrets()
            online = "GITHUB_TOKEN" in sec and "GITHUB_REPO" in sec
            pw = st.text_input("Admin password", type="password") if online else ""
            if st.button("Save permanently"):
                payload = extra_csv(raw, excel_raw["date"].max())
                if online:
                    if not sec.get("ADMIN_PASSWORD") or not hmac.compare_digest(pw, str(sec["ADMIN_PASSWORD"])):
                        st.error("Wrong password.")
                    else:
                        try:
                            github_put(sec["GITHUB_REPO"], sec["GITHUB_TOKEN"], f"data/{EXTRA_NAME}", payload,
                                       f"Add data up to {last_day.date()}", sec.get("GITHUB_BRANCH", "main"))
                            st.cache_data.clear()
                            st.success("Saved. The app restarts by itself in about a minute and will "
                                       "then show the new data to everyone.")
                        except Exception as e:
                            st.error(f"Could not save to GitHub: {e}")
                else:
                    (BASE / "data" / EXTRA_NAME).write_bytes(payload)
                    st.cache_data.clear()
                    st.success("Saved on this computer. Reload the page to see it.")

st.sidebar.markdown("---")
st.sidebar.write(f"**Latest data:** {last_day.date()}")
st.sidebar.write(f"**Hours loaded:** {len(df):,}")
st.sidebar.caption("B.Tech Project 2026 · NIT Manipur")

# ---------------------------------------------------------------- header
st.title("⚡ IEX Day-Ahead Electricity Price Forecast")
st.caption("Market Clearing Price (MCP) in Rs/MWh · forecast made with CatBoost from information that is "
           "available the day before.")

tab1, tab2, tab3 = st.tabs(["Next-day forecast", "Check a past date", "About"])

# ---------------------------------------------------------------- tab 1
with tab1:
    target_day = last_day + pd.Timedelta(days=1)
    st.subheader(f"Forecast for {target_day:%A, %d %B %Y}")

    age = (today - last_day).days
    if age > 1:
        st.warning(f"The newest data is from {last_day.date()}, which is {age} days before today. "
                   "This forecast is for the day after that date. Upload newer data in the sidebar to "
                   "forecast for today or tomorrow.")
    if not history_ok(df):
        st.error("The model needs the latest 8 days in a row with no gaps. "
                 "Please add the missing days and try again.")
    else:
        pred = forecast_next_day(df, get_full_model(str(last_ts), df))
        out = pd.DataFrame({"Hour": [f"{h:02d}:00–{(h + 1) % 24:02d}:00" for h in range(24)],
                            "Forecast MCP (Rs/MWh)": pred.round(0).astype(int).values})
        hi, lo = pred.idxmax(), pred.idxmin()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average price", f"Rs {pred.mean():,.0f}")
        c2.metric("Highest hour", f"{hi:%H:00}", f"Rs {pred.max():,.0f}", delta_color="off")
        c3.metric("Lowest hour", f"{lo:%H:00}", f"Rs {pred.min():,.0f}", delta_color="off")
        c4.metric("Peak vs lowest", f"{pred.max() / pred.min():.1f}×")
        st.line_chart(pd.DataFrame({"Forecast MCP (Rs/MWh)": pred.values},
                                   index=[f"{h:02d}:00" for h in range(24)]), height=320)
        st.dataframe(out, hide_index=True, width="stretch", height=300)
        st.download_button("Download forecast (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"forecast_{target_day.date()}.csv", mime="text/csv")
        st.caption("A forecast is an estimate, not a guarantee. On past test data the average error was "
                   f"about Rs {RES.get('mae_catboost', 460):,.0f} per MWh, and evening peak hours are the hardest to predict.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Compare forecast and actual price")
    st.write("Pick a date from the test period. These days were **never used for training** the model "
             "used here, so the comparison is fair.")
    first_ok = (BACKTEST_CUTOFF + pd.Timedelta(hours=1)).normalize()
    d = st.date_input("Date", value=last_day.date(), min_value=first_ok.date(), max_value=last_day.date())
    day = pd.Timestamp(d)
    idx = pd.date_range(day, periods=24, freq="h")
    X = make_features(df).reindex(idx)
    actual = df["mcp"].reindex(idx)
    if X.isna().any().any() or actual.isna().any():
        st.error("This date does not have complete data.")
    else:
        p = load_model("catboost_backtest.cbm").predict(X[FEATURES])
        cmp_ = pd.DataFrame({"Actual": actual.values, "Forecast": p}, index=[f"{h:02d}:00" for h in range(24)])
        mae = float(np.abs(cmp_.Actual - cmp_.Forecast).mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Actual average", f"Rs {cmp_.Actual.mean():,.0f}")
        c2.metric("Forecast average", f"Rs {cmp_.Forecast.mean():,.0f}")
        c3.metric("Average error (MAE)", f"Rs {mae:,.0f}")
        st.line_chart(cmp_, height=320)
        tbl = cmp_.round(0).astype(int)
        tbl["Error"] = tbl.Forecast - tbl.Actual
        st.dataframe(tbl, width="stretch", height=300)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("How it works")
    st.write(f"""
1. **Data:** hourly IEX Day-Ahead Market prices and volumes ({_fmt(RES["period"][0])} – {_fmt(RES["period"][1])}, {RES["n_hours"]:,} hours).
2. **Features (17):** the price one day, two days and one week earlier, recent average price and
   volatility, yesterday's bid and volume information, and sine/cosine versions of hour, weekday and month.
   Everything is at least 24 hours old, so the forecast could really have been made the day before.
3. **Model:** CatBoost (gradient-boosted trees), compared with Linear Regression and Random Forest.
4. **Testing:** the last 180 days ({_fmt(RES["test_range"][0])} – {_fmt(RES["test_range"][1])}) were kept aside and never used for training.
""")
    try:
        res = json.load(open(BASE / "data" / "results.json"))
        rows = []
        for t, p in zip(res["test"], res["peak"]):
            rows.append({"Model": t["Model"], "MAE": round(t["MAE"]), "RMSE": round(t["RMSE"]),
                         "MAPE %": round(t["MAPE"], 1), "Peak-hour MAE": round(p["MAE"])})
        st.markdown("**Accuracy on the 180-day test period** (Rs/MWh; lower is better)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    except Exception:
        pass
    st.markdown("**Project team:** Bangdi Newme, Salam Hero Singh, Huidrom Loyumba Mangang · "
                "Guide: Dr. Ashwini D. Manchalwar · Dept. of Electrical Engineering, NIT Manipur")
    st.caption("This tool is for education and information. It is not trading or financial advice.")
