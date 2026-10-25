"""Shared data-loading, cleaning and feature code (same as the project notebook)."""
import numpy as np
import pandas as pd

COLS = ["date", "hour", "purchase_bid", "sell_bid", "mcv", "sched_vol", "mcp", "wmcp"]
NUM = COLS[2:]
NEEDED = ["mcp", "mcv", "purchase_bid", "sell_bid"]
FEATURES = ["mcp_lag24", "mcp_lag48", "mcp_lag168", "mcv_lag24", "purchase_lag24", "sell_lag24",
            "bid_ratio_lag24", "mcp_roll24_mean", "mcp_roll168_mean", "mcp_roll168_std",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
BACKTEST_CUTOFF = pd.Timestamp("2026-04-25 23:00")   # backtest model never sees data after this


def read_excel(src) -> pd.DataFrame:
    """Read an IEX DAM Excel file into 8 named columns.

    Works with BOTH layouts:
      * the cleaned project file (sheet 'Hourly')
      * the raw file downloaded from IEX (title rows, dates as text 'DD-MM-YYYY',
        repeated header rows and Total/Max/Min/Avg rows) - those extra rows are skipped.
    """
    x = pd.ExcelFile(src)
    sheet = "Hourly" if "Hourly" in x.sheet_names else x.sheet_names[0]
    raw = x.parse(sheet, header=None, dtype=object)
    if raw.shape[1] < 8:
        raise ValueError("The file needs 8 columns: Date, Hour, Purchase Bid, Sell Bid, MCV, "
                         "Final Scheduled Volume, MCP, Weighted MCP.")
    raw = raw.iloc[:, :8].copy()
    raw.columns = COLS
    hour = pd.to_numeric(raw["hour"], errors="coerce")
    is_text = raw["date"].map(lambda v: isinstance(v, str))
    date = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    date[is_text] = pd.to_datetime(raw.loc[is_text, "date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    date[~is_text] = pd.to_datetime(raw.loc[~is_text, "date"], errors="coerce")
    keep = date.notna() & hour.between(1, 24) & (hour == hour.round())
    out = raw[keep].copy()
    out["date"] = date[keep]
    out["hour"] = hour[keep].astype(int)
    for c in NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out.empty:
        raise ValueError("No hourly rows found. Check that the file is an IEX Day-Ahead Market "
                         "download (Date, Hour 1-24, bids, MCV, MCP ...).")
    return out.reset_index(drop=True)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Timestamp, chronological sort, de-duplicate, regular hourly grid, fill gaps up to 3 h."""
    df = raw.copy()
    df["timestamp"] = df["date"] + pd.to_timedelta(df["hour"].astype(int) - 1, unit="h")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "timestamp"
    df[NUM] = df[NUM].apply(pd.to_numeric, errors="coerce").interpolate(method="time", limit=3)
    return df


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """17 leakage-free features: every market variable is lagged by >= 24 h."""
    X = pd.DataFrame(index=d.index)
    for L in (24, 48, 168):
        X[f"mcp_lag{L}"] = d.mcp.shift(L)
    X["mcv_lag24"] = d.mcv.shift(24)
    X["purchase_lag24"] = d.purchase_bid.shift(24)
    X["sell_lag24"] = d.sell_bid.shift(24)
    X["bid_ratio_lag24"] = (d.purchase_bid / d.sell_bid).shift(24)
    X["mcp_roll24_mean"] = d.mcp.shift(24).rolling(24).mean()
    X["mcp_roll168_mean"] = d.mcp.shift(24).rolling(168).mean()
    X["mcp_roll168_std"] = d.mcp.shift(24).rolling(168).std()
    hour, dow, month = d.index.hour, d.index.dayofweek, d.index.month
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["is_weekend"] = (dow >= 5).astype(int)
    return X[FEATURES]


def history_ok(df: pd.DataFrame, hours: int = 192) -> bool:
    """The newest 8 days (192 h) must have no gaps in the columns the features need."""
    return len(df) >= hours and not df[NEEDED].tail(hours).isna().any().any()


def forecast_next_day(df: pd.DataFrame, model) -> pd.Series:
    """Predict all 24 hours of the day after the last date in df (no recursion needed)."""
    next_day = df.index.max().normalize() + pd.Timedelta(days=1)
    idx = pd.date_range(next_day, periods=24, freq="h")
    ext = pd.concat([df[NEEDED], pd.DataFrame(index=idx)])
    X = make_features(ext).loc[idx]
    return pd.Series(model.predict(X), index=idx, name="forecast")
