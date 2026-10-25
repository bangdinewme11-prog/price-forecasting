"""Helpers for adding new data permanently (merge, CSV, GitHub commit)."""
import base64, json, urllib.error, urllib.request
from pathlib import Path
import pandas as pd
from features import COLS

EXTRA_NAME = "extra_hourly.csv"      # new rows added after the original Excel file


def read_extra(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLS)
    x = pd.read_csv(path)
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    return x[COLS].dropna(subset=["date", "hour"])


def combine(base: pd.DataFrame, upload: pd.DataFrame) -> pd.DataFrame:
    """Add the uploaded rows to the existing ones; the newer upload wins for any repeated hour."""
    both = pd.concat([base, upload], ignore_index=True)
    return (both.drop_duplicates(["date", "hour"], keep="last")
                .sort_values(["date", "hour"]).reset_index(drop=True))


def extra_csv(merged: pd.DataFrame, excel_last_date: pd.Timestamp) -> bytes:
    """Only the rows newer than the original Excel file are stored in the small CSV."""
    e = merged[merged["date"] > excel_last_date][COLS].copy()
    e["date"] = e["date"].dt.strftime("%Y-%m-%d")
    return e.to_csv(index=False).encode()


def github_put(repo: str, token: str, path: str, content: bytes, message: str, branch: str = "main"):
    """Create or update one file in a GitHub repository (this triggers Streamlit to redeploy)."""
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    hdr = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
           "User-Agent": "dam-forecast-app"}
    sha = None
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{url}?ref={branch}", headers=hdr)) as r:
            sha = json.load(r)["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    body = {"message": message, "content": base64.b64encode(content).decode(), "branch": branch}
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr, method="PUT")
    with urllib.request.urlopen(req) as r:
        return json.load(r)
