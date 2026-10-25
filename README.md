# IEX Day-Ahead Price Forecast (Streamlit app)

B.Tech Project 2026, NIT Manipur.

## Folder
```
app.py               the website
features.py          data cleaning + feature code (same as the notebook)
storage.py           merges new data and saves it (GitHub / local)
train_models.py      creates the two model files
requirements.txt     packages the app needs
models/              catboost_full.cbm (forecast) + catboost_backtest.cbm (fair past-date check)
data/                the Excel file, results.json (accuracy table), extra_hourly.csv (created when you save new data)
```

## 1. Run it on your own computer
Open **Anaconda Prompt**:
```
cd path\to\dam_app
pip install -r requirements.txt
streamlit run app.py
```
A browser tab opens at http://localhost:8501.

## 2. Put it online for others (free)
1. Create a free account on github.com and a new **public** repository.
2. Upload everything in this folder (keep the `models` and `data` folders).
3. Go to share.streamlit.io, sign in with GitHub, click **Create app**, choose your repository, set the main file to `app.py`, and click **Deploy**.
4. After a few minutes you get a link like `https://yourname-dam-app.streamlit.app`. Share it.

## 3. Keep it up to date
The forecast is for the day after the newest data. Each day:
- **Quick way:** open the app and upload an Excel file with the recent days (same 8 columns as the original, at least the last 8 days in a row, no gaps). Visitors have to upload it themselves too.
- **Permanent way:** add the new rows to the Excel file in `data/` (current file: `DAM_Market_Snapshot_Apr2022-Oct2026.xlsx`), upload it to GitHub with the same name, and the app uses it for everyone. If you rename the file, change `DATA_FILE` in `app.py` and in `train_models.py`.

Retrain the models every few weeks/months with the newest data: set `DATA_FILE` in `train_models.py` to your newest file and run `python train_models.py`. This rebuilds both model files and `data/results.json` (the accuracy table); upload those to GitHub. If the test period dates change, also update `BACKTEST_CUTOFF` in `features.py` (last data date minus 180 days).


**Reading IEX files:** the upload box accepts the raw files exactly as downloaded from IEX (several at once, e.g. one per month) as well as the cleaned project file. Extra rows such as headers and Total/Max/Min/Avg are skipped automatically.

## 4. Automatic data updates (upload once, everyone sees it)
1. Open the app and upload an Excel file with the newest days in the left sidebar (same columns as the original, no missing hours in the newest 8 days).
2. Open **Save this data permanently**, type the admin password and click **Save permanently**.
3. The app saves the new rows to `data/extra_hourly.csv` in your GitHub repository, restarts by itself in about a minute, and the forecast moves to the day after the newest date.
4. The model is retrained automatically (about 10 seconds to a minute) whenever the data is newer than the saved model.

**One-time setup on Streamlit Cloud**
1. GitHub → profile picture → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**. Choose only your repository and give it **Contents: Read and write**. Copy the token.
2. Streamlit Cloud → your app → **⋮ → Settings → Secrets**, then paste (with your own values):
```
GITHUB_TOKEN = "github_pat_xxxxxxxx"
GITHUB_REPO = "your-username/dam-price-forecast"
ADMIN_PASSWORD = "choose-a-strong-password"
```
3. Click **Save**. Never put the token inside the code or the repository.

When running on your own computer there are no secrets, so the same button saves the data on that computer instead.

Every few months, run `python train_models.py` with the newest Excel file to refresh the saved models and the accuracy table.

## Notes
- The "Check a past date" tab only works for 26 Apr 2026 onwards, because those are the days the backtest model never saw.
- Forecasts are estimates. See the accuracy table in the app's About tab (average CatBoost error on the test period is about Rs 700/MWh).
