import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from fetch_db_data import fetch_data_from_db
from typing import Callable, List, Optional, Dict


# --- Load Data ---
df = fetch_data_from_db()

# --- Split into Train and Test (then combine for below) ---
# Time-series holdout (no shuffle)
# Fraction of data to hold out as the global/test set
global_pct = 0.2  # adjust as needed (e.g., 0.1 - 0.3)

# Defensive copy and ensure a datetime-sorted dataframe
df = df.copy()
if "date" not in df.columns:
    raise KeyError("Expected a 'date' column in df for time-series splitting.")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Compute split index (last global_pct rows become test)
n = len(df)
cut = int((1.0 - float(global_pct)) * n)

if cut <= 0 or cut >= n:
    raise ValueError(f"global_pct={global_pct} produces an invalid split for n={n} rows.")

train_df = df.iloc[:cut].copy()
test_df = df.iloc[cut:].copy()

# Mark source and combine back (so downstream feature-engineering can create lags, etc.)
train_df["source"] = "train"
test_df["source"] = "test"
combined = pd.concat([train_df, test_df], ignore_index=True).sort_values("date").reset_index(drop=True)
# ----

# --- Date Features ---
combined["day_of_week"] = combined["date"].dt.weekday
combined["month"] = combined["date"].dt.month

# --- Feature Engineering ---
sales_categories = ["total_sales", "food", "drink_coffee", "drink_non_coffee", "retail"]

# Add enhanced features (create lags, EMA, centered rolling)
for cat in sales_categories:
    for lag in [1, 3, 7]:
        combined[f"{cat}_lag_{lag}"] = combined[cat].shift(lag)
        combined[f"{cat}_ema_7"] = combined[cat].ewm(span=7, adjust=False).mean()
        combined[f"{cat}_rolling_center_7"] = combined[cat].rolling(window=7, center=True).mean()

# --- Split back into train/test ---
df_train = combined[combined["source"] == "train"].copy()
df_test = combined[combined["source"] == "test"].copy()

# --- Drop zero-sale days ---
df_train = df_train[df_train["total_sales"] > 0]
df_test = df_test[df_test["total_sales"] > 0]

# --- Handle categorical features using pandas.get_dummies ---
# Ensure campaign_type exists
for df_ in (df_train, df_test):
    if "campaign_type" not in df_.columns:
        df_["campaign_type"] = pd.NA

# Create one-hot columns
train_dummies = pd.get_dummies(df_train["campaign_type"], prefix="campaign_type")
test_dummies = pd.get_dummies(df_test["campaign_type"], prefix="campaign_type")

# Drop original categorical col and concat encoded columns
df_train = pd.concat([df_train.drop(columns=["campaign_type"]), train_dummies], axis=1)
df_test = pd.concat([df_test.drop(columns=["campaign_type"]), test_dummies], axis=1)

# Align columns between train and test so both frames have the same set of columns (missing filled with 0)
df_train, df_test = df_train.align(df_test, join="outer", axis=1, fill_value=0)

# Recompute original_base from df_train (after all engineering + encoding)
exclude_cols = set(sales_categories) | {"ignore", "date", "source"}
original_base = [c for c in df_train.columns if c not in exclude_cols]
original_base.sort()

# --- Modeling Loop (with feature validation) ---
results = []

for target in sales_categories:
    target_feats = [f"{target}_lag_{l}" for l in [1, 3, 7]] + [f"{target}_ema_7", f"{target}_rolling_center_7"]
    full_features = original_base + target_feats

    # Ensure features exist in df_train / df_test
    available_features = [c for c in full_features if c in df_train.columns]
    missing = set(full_features) - set(available_features)
    if missing:
        print(f"Warning: missing features for target '{target}': {sorted(missing)}")

    if not available_features:
        print(f"Skipping target '{target}' — no features available after validation.")
        continue

    # Drop rows with NA in the required features or target
    train_clean = df_train.dropna(subset=available_features + [target])
    test_clean = df_test.dropna(subset=available_features + [target])

    if train_clean.shape[0] < 10:
        print(f"Skipping '{target}' — too few training rows ({train_clean.shape[0]}).")
        continue
    if test_clean.shape[0] == 0:
        print(f"Skipping '{target}' — no test rows after dropna.")
        continue

    X_train = train_clean[available_features]
    y_train = train_clean[target]
    X_test = test_clean[available_features]
    y_test = test_clean[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    results.append((target, r2, rmse))
    print(f"{target:20} → R² = {r2:.4f}, RMSE = {rmse:.2f}")


# --- Backtest / Walk-forward module ---

def walk_forward_backtest(full_df: pd.DataFrame,
                          target: str,
                          features: List[str],
                          model_ctor: Optional[Callable] = None,
                          initial_train_size: Optional[int] = None,
                          test_window: int = 7,
                          expand_train: bool = True,
                          min_train_size: int = 30) -> pd.DataFrame:
    """Perform a simple walk-forward (rolling-origin) backtest."""
    if model_ctor is None:
        model_ctor = lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    df = full_df.sort_values("date").reset_index(drop=True).copy()
    n = len(df)
    if initial_train_size is None:
        initial_train_size = max(int(0.5 * n), min_train_size)

    train_start = 0
    train_end = initial_train_size

    rows = []
    while train_end < n:
        test_start = train_end
        test_end = min(train_end + test_window, n)

        train_slice = df.iloc[train_start:train_end].dropna(subset=features + [target])
        test_slice = df.iloc[test_start:test_end].dropna(subset=features + [target])

        if train_slice.shape[0] < min_train_size or test_slice.shape[0] == 0:
            train_end = test_end if expand_train else train_end + test_window
            if not expand_train:
                train_start = train_end - initial_train_size
            continue

        X_train = train_slice[features]
        y_train = train_slice[target]
        X_test = test_slice[features]
        y_test = test_slice[target]

        m = model_ctor()
        m.fit(X_train, y_train)
        preds = m.predict(X_test)

        for di, y_t, p in zip(test_slice["date"].tolist(), y_test.tolist(), preds.tolist()):
            rows.append({"date": di, "y_true": y_t, "y_pred": p, "target": target})

        train_end = test_end if expand_train else train_end + test_window
        if not expand_train:
            train_start = max(0, train_end - initial_train_size)

    return pd.DataFrame(rows)


def backtest_all_targets(full_df: pd.DataFrame,
                          targets: List[str],
                          base_features: List[str],
                          test_window: int = 7) -> pd.DataFrame:
    """Run walk-forward backtest for each target and return aggregated results + metrics."""
    all_rows = []
    metrics = []
    for target in targets:
        target_feats = [f"{target}_lag_{l}" for l in [1, 3, 7]] + [f"{target}_ema_7", f"{target}_rolling_center_7"]
        features = [c for c in base_features + target_feats if c in full_df.columns]
        if not features:
            print(f"Skipping backtest for {target} — no features available.")
            continue
        df_preds = walk_forward_backtest(full_df, target, features, test_window=test_window)
        if df_preds.empty:
            print(f"No backtest predictions for {target} (empty).")
            continue
        r2 = r2_score(df_preds["y_true"], df_preds["y_pred"]) if not df_preds.empty else float("nan")
        rmse = mean_squared_error(df_preds["y_true"], df_preds["y_pred"], squared=False) if not df_preds.empty else float("nan")
        metrics.append({"target": target, "r2": r2, "rmse": rmse})
        df_preds["target_name"] = target
        all_rows.append(df_preds)

    if all_rows:
        return pd.concat(all_rows, ignore_index=True), pd.DataFrame(metrics)
    else:
        return pd.DataFrame(), pd.DataFrame(metrics)


# --- Forecasting (multi-step iterative) ---

def forecast_horizon(full_df: pd.DataFrame,
                      target: str,
                      base_features: list,
                      horizon: int = 10,
                      model_ctor: Optional[Callable] = None) -> pd.DataFrame:
    """Forecast the given target for a multi-day horizon using an iterative approach."""
    if model_ctor is None:
        model_ctor = lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    df = full_df.sort_values("date").reset_index(drop=True).copy()
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe.")

    target_feats = [f"{target}_lag_{l}" for l in [1, 3, 7]] + [f"{target}_ema_7", f"{target}_rolling_center_7"]
    features = [c for c in base_features + target_feats if c in df.columns]
    if not features:
        raise ValueError("No features available for forecasting; check base_features and engineered features.")

    train = df.dropna(subset=features + [target])
    if train.shape[0] < 10:
        raise ValueError(f"Not enough rows to train forecast model (need >=10, got {train.shape[0]}).")

    X_train = train[features]
    y_train = train[target]

    model = model_ctor()
    model.fit(X_train, y_train)

    hist_series = df[target].tolist()
    preds = []

    last_row = df.iloc[[-1]]
    last_date = last_row["date"].iloc[0]

    for step in range(1, horizon + 1):
        forecast_date = last_date + pd.Timedelta(days=step)
        fv = {}

        if "day_of_week" in base_features or "day_of_week" in features:
            fv["day_of_week"] = forecast_date.weekday()
        if "month" in base_features or "month" in features:
            fv["month"] = forecast_date.month

        for bf in base_features:
            if bf in ["day_of_week", "month"]:
                continue
            fv[bf] = float(last_row[bf].iloc[0]) if (bf in last_row.columns and pd.notna(last_row[bf].iloc[0])) else 0.0

        sim = hist_series + preds

        for lag in [1, 3, 7]:
            pos = len(sim) - lag
            fv[f"{target}_lag_{lag}"] = float(sim[pos]) if pos >= 0 else 0.0

        window_vals = sim[-7:] if len(sim) >= 1 else [0.0]
        s = pd.Series(window_vals)
        fv[f"{target}_ema_7"] = float(s.ewm(span=7, adjust=False).mean().iloc[-1]) if not s.empty else 0.0
        if len(window_vals) >= 7:
            fv[f"{target}_rolling_center_7"] = float(s.rolling(window=7, center=True).mean().iloc[-1])
        else:
            fv[f"{target}_rolling_center_7"] = float(s.mean())

        Xf = pd.DataFrame([fv], columns=features).fillna(0.0)

        p = float(model.predict(Xf)[0])
        preds.append(p)

    result_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    res_df = pd.DataFrame({"date": result_dates, f"{target}_pred": preds})
    return res_df


def forecast_all_targets(full_df: pd.DataFrame, targets: list, base_features: list, horizon: int = 10) -> dict:
    """Return a dict of DataFrames with forecasts for each target."""
    out = {}
    for t in targets:
        try:
            out[t] = forecast_horizon(full_df, t, base_features, horizon=horizon)
        except Exception as e:
            out[t] = pd.DataFrame({"error": [str(e)]})
    return out


# --- Database utilities for saving forecasts ---

def save_forecasts_to_db(forecasts: Dict[str, pd.DataFrame], table_name: str = "sales_forecasts",
                         schema: Optional[str] = None, pk_col: str = "date_target"):
    """Normalize forecasts dict into a long DataFrame and upsert into Postgres.

    The function will create a surrogate primary key column `date_target` combining date and target
    so we can upsert multiple targets for the same date into a single-table primary-key-based update.
    """
    try:
        from database import upsert_df_to_postgres
    except Exception as e:
        print(f"Could not import database.upsert_df_to_postgres: {e}")
        raise

    rows = []
    for target, fdf in (forecasts or {}).items():
        if fdf is None or fdf.empty:
            continue
        # If the forecast DataFrame contains an 'error' column, skip and report
        if 'error' in fdf.columns:
            print(f"Skipping forecast for {target} due to error: {fdf['error'].iloc[0]}")
            continue

        # Identify prediction column (convention: '{target}_pred')
        pred_col_candidates = [c for c in fdf.columns if c.endswith('_pred')]
        pred_col = None
        if len(pred_col_candidates) == 1:
            pred_col = pred_col_candidates[0]
        else:
            # prefer exact match
            if f"{target}_pred" in fdf.columns:
                pred_col = f"{target}_pred"
            elif 'y_pred' in fdf.columns:
                pred_col = 'y_pred'
            else:
                # fallback: any non-date column
                non_date = [c for c in fdf.columns if c.lower() != 'date']
                pred_col = non_date[0] if non_date else None

        if pred_col is None:
            print(f"Could not find prediction column for {target}; skipping.")
            continue

        tmp = fdf.copy()
        if 'date' not in tmp.columns:
            print(f"Forecast for {target} has no 'date' column; skipping.")
            continue
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp[['date', pred_col]].rename(columns={pred_col: 'forecast'})
        tmp['target'] = target
        tmp['created_at'] = pd.Timestamp.utcnow()
        # create surrogate pk column combining date and target (strings)
        tmp['date_target'] = tmp['date'].dt.strftime('%Y-%m-%d') + '_' + tmp['target'].astype(str)
        rows.append(tmp)

    if not rows:
        print("No forecasts to save.")
        return

    out_df = pd.concat(rows, ignore_index=True)
    out_df['forecast'] = pd.to_numeric(out_df['forecast'], errors='coerce')

    # Upsert to Postgres using helper from database.py
    upsert_df_to_postgres(out_df, table_name, schema=schema, pk_col=pk_col)
    print(f"Saved {len(out_df)} forecast rows to '{table_name}'.")


# If run as script, execute existing holdout evaluation and then the backtest + forecasts
if __name__ == "__main__":
    print("Running holdout evaluation (existing code results):")
    for t, r, rm in results:
        print(f"{t:20} → R² = {r:.4f}, RMSE = {rm:.2f}")

    print("\nRunning walk-forward backtest for all targets:")
    preds_df, metrics_df = backtest_all_targets(combined, sales_categories, original_base, test_window=7)

    # Print unique target names to confirm
    if not preds_df.empty and "target_name" in preds_df.columns:
        unique_targets = sorted(preds_df["target_name"].dropna().unique().tolist())
        print(f"\nUnique target names found: {unique_targets}")
    else:
        print("\nNo target names found in preds_df")

    print(metrics_df.to_string(index=False))
    print(preds_df.head())

    # Run 10-day forecasts (simple call — no gating)
    print("\nRunning 10-day forecasts for all targets:")
    forecasts = forecast_all_targets(combined, sales_categories, original_base, horizon=10)
    for t, fdf in forecasts.items():
        print(f"\nForecast for {t}:")
        print(fdf.head(10))

    # Save forecasts to database
    print("\nSaving forecasts to database:")
    save_forecasts_to_db(forecasts)