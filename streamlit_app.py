from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pyarrow import parquet as pq

# ============================================================
# M5 DEMAND FORECASTING
# Single-file Streamlit portfolio app
# Frozen artifacts only:
# - No retraining
# - No replacement forecasting
# - No runtime inventory simulation
# - No fabricated inventory trajectories
# ============================================================

st.set_page_config(page_title="M5 Demand Forecasting", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed" / "predictions" / "app"

FILES = {
    "metadata": DATA_DIR / "app_sku_metadata.parquet",
    "forecasts": DATA_DIR / "app_forecasts.parquet",
    "actuals": DATA_DIR / "app_actuals.parquet",
    "history": DATA_DIR / "app_historical_demand.parquet",
    "inventory": DATA_DIR / "app_inventory_policy.parquet",
    "regime_trajectories": DATA_DIR / "app_regime_trajectories.parquet",
    "regime_actuals": DATA_DIR / "app_regime_actuals.parquet",
    # Optional: weekly historical demand for Intermittent / Lumpy SKUs (the daily history artifact covers Smooth / Erratic).
    "regime_history": DATA_DIR / "app_regime_history.parquet",
}
OPTIONAL_FILES = {"regime_history"}

CHART_FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

FORECAST_START = pd.Timestamp("2015-02-01")
FORECAST_END = pd.Timestamp("2016-01-31")
HISTORY_START = pd.Timestamp("2014-02-01")
HISTORY_END = pd.Timestamp("2015-01-31")

HISTORICAL_CONTEXT_DAYS = 28
PRIMARY_SERVICE_LEVEL = 0.80

DISPLAY_UNCERTAINTY = {"Balanced · q80": "q80", "More protection · q90": "q90", "Extra buffer · q95": "q95"}

REGIME_ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

# Forecast routes follow the authoritative regime.
DAILY_REGIMES = {"Smooth", "Erratic"}  # app_forecasts + app_actuals (daily)
WEEKLY_REGIMES = {"Intermittent", "Lumpy"}  # app_regime_trajectories + app_regime_actuals (weekly)
N_EVAL_WEEKS = 52  # complete 7-day evaluation periods, 2015-02-01 -> 2016-01-24

REGIME_METHODS = {
    "Smooth": "LightGBM Tweedie",
    "Erratic": "LightGBM Tweedie",
    "Intermittent": "TSB",
    "Lumpy": "Historical policy",
}


# ============================================================
# HELPERS
# ============================================================


def normalized_name(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_col(df, candidates, required=False):
    lookup = {normalized_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalized_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find one of {candidates}. " f"Available columns: {list(df.columns)}")
    return None


def find_col_like(columns, candidates):
    lookup = {normalized_name(c): c for c in columns}
    for candidate in candidates:
        key = normalized_name(candidate)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Could not find one of {candidates} " f"in {list(columns)}")


def to_ns(values):
    """Normalize any datetime resolution (s/ms/us/ns) to datetime64[ns].

    Parquet files written by different pandas/pyarrow versions carry different
    datetime units, and pandas 2.x treats them as unequal.
    """
    return pd.to_datetime(values).astype("datetime64[ns]")


def safe_str_series(series):
    # Categorical columns cannot be filled with a value outside their categories.
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    return series.fillna("").astype(str)


def fmt_num(value, digits=1):
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value, digits=1):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}%"


def normalize_service_level(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace("%", "")
    if text.startswith("q"):
        text = text[1:]
    try:
        numeric = float(text)
    except Exception:
        return np.nan
    return numeric / 100.0 if numeric > 1 else numeric


def normalize_bool(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def regime_label(value):
    if value is None or pd.isna(value):
        return "—"
    return str(value).strip().title()


def display_sku_name(sku):
    text = str(sku)
    suffix = "_validation"
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def _parquet_columns(path):
    return pq.read_schema(path).names


def _id_column_from_parquet(path):
    return find_col_like(_parquet_columns(path), ["id", "sku_id"])


def required_files_present():
    return all(path.exists() for key, path in FILES.items() if key not in OPTIONAL_FILES)


# ============================================================
# DATA LOADERS
# ============================================================


@st.cache_data(show_spinner=False)
def load_metadata():
    df = pd.read_parquet(FILES["metadata"])
    id_col = find_col(df, ["id", "sku_id"], required=True)
    df[id_col] = safe_str_series(df[id_col])
    return df


@st.cache_data(show_spinner=False)
def load_forecast_catalog_ids():
    """
    IDs with stored frozen daily forecast trajectories.
    """
    path = FILES["forecasts"]
    id_col = _id_column_from_parquet(path)
    ids = pd.read_parquet(path, columns=[id_col])[id_col]
    return pd.Index(safe_str_series(ids).drop_duplicates().sort_values())


@st.cache_data(show_spinner=False)
def load_sku_rows(kind, sku):
    """
    Read only the selected SKU from a frozen artifact.
    """
    path = FILES[kind]
    id_col = _id_column_from_parquet(path)
    df = pd.read_parquet(path, filters=[(id_col, "==", str(sku))])
    date_col = find_col(df, ["date"], required=True)
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"] = to_ns(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_inventory_sku(sku):
    path = FILES["inventory"]
    id_col = _id_column_from_parquet(path)
    return pd.read_parquet(path, filters=[(id_col, "==", str(sku))])


@st.cache_data(show_spinner=False)
def load_regime_trajectory(sku):
    """
    Frozen weekly trajectory for one Intermittent (TSB) or Lumpy (historical policy) SKU.
    """
    path = FILES["regime_trajectories"]
    id_col = _id_column_from_parquet(path)
    df = pd.read_parquet(path, filters=[(id_col, "==", str(sku))])
    df["period"] = to_ns(df["period"])
    return df.sort_values("period").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_regime_actuals(sku):
    """
    Frozen weekly observed demand for one Intermittent or Lumpy SKU (52 complete weeks).
    Reads only the selected SKU from the Parquet file.
    """
    path = FILES["regime_actuals"]
    id_col = _id_column_from_parquet(path)
    df = pd.read_parquet(path, filters=[(id_col, "==", str(sku))])
    df["period"] = to_ns(df["period"])
    return df.sort_values("period").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_regime_history(sku):
    """
    Frozen weekly historical demand for one Intermittent or Lumpy SKU.
    Returns an empty frame if the optional artifact is not present.
    """
    path = FILES["regime_history"]
    if not path.exists():
        return pd.DataFrame()
    id_col = _id_column_from_parquet(path)
    df = pd.read_parquet(path, filters=[(id_col, "==", str(sku))])
    df["period"] = to_ns(df["period"])
    return df.sort_values("period").reset_index(drop=True)


def _read_pooled_frame(kind, value_aliases, value_name, sku_index, date_aliases=("date",)):
    """
    Read (sku, date, value) from a frozen artifact.

    SKU ids are dictionary-encoded on read and mapped to integer
    positions in the metadata index to keep memory small.
    """
    path = FILES[kind]
    columns = _parquet_columns(path)
    id_name = find_col_like(columns, ["id", "sku_id"])
    date_name = find_col_like(columns, list(date_aliases))
    value_source = find_col_like(columns, value_aliases)
    df = pq.read_table(path, columns=[id_name, date_name, value_source], read_dictionary=[id_name]).to_pandas()
    df[id_name] = df[id_name].astype("category")
    categories = pd.Index(df[id_name].cat.categories.astype(str))
    lookup = sku_index.get_indexer(categories)
    codes = df[id_name].cat.codes.to_numpy()
    out = pd.DataFrame(
        {
            "sku": np.where(codes >= 0, lookup[codes], -1),
            "date": to_ns(df[date_name]),
            value_name: pd.to_numeric(df[value_source], errors="coerce"),
        }
    )
    return out[out["sku"] >= 0]


def portfolio_catalog():
    """
    De-duplicated metadata. Row position == the integer SKU code used by load_portfolio_summary().
    """
    meta = load_metadata()
    meta_id = find_col(meta, ["id", "sku_id"], required=True)
    return meta.drop_duplicates(subset=meta_id).reset_index(drop=True)


@st.cache_data(show_spinner="Loading portfolio summary…")
def load_portfolio_summary():
    """
    Compact per-SKU weekly table covering all four regimes, built once from frozen artifacts.

    Smooth / Erratic   : daily forecast + daily actual, summed to complete weeks.
    Intermittent / Lumpy: weekly trajectory + weekly regime actual, joined on id + period.

    Weeks use the app convention anchored to FORECAST_START. Only complete weeks
    (week index < N_EVAL_WEEKS) are kept; nothing is interpolated or invented.
    Returns columns: sku (catalog position), week, actual, forecast, abs_error
    (abs_error is measured on weekly totals for every route).
    """
    meta = portfolio_catalog()
    meta_id = find_col(meta, ["id", "sku_id"], required=True)
    meta_regime = find_col(meta, ["regime", "demand_regime"], required=True)
    sku_index = pd.Index(meta[meta_id].astype(str))
    regime_by_sku = meta[meta_regime].map(regime_label).to_numpy(dtype=object)
    parts = []

    # ---- daily route (Smooth / Erratic) ----
    forecast = _read_pooled_frame("forecasts", ["point_forecast", "forecast"], "forecast", sku_index)
    actual = _read_pooled_frame("actuals", ["actual_demand", "actual", "demand", "units", "sales", "y"], "actual", sku_index)
    merged = actual.merge(forecast, on=["sku", "date"], how="inner").dropna(subset=["actual", "forecast"])
    del forecast, actual
    merged = merged[np.isin(regime_by_sku[merged["sku"].to_numpy()], list(DAILY_REGIMES))]
    merged["week"] = ((merged["date"] - FORECAST_START).dt.days // 7).astype(int)
    merged = merged[(merged["week"] >= 0) & (merged["week"] < N_EVAL_WEEKS)]
    daily_weekly = merged.groupby(["sku", "week"], as_index=False).agg(actual=("actual", "sum"), forecast=("forecast", "sum"), days=("date", "count"))
    del merged
    parts.append(daily_weekly[daily_weekly["days"] == 7].drop(columns="days"))

    # ---- weekly route (Intermittent / Lumpy) ----
    trajectory = _read_pooled_frame("regime_trajectories", ["forecast"], "forecast", sku_index, date_aliases=("period",))
    weekly_actual = _read_pooled_frame("regime_actuals", ["actual"], "actual", sku_index, date_aliases=("period",))
    weekly = weekly_actual.merge(trajectory, on=["sku", "date"], how="inner").dropna(subset=["actual", "forecast"])
    del trajectory, weekly_actual
    weekly = weekly[np.isin(regime_by_sku[weekly["sku"].to_numpy()], list(WEEKLY_REGIMES))]
    weekly["week"] = ((weekly["date"] - FORECAST_START).dt.days // 7).astype(int)
    weekly = weekly[(weekly["week"] >= 0) & (weekly["week"] < N_EVAL_WEEKS)]
    parts.append(weekly[["sku", "week", "actual", "forecast"]])

    out = pd.concat(parts, ignore_index=True)
    out["abs_error"] = (out["forecast"] - out["actual"]).abs()
    return out.astype({"sku": "int32", "week": "int16", "actual": "float32", "forecast": "float32", "abs_error": "float32"})


# ============================================================
# CATALOG
# ============================================================


def apply_catalog_filters(metadata, filters):
    working = metadata
    for column, choice in filters:
        if column is not None and choice != "All":
            working = working[working[column].astype(str) == str(choice)]
    return working


def build_product_option(row, id_col, regime_col, item_col, cat_col, dept_col, store_col, state_col):
    sku = display_sku_name(row[id_col])
    pieces = []
    for column in [item_col, cat_col, dept_col, store_col, state_col, regime_col]:
        if column is not None and column in row.index and pd.notna(row[column]):
            value = str(row[column])
            if value not in pieces:
                pieces.append(value)
    if not pieces:
        return sku
    return f"{sku} · " f"{' · '.join(pieces)}"


# ============================================================
# INVENTORY POLICY
# ============================================================


def select_policy_row(policy, target_service_level):
    """
    Select an exact frozen service-level row.
    Never silently substitute another service level.
    """
    if policy.empty:
        return None
    service_col = find_col(policy, ["service_level", "service", "protection_level"])
    if service_col is None:
        return policy.iloc[0]
    work = policy.copy()
    work["_service"] = work[service_col].map(normalize_service_level)
    valid = work.dropna(subset=["_service"])
    if valid.empty:
        return None
    matches = valid[np.isclose(valid["_service"].to_numpy(dtype=float), float(target_service_level), atol=1e-9, rtol=0)]
    if matches.empty:
        return None
    validated_col = find_col(matches, ["validated_configuration", "validated"])
    if len(matches) > 1 and validated_col is not None:
        preferred = matches[matches[validated_col].map(normalize_bool) == True]
        if not preferred.empty:
            matches = preferred
    return matches.iloc[0]


def policy_value(policy_row, aliases):
    if policy_row is None:
        return None
    column = find_col(pd.DataFrame([policy_row]), aliases)
    if column is None:
        return None
    return policy_row[column]


def extract_policy_signals(policy_row):
    if policy_row is None:
        return []
    mapping = [
        (["expected_lead_time_demand"], "Lead-time demand"),
        (["safety_buffer"], "Safety buffer"),
        (["reorder_point", "protection_level"], "Reorder point"),
        (["order_qty", "order_quantity"], "Order quantity"),
    ]
    signals = []
    for aliases, label in mapping:
        value = policy_value(policy_row, aliases)
        if value is not None:
            signals.append((label, fmt_num(value, 2)))
    return signals


# ============================================================
# SELECTED SKU VALIDATION
# ============================================================


def validate_actual_history(actuals, history):
    """
    Integrity checks for rows that exist.

    A SKU without actual-evaluation rows or without history rows
    is a valid state (handled by the page), not an error.
    """
    if not actuals.empty and "date" not in actuals.columns:
        raise ValueError("Actual-demand artifact has no date column.")
    if history.empty:
        return
    if "date" not in history.columns:
        raise ValueError("Historical-demand artifact has no date column.")
    history_dates = pd.DatetimeIndex(history["date"]).sort_values()
    if history_dates.min() != HISTORY_START or history_dates.max() != HISTORY_END:
        raise ValueError("Unexpected frozen historical " "date range: " f"{history_dates.min():%Y-%m-%d} " "to " f"{history_dates.max():%Y-%m-%d}.")


def validate_forecast_data(forecasts, actuals):
    if forecasts.empty:
        return
    if "date" not in forecasts.columns:
        raise ValueError("Forecast artifact has no date column.")
    forecast_dates = pd.DatetimeIndex(to_ns(forecasts["date"])).sort_values()
    actual_dates = pd.DatetimeIndex(to_ns(actuals["date"])).sort_values()
    if not actual_dates.empty and not np.array_equal(forecast_dates.asi8, actual_dates.asi8):
        raise ValueError("Frozen forecast and actual " "dates do not align.")
    if forecast_dates.min() != FORECAST_START or forecast_dates.max() != FORECAST_END:
        raise ValueError("Unexpected frozen forecast " "date range: " f"{forecast_dates.min():%Y-%m-%d} " "to " f"{forecast_dates.max():%Y-%m-%d}.")


def validate_regime_trajectory(trajectory):
    if trajectory.empty:
        return
    if "period" not in trajectory.columns or "forecast" not in trajectory.columns:
        raise ValueError("Regime trajectory artifact is missing period/forecast columns.")
    if trajectory["period"].min() != FORECAST_START:
        raise ValueError(f"Unexpected frozen trajectory start: {trajectory['period'].min():%Y-%m-%d} " f"(expected {FORECAST_START:%Y-%m-%d}).")


def validate_regime_history(history):
    if history.empty:
        return
    periods = pd.DatetimeIndex(history["date"])
    if periods.min() < HISTORY_START or periods.max() + pd.Timedelta(days=6) > HISTORY_END:
        raise ValueError("Regime history periods fall outside the historical window or include an incomplete week.")


def validate_regime_actuals(actuals):
    if actuals.empty:
        return
    if "period" not in actuals.columns and "date" not in actuals.columns:
        raise ValueError("Regime actuals artifact is missing the period column.")
    periods = pd.DatetimeIndex(actuals["date"] if "date" in actuals.columns else actuals["period"])
    if periods.min() != FORECAST_START:
        raise ValueError(f"Unexpected regime actuals start: {periods.min():%Y-%m-%d} (expected {FORECAST_START:%Y-%m-%d}).")
    if periods.max() + pd.Timedelta(days=6) > FORECAST_END:
        raise ValueError("Regime actuals include an incomplete final week.")


# ============================================================
# CHART DATA
# ============================================================


def prepare_chart_data(history, actuals, forecasts, upper_quantile):
    actual_col = find_col(actuals, ["actual_demand", "actual", "demand", "units", "sales", "y"], required=True)
    history_actual_col = find_col(history, ["actual_demand", "actual", "demand", "units", "sales", "y"], required=True)
    historical = history[["date", history_actual_col]].copy().rename(columns={history_actual_col: "actual"})
    evaluation = actuals[["date", actual_col]].copy().rename(columns={actual_col: "actual"})
    historical["actual"] = pd.to_numeric(historical["actual"], errors="coerce")
    evaluation["actual"] = pd.to_numeric(evaluation["actual"], errors="coerce")
    forecast = pd.DataFrame()
    if not forecasts.empty:
        forecast_col = find_col(forecasts, ["point_forecast", "forecast"], required=True)
        forecast_columns = ["date", forecast_col]
        if "q50" in forecasts.columns:
            forecast_columns.append("q50")
        if upper_quantile in forecasts.columns:
            forecast_columns.append(upper_quantile)
        forecast = forecasts[forecast_columns].copy()
        rename = {forecast_col: "point_forecast"}
        if "q50" in forecast.columns:
            rename["q50"] = "q50"
        if upper_quantile in forecast.columns:
            rename[upper_quantile] = "upper"
        forecast = forecast.rename(columns=rename)
        for column in ["point_forecast", "q50", "upper"]:
            if column in forecast.columns:
                forecast[column] = pd.to_numeric(forecast[column], errors="coerce")
        forecast = forecast.sort_values("date").reset_index(drop=True)
    historical = historical.sort_values("date").reset_index(drop=True)
    evaluation = evaluation.sort_values("date").reset_index(drop=True)
    return (historical, evaluation, forecast)


def prepare_trajectory(trajectory, upper_quantile, display_end):
    """
    Frozen weekly trajectory -> chart frame (complete weeks inside the display window).

    The uncertainty columns are constant frozen 7-day lead-time-demand bootstrap quantiles.
    They are NaN for Lumpy, in which case no q50/upper column is returned and no band is drawn.
    """
    window = trajectory[trajectory["period"] + pd.Timedelta(days=6) <= display_end]
    window = window.sort_values("period").reset_index(drop=True)
    forecast = pd.DataFrame({"period": window["period"], "point_forecast": pd.to_numeric(window["forecast"], errors="coerce")})
    for source, target in [("uncertainty_q50", "q50"), (f"uncertainty_{upper_quantile}", "upper")]:
        if source in window.columns and window[source].notna().any():
            forecast[target] = pd.to_numeric(window[source], errors="coerce")
    return forecast


def evaluation_window_end(display_days, weekly):
    """
    Last date of the evaluation window that matches the selected display window.

    Weekly regimes use whole 7-day periods (7 -> 1 week, 28 -> 4, 90 -> 12, 365 -> 52);
    daily regimes use the first `display_days` days. Same FORECAST_START anchor as the chart.
    """
    days = max(display_days // 7, 1) * 7 if weekly else display_days
    return min(FORECAST_START + pd.Timedelta(days=days - 1), FORECAST_END)


def weekly_metric_frames(actuals, trajectory, display_days=365):
    """
    Weekly actual / forecast pairs for the weekly-trajectory regimes.

    `actuals` holds the frozen weekly regime actuals (date == weekly period start).
    Only complete 7-day periods inside the selected display window are kept, so the trailing
    2016-01-31 snapshot point and any partial week are excluded.
    Pairing is by period (id is constant for a single SKU).
    """
    window_end = evaluation_window_end(display_days, weekly=True)
    actual_col = find_col(actuals, ["actual_demand", "actual", "demand", "units", "sales", "y"], required=True)
    complete = lambda s: (s >= FORECAST_START) & (s + pd.Timedelta(days=6) <= window_end)
    weekly_actual = actuals[complete(actuals["date"])][["date", actual_col]]
    weeks = trajectory[complete(trajectory["period"])]
    return weekly_actual, weeks.rename(columns={"period": "date"})[["date", "forecast"]]


# ============================================================
# CHART
# ============================================================


def trace_mode(x):
    return "lines+markers" if len(x) == 1 else "lines"


def build_chart(
    history, actuals, forecasts, display_days, weekly, upper_quantile, trajectory=None, series_name="Forecast", range_name="Planning range"
):
    history_start = FORECAST_START - pd.Timedelta(days=HISTORICAL_CONTEXT_DAYS)
    # Weekly trajectories (Intermittent / Lumpy) show complete weeks only.
    from_trajectory = trajectory is not None
    display_end = FORECAST_START + pd.Timedelta(days=(display_days // 7 * 7 if from_trajectory else display_days) - 1)
    historical, evaluation, forecast = prepare_chart_data(history, actuals, forecasts, upper_quantile)
    if from_trajectory:
        forecast = prepare_trajectory(trajectory, upper_quantile, display_end)
    historical = historical[(historical["date"] >= history_start) & (historical["date"] < FORECAST_START)].copy()
    evaluation = evaluation[(evaluation["date"] >= FORECAST_START) & (evaluation["date"] <= display_end)].copy()
    if not forecast.empty and not from_trajectory:
        forecast = forecast[(forecast["date"] >= FORECAST_START) & (forecast["date"] <= display_end)].copy()
    # --------------------------------------------------------
    # Weekly aggregation
    # --------------------------------------------------------
    if weekly:
        for frame in [historical, evaluation] + ([] if from_trajectory else [forecast]):
            if frame.empty:
                if "date" in frame.columns:
                    frame["period"] = pd.Series(dtype="datetime64[ns]")
                continue
            frame["week_index"] = ((frame["date"] - FORECAST_START).dt.days // 7).astype(int)
            frame["period"] = FORECAST_START + pd.to_timedelta(frame["week_index"] * 7, unit="D")
        historical = historical.groupby("period", as_index=False)["actual"].sum().sort_values("period")
        evaluation = evaluation.groupby("period", as_index=False)["actual"].sum().sort_values("period")
        if not forecast.empty and not from_trajectory:
            aggregation = {"point_forecast": "sum"}
            if "q50" in forecast.columns:
                aggregation["q50"] = "sum"
            if "upper" in forecast.columns:
                aggregation["upper"] = "sum"
            forecast = forecast.groupby("period", as_index=False).agg(aggregation).sort_values("period")
        historical_x = historical["period"]
        historical_y = historical["actual"]
        actual_x = evaluation["period"]
        actual_y = evaluation["actual"]
        if not forecast.empty:
            forecast_x = forecast["period"]
            forecast_y = forecast["point_forecast"]
    else:
        historical_x = historical["date"]
        historical_y = historical["actual"]
        actual_x = evaluation["date"]
        actual_y = evaluation["actual"]
        if not forecast.empty:
            forecast_x = forecast["date"]
            forecast_y = forecast["point_forecast"]
    fig = go.Figure()
    # --------------------------------------------------------
    # Historical actual
    # --------------------------------------------------------
    if len(historical):
        fig.add_trace(
            go.Scatter(
                x=historical_x,
                y=historical_y,
                mode=trace_mode(historical_x),
                name="Historical actual",
                line=dict(width=2, color="#94A3B8"),
                showlegend=True,
                connectgaps=True,
                hovertemplate=("Date=%{x|%b %d, %Y}" "<br>Actual=%{y:,.1f}" "<extra></extra>"),
            )
        )
    # --------------------------------------------------------
    # Visual connector (display only)
    #
    # Joins the last historical point to the first post-origin
    # point so the timeline reads as continuous. It is not a
    # data series, forecast value, or observation, and it is
    # excluded from the legend and hover.
    # --------------------------------------------------------
    connector_end = None
    if len(evaluation):
        connector_end = (actual_x.iloc[0], actual_y.iloc[0])
    elif not forecast.empty:
        connector_end = (forecast_x.iloc[0], forecast_y.iloc[0])
    if len(historical) and connector_end is not None:
        fig.add_trace(
            go.Scatter(
                x=[historical_x.iloc[-1], connector_end[0]],
                y=[historical_y.iloc[-1], connector_end[1]],
                mode="lines",
                line=dict(width=2, color="#94A3B8"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    # --------------------------------------------------------
    # Evaluation-period actual
    # --------------------------------------------------------
    if len(evaluation):
        fig.add_trace(
            go.Scatter(
                x=actual_x,
                y=actual_y,
                mode=trace_mode(actual_x),
                name="Actual",
                line=dict(width=3.2, color="#60A5FA"),
                showlegend=True,
                connectgaps=True,
                hovertemplate=("Date=%{x|%b %d, %Y}" "<br>Actual=%{y:,.1f}" "<extra></extra>"),
            )
        )
    # --------------------------------------------------------
    # Planning range
    # --------------------------------------------------------
    if not forecast.empty and "q50" in forecast.columns and "upper" in forecast.columns:
        band_x = forecast["period"] if weekly else forecast["date"]
        # Upper boundary.
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=forecast["upper"],
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                connectgaps=True,
            )
        )
        # q50 boundary + fill.
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=forecast["q50"],
                mode="lines",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                fill="tonexty",
                fillcolor=("rgba(37,99,235,0.13)"),
                name=f"{range_name} · q50–{upper_quantile}",
                showlegend=True,
                hoverinfo="skip",
                connectgaps=True,
            )
        )
    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------
    if not forecast.empty:
        customdata = None
        if "q50" in forecast.columns and "upper" in forecast.columns:
            customdata = np.column_stack([forecast["q50"].to_numpy(), forecast["upper"].to_numpy()])
            if from_trajectory:
                hovertemplate = (
                    "Week of %{x|%b %d, %Y}"
                    f"<br>{series_name}=%{{y:,.1f}}"
                    "<br>q50 7-day demand=%{customdata[0]:,.1f}"
                    f"<br>{upper_quantile} 7-day demand=%{{customdata[1]:,.1f}}"
                    "<extra></extra>"
                )
            elif weekly:
                hovertemplate = (
                    "Week of "
                    "%{x|%b %d, %Y}"
                    "<br>Forecast="
                    "%{y:,.1f}"
                    "<br>q50 daily-scenario sum="
                    "%{customdata[0]:,.1f}"
                    f"<br>{upper_quantile} "
                    "daily-scenario sum="
                    "%{customdata[1]:,.1f}"
                    "<extra></extra>"
                )
            else:
                hovertemplate = (
                    "Date="
                    "%{x|%b %d, %Y}"
                    "<br>Forecast="
                    "%{y:,.1f}"
                    "<br>q50="
                    "%{customdata[0]:,.1f}"
                    f"<br>{upper_quantile}="
                    "%{customdata[1]:,.1f}"
                    "<extra></extra>"
                )
        else:
            hovertemplate = ("Week of " if weekly else "Date=") + "%{x|%b %d, %Y}" + f"<br>{series_name}=" + "%{y:,.1f}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode=trace_mode(forecast_x),
                name=series_name,
                line=dict(width=2.8, dash="dash", color="#2563EB"),
                customdata=customdata,
                hovertemplate=hovertemplate,
                connectgaps=True,
            )
        )
    # --------------------------------------------------------
    # Forecast origin
    #
    # Uses the exact x value of the first plotted forecast point,
    # which is also where q50 and the upper quantile begin.
    # The marker is a shape/annotation, so it is not a legend item.
    # --------------------------------------------------------
    origin_x = None
    if not forecast.empty:
        origin_x = forecast_x.iloc[0]
    elif len(evaluation):
        origin_x = actual_x.iloc[0]
    if origin_x is not None:
        origin_x = pd.Timestamp(origin_x)
        fig.add_shape(type="line", x0=origin_x, x1=origin_x, xref="x", y0=0, y1=1, yref="paper", line=dict(color="#475569", width=1.5, dash="dot"))
        fig.add_annotation(
            x=origin_x,
            y=0.97,
            yref="paper",
            text="Forecast origin",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=7,
            font=dict(size=11, color="#475569"),
        )
    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------
    fig.update_layout(
        height=500,
        margin=dict(l=8, r=8, t=18, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.14, x=0, xanchor="left", yanchor="top"),
        yaxis_title=("Units / week" if weekly else "Units / day"),
        xaxis_title="",
        plot_bgcolor=("rgba(0,0,0,0)"),
        paper_bgcolor=("rgba(0,0,0,0)"),
        hoverlabel=dict(namelength=-1),
        font=dict(family=CHART_FONT, size=12),
        xaxis=dict(showgrid=False, zeroline=False, rangemode="normal"),
        yaxis=dict(showgrid=True, gridcolor=("rgba(100,116,139,0.14)"), zeroline=False),
    )
    return fig


# ============================================================
# FORECAST METRICS
# ============================================================


def calculate_metrics(actuals, forecasts, end_date=None):
    """
    MAE / WAPE / Bias over the rows shared by actuals and forecasts.
    `end_date` (inclusive) restricts the evaluation to the selected display window.
    """
    if forecasts.empty:
        return None
    if end_date is not None:
        actuals = actuals[(actuals["date"] >= FORECAST_START) & (actuals["date"] <= end_date)]
        forecasts = forecasts[(forecasts["date"] >= FORECAST_START) & (forecasts["date"] <= end_date)]
    forecast_col = find_col(forecasts, ["point_forecast", "forecast"], required=True)
    actual_col = find_col(actuals, ["actual_demand", "actual", "demand", "units", "sales", "y"], required=True)
    merged = actuals[["date", actual_col]].merge(forecasts[["date", forecast_col]], on="date", how="inner", validate="one_to_one")
    merged["actual"] = pd.to_numeric(merged[actual_col], errors="coerce")
    merged["forecast"] = pd.to_numeric(merged[forecast_col], errors="coerce")
    merged = merged.dropna(subset=["actual", "forecast"])
    if merged.empty:
        return None
    error = merged["forecast"] - merged["actual"]
    actual_total = merged["actual"].abs().sum()
    return {
        "rows": len(merged),
        "mae": error.abs().mean(),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": (error.abs().sum() / actual_total * 100 if actual_total > 0 else np.nan),
        "bias": (error.sum() / actual_total * 100 if actual_total > 0 else np.nan),
    }


# ============================================================
# PORTFOLIO OVERVIEW HELPERS
# ============================================================


def summarize_portfolio(per_sku_week):
    """
    Pooled metrics come from weekly actual and forecast totals across the selected SKUs.
    Median SKU WAPE is the median of each SKU's own WAPE over the same weeks.
    These are different measures.
    """
    totals = per_sku_week.groupby("week", as_index=False)[["actual", "forecast"]].sum()
    totals[["actual", "forecast"]] = totals[["actual", "forecast"]].astype("float64")
    totals["period"] = FORECAST_START + pd.to_timedelta(totals["week"].astype(int) * 7, unit="D")
    totals = totals.sort_values("period").reset_index(drop=True)
    metrics = calculate_metrics(
        totals[["period", "actual"]].rename(columns={"period": "date"}), totals[["period", "forecast"]].rename(columns={"period": "date"})
    )
    per_sku = per_sku_week.groupby("sku")[["actual", "abs_error"]].sum().astype("float64")
    with_demand = per_sku[per_sku["actual"] > 0]
    median_sku_wape = (with_demand["abs_error"] / with_demand["actual"] * 100).median() if not with_demand.empty else np.nan
    return totals, metrics, median_sku_wape, len(per_sku)


def apply_overview_layout(fig, height, y_title="", show_legend=True):
    fig.update_layout(
        height=height,
        font=dict(family=CHART_FONT, size=12),
        margin=dict(l=8, r=8, t=12, b=52 if show_legend else 30),
        showlegend=show_legend,
        legend=dict(orientation="h", y=-0.16, x=0, xanchor="left", yanchor="top"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", showgrid=False, zeroline=False),
        yaxis=dict(title=y_title, showgrid=True, gridcolor="rgba(100,116,139,0.14)", zeroline=False),
    )
    return fig


def build_portfolio_chart(totals):
    """
    Weekly aggregate actual demand vs forecast (line chart).
    `totals` has one row per complete weekly period: period, actual, forecast.
    """
    weekly = totals.sort_values("period").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly["period"],
            y=weekly["actual"],
            mode=trace_mode(weekly["period"]),
            name="Actual demand",
            line=dict(width=3.2, color="#60A5FA"),
            hovertemplate=("Week of %{x|%b %d, %Y}" "<br>Actual=%{y:,.0f}" "<extra></extra>"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weekly["period"],
            y=weekly["forecast"],
            mode=trace_mode(weekly["period"]),
            name="Forecast",
            line=dict(width=2.8, dash="dash", color="#2563EB"),
            hovertemplate=("Week of %{x|%b %d, %Y}" "<br>Forecast=%{y:,.0f}" "<extra></extra>"),
        )
    )
    if not weekly.empty:
        origin_x = pd.Timestamp(weekly["period"].iloc[0])
        fig.add_shape(type="line", x0=origin_x, x1=origin_x, xref="x", y0=0, y1=1, yref="paper", line=dict(color="#475569", width=1.5, dash="dot"))
        fig.add_annotation(
            x=origin_x,
            y=0.97,
            yref="paper",
            text="Forecast origin",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            xshift=7,
            font=dict(size=11, color="#475569"),
        )
    apply_overview_layout(fig, height=380, y_title="Units / week")
    fig.update_layout(hovermode="x unified")
    return fig


# ============================================================
# STYLE
# ============================================================

CUSTOM_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 4rem; }

/* Hero */
[data-testid="stElementContainer"]:has(.hero), [data-testid="element-container"]:has(.hero) { position: relative; z-index: 1000; }
.view-label { text-align: right; padding-top: .55rem; font-size: .9rem; opacity: .65; }
.hero { position: relative; z-index: 1000; padding: .2rem 0 1.1rem; margin-bottom: .4rem; border-bottom: 1px solid rgba(128,128,128,.18); }
.hero-title { font-size: 2.5rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.05; margin: 0; }
.hero-subtitle { font-size: 1.05rem; opacity: .68; margin-top: .5rem; font-weight: 400; }
.info-tip { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 1.05rem; height: 1.05rem;
  margin-left: .5rem; border: 1px solid rgba(128,128,128,.55); border-radius: 50%; font-size: .68rem; font-weight: 600;
  font-style: normal; cursor: help; vertical-align: middle; opacity: .8; }
.info-tip:hover { opacity: 1; border-color: #2563EB; color: #2563EB; }
.info-tip .tip-body { display: none; position: absolute; top: 1.6rem; left: -.4rem; z-index: 1000; width: min(460px, 82vw);
  padding: .8rem .95rem; border-radius: 10px; background: rgba(15,23,42,.98); color: #E2E8F0; font-size: .82rem;
  font-weight: 400; line-height: 1.5; text-align: left; box-shadow: 0 8px 24px rgba(0,0,0,.35); opacity: 1; cursor: default; }
.info-tip:hover .tip-body { display: block; }
.snapshot { display: inline-flex; align-items: center; gap: .45rem; margin-top: .9rem; padding: .25rem .75rem;
  border: 1px solid rgba(37,99,235,.35); background: rgba(37,99,235,.07); color: #2563EB;
  border-radius: 999px; font-size: .75rem; font-weight: 500; letter-spacing: .01em; }
.snapshot::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: #2563EB; }

/* Sections */
.section-title { font-size: 1.05rem; font-weight: 650; letter-spacing: -.01em; margin: 1.6rem 0 .7rem;
  padding-left: .7rem; border-left: 3px solid #2563EB; line-height: 1.25; }

/* Product header */
.product-name { font-size: 1.6rem; font-weight: 700; letter-spacing: -.02em; margin-top: .3rem; }
.product-meta { opacity: .7; margin: .35rem 0 .5rem; font-size: .92rem; display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }
.regime-pill { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .74rem; font-weight: 600;
  background: rgba(37,99,235,.12); color: #2563EB; border: 1px solid rgba(37,99,235,.28); opacity: 1; }

/* Forecast info panel */
.info-panel { border: 1px solid rgba(128,128,128,.18); border-radius: 12px; padding: .9rem 1rem .3rem; background: rgba(128,128,128,.04); }
.info-label { font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; opacity: .55; margin-bottom: .15rem; font-weight: 500; }
.info-value { font-size: 1rem; font-weight: 600; margin-bottom: .8rem; }

/* Metrics */
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.18); border-radius: 12px; padding: .8rem 1rem;
  background: rgba(128,128,128,.04); }
[data-testid="stMetricLabel"] p { font-size: .76rem; text-transform: uppercase; letter-spacing: .05em; opacity: .65; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 650; letter-spacing: -.02em; }

/* Charts */
[data-testid="stPlotlyChart"] { border: 1px solid rgba(128,128,128,.16); border-radius: 14px; padding: .5rem .6rem .2rem; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.16); }
[data-testid="stSidebar"] h3 { font-size: .8rem; text-transform: uppercase; letter-spacing: .09em; opacity: .7; font-weight: 600; margin-top: .6rem; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { border-radius: 8px; }

/* How the system works */
.flow { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1.5rem; margin: .4rem 0 1.4rem; }
.flow-step { position: relative; border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: .85rem .85rem .9rem;
  background: rgba(128,128,128,.04); }
.flow-step:not(:last-child)::after { content: "›"; position: absolute; right: -1.15rem; top: 50%; transform: translateY(-50%);
  font-size: 1.5rem; color: #2563EB; font-weight: 600; }
.flow-num { display: inline-flex; align-items: center; justify-content: center; width: 1.4rem; height: 1.4rem; border-radius: 50%;
  background: #2563EB; color: #fff; font-size: .72rem; font-weight: 600; margin-bottom: .5rem; }
.flow-title { font-size: .92rem; font-weight: 650; margin-bottom: .25rem; }
.flow-desc { font-size: .78rem; opacity: .7; line-height: 1.45; }
.route-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: .3rem 0 1.2rem; }
.route-card { border: 1px solid rgba(128,128,128,.2); border-top: 3px solid #2563EB; border-radius: 10px; padding: .8rem .95rem;
  background: rgba(128,128,128,.04); }
.route-regime { font-size: .95rem; font-weight: 650; }
.route-count { font-size: .74rem; opacity: .55; margin-bottom: .45rem; }
.route-when { font-size: .8rem; opacity: .7; margin-bottom: .55rem; line-height: 1.4; }
.route-method { font-size: .82rem; font-weight: 600; color: #2563EB; }
.route-note { font-size: .74rem; opacity: .6; font-weight: 400; margin-top: .15rem; }
@media (max-width: 1100px) { .flow { grid-template-columns: repeat(3, 1fr); } .route-grid { grid-template-columns: repeat(2, 1fr); }
  .flow-step:nth-child(3n)::after { display: none; } }
@media (max-width: 640px) { .flow, .route-grid { grid-template-columns: 1fr; } .flow-step::after { display: none; } }
.app-footer { margin-top: 2.5rem; padding-top: 1.1rem; border-top: 1px solid rgba(128,128,128,.18); font-size: .8rem;
  opacity: .6; text-align: center; line-height: 1.8; }

/* Misc */
[data-testid="stExpander"] { border: 1px solid rgba(128,128,128,.18); border-radius: 12px; }
[data-testid="stCaptionContainer"] { opacity: .75; line-height: 1.5; }
hr { margin: 2rem 0 1rem; opacity: .5; }
</style>"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# FILE CHECK
# ============================================================

if not required_files_present():
    st.error("The frozen application data bundle " "is not available.")
    st.code("\n".join(f"{key}: {path}" for key, path in FILES.items()))
    st.stop()


# ============================================================
# LOAD FULL CATALOG
# ============================================================

metadata = load_metadata()
forecast_ids = load_forecast_catalog_ids()

forecast_id_set = set(forecast_ids.astype(str))

id_col = find_col(metadata, ["id", "sku_id"], required=True)

regime_col = find_col(metadata, ["regime", "demand_regime"])

state_col = find_col(metadata, ["state_id", "state"])

store_col = find_col(metadata, ["store_id", "store"])

dept_col = find_col(metadata, ["dept_id", "department", "dept"])

cat_col = find_col(metadata, ["cat_id", "category", "cat"])

item_col = find_col(metadata, ["item_id", "item"])

metadata[id_col] = safe_str_series(metadata[id_col])


# ============================================================
# SIDEBAR — FIND PRODUCT
# ============================================================

st.sidebar.markdown("### Find a product")

st.sidebar.caption("Browse the hierarchy or type directly " "into the product dropdown.")

regime_choice = "All"
state_choice = "All"
store_choice = "All"
dept_choice = "All"
cat_choice = "All"

browse_filters = []


# ------------------------------------------------------------
# Regime
# ------------------------------------------------------------

if regime_col is not None:
    regime_values = metadata[regime_col].dropna().astype(str).str.strip().unique().tolist()
    regime_values = sorted(regime_values, key=lambda x: (REGIME_ORDER.index(x.title()) if x.title() in REGIME_ORDER else 99, x.lower()))
    regime_choice = st.sidebar.selectbox(
        "Regime", ["All"] + regime_values, key="browse_regime", format_func=regime_label, help=("Frozen demand regime assigned " "using ADI and CV².")
    )
    if regime_choice != "All":
        browse_filters.append((regime_col, regime_choice))


current = apply_catalog_filters(metadata, browse_filters)


# ------------------------------------------------------------
# State
# ------------------------------------------------------------

if regime_choice != "All" and state_col is not None and not current.empty:
    state_values = sorted(current[state_col].dropna().astype(str).unique().tolist())
    state_choice = st.sidebar.selectbox("State", ["All"] + state_values, key=(f"browse_state_" f"{regime_choice}"))
    if state_choice != "All":
        browse_filters.append((state_col, state_choice))


current = apply_catalog_filters(metadata, browse_filters)


# ------------------------------------------------------------
# Store
# ------------------------------------------------------------

if regime_choice != "All" and state_choice != "All" and store_col is not None and not current.empty:
    store_values = sorted(current[store_col].dropna().astype(str).unique().tolist())
    store_choice = st.sidebar.selectbox("Store", ["All"] + store_values, key=(f"browse_store_" f"{regime_choice}_" f"{state_choice}"))
    if store_choice != "All":
        browse_filters.append((store_col, store_choice))


current = apply_catalog_filters(metadata, browse_filters)


# ------------------------------------------------------------
# Department
# ------------------------------------------------------------

if regime_choice != "All" and state_choice != "All" and store_choice != "All" and dept_col is not None and not current.empty:
    dept_values = sorted(current[dept_col].dropna().astype(str).unique().tolist())
    dept_choice = st.sidebar.selectbox(
        "Department", ["All"] + dept_values, key=(f"browse_department_" f"{regime_choice}_" f"{state_choice}_" f"{store_choice}")
    )
    if dept_choice != "All":
        browse_filters.append((dept_col, dept_choice))


current = apply_catalog_filters(metadata, browse_filters)


# ------------------------------------------------------------
# Category
# ------------------------------------------------------------

if regime_choice != "All" and state_choice != "All" and store_choice != "All" and dept_choice != "All" and cat_col is not None and not current.empty:
    cat_values = sorted(current[cat_col].dropna().astype(str).unique().tolist())
    cat_choice = st.sidebar.selectbox(
        "Category", ["All"] + cat_values, key=(f"browse_category_" f"{regime_choice}_" f"{state_choice}_" f"{store_choice}_" f"{dept_choice}")
    )
    if cat_choice != "All":
        browse_filters.append((cat_col, cat_choice))


current = apply_catalog_filters(metadata, browse_filters)


# ============================================================
# PRODUCT DROPDOWN
# ============================================================

if current.empty:
    st.sidebar.warning("No products match these selections.")
    st.stop()


product_options = []
product_lookup = {}

for _, row in current.sort_values(id_col).iterrows():
    option = build_product_option(
        row=row, id_col=id_col, regime_col=regime_col, item_col=item_col, cat_col=cat_col, dept_col=dept_col, store_col=store_col, state_col=state_col
    )
    product_options.append(option)
    product_lookup[option] = str(row[id_col])


selected_product = st.sidebar.selectbox(
    "Product · type to search",
    product_options,
    key=(f"product_selector_" f"{regime_choice}_" f"{state_choice}_" f"{store_choice}_" f"{dept_choice}_" f"{cat_choice}"),
    help=("Type a SKU, item, category, department, " "store, state, or regime to search."),
)

selected_sku = product_lookup[selected_product]


# ============================================================
# SELECTED METADATA
# ============================================================

selected_meta = metadata[metadata[id_col].astype(str) == str(selected_sku)].iloc[0]

selected_regime = regime_label(selected_meta[regime_col]) if (regime_col is not None and pd.notna(selected_meta[regime_col])) else "—"

# Routing follows the SKU's authoritative regime, not membership in app_forecasts.
regime_requires_weekly = selected_regime in WEEKLY_REGIMES


# ============================================================
# SIDEBAR — CHART CONTROLS
# ============================================================

st.sidebar.markdown("### Chart")

display_days = st.sidebar.selectbox(
    "Display window",
    [7, 28, 90, 365],
    index=3,
    format_func=lambda value: f"{value} days",
    help=("Changes how much of the stored " "forecast trajectory is displayed. " "It does not change the forecast."),
)


if regime_requires_weekly:
    aggregation = "Weekly"
    st.sidebar.radio(
        "Display",
        ["Weekly"],
        index=0,
        horizontal=True,
        help=("Intermittent and lumpy demand are " "shown at weekly resolution in the " "interactive application."),
    )

else:
    aggregation = st.sidebar.radio(
        "Display", ["Daily", "Weekly"], index=1, horizontal=True, help=("Weekly view groups the stored daily " "values into seven-day periods.")
    )


uncertainty_label = st.sidebar.selectbox(
    "Planning range",
    list(DISPLAY_UNCERTAINTY),
    help=("q80 is the primary validated configuration. " "q90 and q95 are higher-protection scenario " "alternatives."),
)

selected_quantile = DISPLAY_UNCERTAINTY[uncertainty_label]


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">M5 Demand Forecasting</div>'
    '<div class="hero-subtitle">Forecasting demand and planning inventory.'
    '<span class="info-tip">i<span class="tip-body">'
    "An end-to-end demand forecasting and inventory-planning framework built on the M5 retail dataset (30,490 SKUs). "
    "SKUs are grouped into four demand regimes (Smooth, Erratic, Intermittent, Lumpy), and each regime uses a suitable method: "
    "LightGBM Tweedie, TSB, or a historical policy. Forecasts and demand uncertainty feed a frozen inventory policy, "
    "and results are evaluated on future periods in a time-series backtest (locked Fold 3). "
    "The app only displays frozen results; nothing is retrained or simulated here."
    '</span></span></div>'
    '<div class="snapshot">Historical snapshot · 2015–2016</div>'
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PORTFOLIO OVERVIEW
# Shown only on first load (all dropdowns at their defaults).
# ============================================================

show_overview = regime_choice == "All" and selected_product == product_options[0]

if show_overview:
    title_col, label_col, view_col, selection_col = st.columns([2.2, 0.3, 1, 1], gap="small")
    with title_col:
        st.markdown('<div class="section-title" style="margin-top:.35rem">Portfolio overview</div>', unsafe_allow_html=True)
    with label_col:
        st.markdown('<div class="view-label">View</div>', unsafe_allow_html=True)
    try:
        portfolio_weekly = load_portfolio_summary()
    except Exception as exc:
        portfolio_weekly = None
        st.warning(f"Portfolio forecast summary is unavailable: {exc}")
    if portfolio_weekly is not None and not portfolio_weekly.empty:
        catalog = portfolio_catalog()
        catalog_regime = catalog[find_col(catalog, ["regime", "demand_regime"], required=True)].map(regime_label)
        view_columns = {"All": None, "Regime": regime_col, "State": state_col, "Store": store_col, "Department": dept_col, "Category": cat_col}
        view_columns = {name: column for name, column in view_columns.items() if name == "All" or column is not None}
        view_choice = view_col.selectbox(
            "View",
            list(view_columns),
            key="portfolio_view",
            format_func=lambda name: name if name == "All" else f"By {name.lower()}",
            label_visibility="collapsed",
            help="Group the aggregate line chart by a catalog dimension.",
        )
        view_column = view_columns[view_choice]
        selection = "All"
        sku_mask = np.ones(len(catalog), dtype=bool)
        if view_column is not None:
            dimension = catalog_regime if view_column == regime_col else catalog[view_column].astype(str)
            options = sorted(dimension.dropna().unique().tolist())
            if view_column == regime_col:
                options = [r for r in REGIME_ORDER if r in options] + [r for r in options if r not in REGIME_ORDER]
            selection = selection_col.selectbox(view_choice, options, key=f"portfolio_selection_{view_choice}", label_visibility="collapsed")
            sku_mask = (dimension == selection).to_numpy()
        # Follows the sidebar display window (complete weeks only).
        n_weeks = min(max(display_days // 7, 1), N_EVAL_WEEKS)
        selected_rows = portfolio_weekly[(portfolio_weekly["week"] < n_weeks) & sku_mask[portfolio_weekly["sku"].to_numpy()]]
        if selected_rows.empty:
            st.info("No frozen forecast rows are available for this selection.")
        else:
            totals, pooled, median_sku_wape, n_skus = summarize_portfolio(selected_rows)
            present = set(catalog_regime.to_numpy()[np.unique(selected_rows["sku"].to_numpy())])
            overview_config = {"displaylogo": False}
            st.plotly_chart(build_portfolio_chart(totals), width="stretch", config=overview_config, key="portfolio_aggregate_chart")
            if pooled:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Aggregate MAE", fmt_num(pooled["mae"], 1), help="Average weekly gap, in units, between total forecast and total actual demand.")
                m2.metric("Aggregate WAPE", fmt_pct(pooled["wape"], 1), help="Pooled WAPE: total absolute error of the weekly totals relative to total demand.")
                m3.metric(
                    "Aggregate Bias",
                    fmt_pct(pooled["bias"], 1),
                    help="Signed error of the pooled weekly totals relative to total demand. Positive means the forecast tends to be high.",
                )
                m4.metric("Median SKU WAPE", fmt_pct(median_sku_wape, 1), help="Median of each SKU's own weekly WAPE. It describes a typical individual SKU.")
            scope = "the full portfolio" if view_column is None else f"{view_choice.lower()} {selection}"
            methods = [f"{r}: {REGIME_METHODS[r]}" for r in REGIME_ORDER if r in present]
            st.caption(
                f"Weekly totals for {scope} · {n_skus:,} SKUs · complete weeks only. "
                f"Methods differ by demand pattern ({'; '.join(methods)}). "
                + ("Lumpy is a policy level, not a statistical forecast. " if "Lumpy" in present else "")
                + "Errors are measured on weekly totals. Aggregate metrics compare totals across SKUs; "
                "median SKU WAPE is the middle of the individual SKU WAPEs. They measure different things."
            )
    st.divider()


# ============================================================
# PRODUCT HEADER
# ============================================================

meta_bits = []

for column in [item_col, cat_col, dept_col, store_col, state_col]:
    if column is not None and pd.notna(selected_meta[column]):
        value = str(selected_meta[column])
        if value not in meta_bits:
            meta_bits.append(value)


st.markdown((f'<div class="product-name">' f"{display_sku_name(selected_sku)}" f"</div>"), unsafe_allow_html=True)

st.markdown(
    (
        '<div class="product-meta"><span class="regime-pill">'
        + selected_regime
        + (f" · {REGIME_METHODS[selected_regime]}" if selected_regime in REGIME_METHODS else "")
        + "</span>"
        + (f"<span>{' · '.join(meta_bits)}</span>" if meta_bits else "")
        + "</div>"
    ),
    unsafe_allow_html=True
)


# ============================================================
# LOAD SELECTED SKU
# ============================================================

history = load_sku_rows("history", selected_sku)

if regime_requires_weekly:
    # Intermittent / Lumpy: frozen weekly actuals + weekly trajectory (never interpolated to daily).
    # Weekly period is exposed as `date` so the shared chart/metric code can reuse it.
    actuals = load_regime_actuals(selected_sku).rename(columns={"period": "date"})
    regime_history = load_regime_history(selected_sku)
    if not regime_history.empty:
        history = regime_history.rename(columns={"period": "date"})
        validate_regime_history(history)
        validate_actual_history(actuals, pd.DataFrame())
    else:
        validate_actual_history(actuals, history)
    validate_regime_actuals(actuals)
    trajectory = load_regime_trajectory(selected_sku)
    has_forecast = not trajectory.empty
    forecasts = pd.DataFrame()
    validate_regime_trajectory(trajectory)

else:
    actuals = load_sku_rows("actuals", selected_sku)
    validate_actual_history(actuals, history)
    trajectory = pd.DataFrame()
    has_forecast = str(selected_sku) in forecast_id_set
    forecasts = load_sku_rows("forecasts", selected_sku) if has_forecast else pd.DataFrame()
    if has_forecast:
        validate_forecast_data(forecasts, actuals)

chart_labels = {
    "Intermittent": dict(series_name="TSB forecast", range_name="7-day bootstrap demand range"),
    "Lumpy": dict(series_name="Historical policy level"),
}.get(selected_regime, {})


# ============================================================
# MAIN FORECAST / DEMAND AREA
# ============================================================

left, right = st.columns([3.45, 1], gap="large")


with left:
    st.markdown(
        (
            '<div class="section-title">'
            + (("Actual demand vs policy level" if selected_regime == "Lumpy" else "Actual demand vs forecast") if has_forecast else "Demand history")
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    has_chart_data = not history.empty or not actuals.empty or has_forecast
    if has_chart_data:
        fig = build_chart(
            history=history,
            actuals=actuals,
            forecasts=forecasts,
            display_days=display_days,
            weekly=(aggregation == "Weekly"),
            upper_quantile=selected_quantile,
            trajectory=trajectory if regime_requires_weekly and has_forecast else None,
            **chart_labels,
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
    else:
        st.info("No frozen demand rows are available " "for this SKU.")
    if not has_forecast:
        st.caption("Stored forecast trajectory unavailable for this SKU. No replacement forecast is generated.")


with right:
    st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
    def info_row(label, value):
        return f'<div class="info-label">{label}</div><div class="info-value">{value}</div>'
    faint = '<span style="font-weight:400;opacity:.65">'
    if has_forecast:
        method, source = "LightGBM Tweedie", "real"
        if regime_requires_weekly:
            method, source = trajectory["forecast_method"].iloc[0], trajectory["forecast_source"].iloc[0]
        rows_html = info_row("Forecast origin", f"{FORECAST_START:%b %d, %Y}")
        rows_html += info_row("Method", method + (f" {faint}({source})</span>" if source == "proxy" else ""))
        rows_html += info_row("Display window", f"{display_days} days · {aggregation.lower()}")
        if selected_regime != "Lumpy":
            range_label = "Bootstrap demand range" if selected_regime == "Intermittent" else "Planning range"
            rows_html += info_row(range_label, f"{selected_quantile.upper()} {faint}({uncertainty_label.split(' · ')[0].lower()})</span>")
        st.markdown('<div class="info-panel">' + rows_html + "</div>", unsafe_allow_html=True)
        st.caption("Historical context is shown before the forecast origin; the post-origin period is the evaluation period.")
        if selected_regime == "Intermittent":
            st.caption(
                "Weekly TSB forecast. The range shows frozen 7-day lead-time-demand bootstrap quantiles "
                "(q50 to the selected level), not weekly forecast intervals."
            )
        elif selected_regime == "Lumpy":
            st.caption("Weekly level from the frozen historical inventory policy (proxy), not a statistical forecast. " "No forecast uncertainty is stored.")
        elif aggregation == "Weekly":
            st.caption("In the weekly view, the planning range shows sums of daily scenario values, not weekly quantiles.")
    else:
        st.markdown(info_row("Forecast origin", f"{FORECAST_START:%b %d, %Y}") + info_row("Display", aggregation), unsafe_allow_html=True)
        st.caption("Stored forecast trajectory unavailable for this SKU.")


# ============================================================
# FORECAST EVALUATION
# ============================================================

st.markdown(
    '<div class="section-title">' + ("Policy-level comparison" if selected_regime == "Lumpy" else "Forecast evaluation") + "</div>",
    unsafe_allow_html=True,
)

if has_forecast:
    # Metrics use exactly the window shown in the chart (same FORECAST_START anchor and display window).
    if regime_requires_weekly:
        metrics = calculate_metrics(*weekly_metric_frames(actuals, trajectory, display_days))
    else:
        metrics = calculate_metrics(actuals, forecasts, end_date=evaluation_window_end(display_days, weekly=False))
    if metrics:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", fmt_num(metrics["mae"], 2), help=("Average absolute gap, in units, between " "the forecast and actual demand."))
        c2.metric("WAPE", fmt_pct(metrics["wape"], 1), help=("Total absolute error relative " "to total demand."))
        c3.metric(
            "Bias",
            fmt_pct(metrics["bias"], 1),
            help=(
                "Signed error relative "
                "to total demand. Positive means the forecast "
                "tends to be high; negative means it "
                "tends to be low."
            ),
        )
        if selected_regime == "Intermittent":
            st.caption(f"Weekly actual demand vs the weekly TSB forecast over {metrics['rows']} complete week(s) — the selected {display_days}-day window.")
        elif selected_regime == "Lumpy":
            st.caption(
                f"Weekly actual demand compared with the frozen historical inventory policy level over {metrics['rows']} complete week(s) "
                f"— the selected {display_days}-day window. Lumpy uses a policy-based planning level rather than a statistical forecast model, "
                "so these values are a policy-level comparison, not model accuracy."
            )
        else:
            st.caption(f"Computed on daily values over the first {metrics['rows']} days of the evaluation horizon — the selected {display_days}-day window.")
    else:
        st.info("Evaluation metrics are unavailable for this product in the selected window.")

else:
    st.info("Evaluation metrics are unavailable: the stored forecast trajectory is unavailable for this SKU.")


# ============================================================
# INVENTORY PLANNING
# ============================================================

st.markdown('<div class="section-title">' "Inventory planning" "</div>", unsafe_allow_html=True)

policy = load_inventory_sku(selected_sku)

target_level = normalize_service_level(selected_quantile)

policy_row = select_policy_row(policy, target_level)

signals = extract_policy_signals(policy_row)


if signals:
    signal_help = {
        "Lead-time demand": ("Expected demand during the frozen " "lead-time assumption."),
        "Safety buffer": ("Additional inventory above expected " "lead-time demand to provide protection " "against uncertainty."),
        "Reorder point": ("Inventory threshold used by the " "frozen policy to trigger replenishment."),
        "Order quantity": ("Planned replenishment quantity from " "the frozen policy."),
    }
    signal_columns = st.columns(len(signals))
    for column, (label, value) in zip(signal_columns, signals):
        with column:
            st.metric(label, value, help=signal_help.get(label))
    lead_time = policy_value(policy_row, ["lead_time_days", "lead_time"])
    review_period = policy_value(policy_row, ["review_period_weeks", "review_weeks"])
    validated = normalize_bool(policy_value(policy_row, ["validated_configuration", "validated"]))
    assumption_bits = []
    if lead_time is not None:
        assumption_bits.append(f"{fmt_num(lead_time, 0)}-day lead time")
    if review_period is not None:
        assumption_bits.append(f"{fmt_num(review_period, 0)}-week review")
    if validated is True:
        assumption_bits.append("q80 validated primary")
    elif validated is False:
        assumption_bits.append("scenario alternative")
    if assumption_bits:
        st.caption(" · ".join(assumption_bits))

else:
    if policy.empty:
        st.info("No frozen inventory policy is available for this SKU.")
    else:
        st.info("No exact frozen inventory-policy row " f"is available for " f"{selected_quantile.upper()} " "for this SKU.")


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

with st.expander("How the system works"):
    FLOW_STEPS = [('Historical demand', 'Daily sales history up to the 2015-02-01 forecast origin.'), ('Demand regimes', 'ADI and CV² classify each SKU: Smooth, Erratic, Intermittent or Lumpy.'), ('Regime-specific forecast', 'Each regime is routed to a suitable method rather than one model for all.'), ('Demand uncertainty', 'Bootstrap 7-day demand quantiles (Intermittent) and stored daily scenarios.'), ('Inventory policy', 'Service-level scenarios produce reorder point and order quantity.'), ('Future-period evaluation', 'Locked Fold 3 backtest: MAE, WAPE and Bias on unseen weeks.')]
    ROUTES = [('Smooth', '8,111 SKUs', 'Frequent demand, stable size', 'LightGBM Tweedie', 'Daily forecast trajectories'), ('Erratic', '906 SKUs', 'Frequent demand, highly variable size', 'LightGBM Tweedie', 'Daily forecast trajectories'), ('Intermittent', '17,303 SKUs', 'Infrequent demand, stable size', 'TSB (α = 0.5, β = 0.5)', 'Weekly walk-forward forecast'), ('Lumpy', '4,170 SKUs', 'Infrequent demand, highly variable size', 'Historical policy', 'Planning proxy, not an ML forecast')]
    st.markdown(
        '<div class="flow">'
        + "".join(
            f'<div class="flow-step"><div class="flow-num">{n}</div><div class="flow-title">{title}</div><div class="flow-desc">{desc}</div></div>'
            for n, (title, desc) in enumerate(FLOW_STEPS, start=1)
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Regime thresholds: ADI = 1.32 and CV² = 0.49, computed on training data. 30,490 SKUs in total.")
    st.markdown(
        '<div class="route-grid">'
        + "".join(
            f'<div class="route-card"><div class="route-regime">{regime}</div><div class="route-count">{count}</div>'
            f'<div class="route-when">{when}</div><div class="route-method">{method}<div class="route-note">{note}</div></div></div>'
            for regime, count, when, method, note in ROUTES
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Methodology details**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Time-series validation**")
        st.write(
            "Evaluation is chronological, never a random split: the past is used for training and model selection, "
            "the future for evaluation. Methods and parameters were chosen on earlier folds; "
            "Fold 3 is the final locked evaluation and was not retuned after seeing results."
        )
        st.markdown("**2. Demand regimes**")
        st.write(
            "Regimes are computed on training data from ADI (how often demand occurs) and CV² (how variable non-zero demand is), "
            "with thresholds ADI = 1.32 and CV² = 0.49. The 30,490 SKUs split into "
            "Smooth (8,111), Erratic (906), Intermittent (17,303) and Lumpy (4,170)."
        )
    with col2:
        st.markdown("**3. Regime-specific forecasting**")
        st.write(
            "Smooth and Erratic: LightGBM Tweedie, stored as daily forecast trajectories. "
            "Intermittent: TSB (α = 0.5, β = 0.5, locked), a causal weekly walk-forward forecast that models demand occurrence and size separately. "
            "Lumpy: a historical planning policy, not a predictive model, shown as a proxy level."
        )
        st.markdown("**4. Uncertainty**")
        st.write(
            "Intermittent uncertainty is a frozen moving-block bootstrap of 7-day lead-time demand (q50–q99). "
            "These are demand quantiles, not confidence intervals. Lumpy has no stored uncertainty. "
            "Smooth/Erratic weekly ranges are sums of daily scenario values, not weekly quantiles."
        )
    with col3:
        st.markdown("**5. Inventory planning**")
        st.write(
            "A separate frozen layer turns demand uncertainty and a service-level scenario into lead-time demand, safety buffer, "
            "reorder point and order quantity (7-day lead time, 1-week review). "
            "q80 is the validated primary scenario; other levels are scenario analyses."
        )
        st.markdown("**6. Evaluation**")
        st.write(
            "MAE, WAPE and Bias are computed over the selected display window. Smooth/Erratic use daily values; "
            "Intermittent and Lumpy use complete 7-day periods anchored at the 2015-02-01 forecast origin. "
            "For Lumpy this is a policy-level comparison, not model accuracy."
        )
    st.divider()
    st.markdown("**Data notes**")
    st.write(
        "This is a historical snapshot of frozen Fold 3 outputs covering February 1, 2015 through January 31, 2016. "
        "Daily LightGBM trajectories are stored for 8,863 Smooth/Erratic SKUs; weekly TSB and policy trajectories cover all "
        "21,473 Intermittent and Lumpy SKUs. The application does not retrain models, generate replacement forecasts, "
        "or run a new inventory simulation."
    )
    st.caption(
        "SHAP represents predictive contribution, not "
        "causality. Inventory and cost results are modeled "
        "historical scenarios, not realized savings."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="app-footer">'
    "Built by Zach Skiba · M5 demand forecasting and inventory-planning portfolio project<br>"
    "Data: M5 Forecasting competition (Walmart retail sales, via Kaggle) · Frozen historical snapshot, February 2015 – January 2016"
    "</div>",
    unsafe_allow_html=True,
)