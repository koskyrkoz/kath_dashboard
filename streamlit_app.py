# streamlit_app.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

st.set_page_config(page_title="Fruit & Veg Prices", layout="wide")

MIN_OBS = 180
YEARS = [2021, 2022, 2023, 2024, 2025]
# file_path = r"C:\Users\kosti\Documents\fruitveg-prices\data\data_processed\daily_prices.csv"
file_path = "daily_prices.csv"

@st.cache_data
def load_data():
    # Tries DuckDB, then Parquet, then CSV
    try:
        import duckdb
        db_path = os.getenv("KATH_DUCKDB", "kath.duckdb")
        con = duckdb.connect(db_path, read_only=True)
        df = con.execute("SELECT * FROM daily_prices").df()
    except Exception:
        try:
            df = pd.read_parquet(os.getenv("KATH_PARQUET", "daily_prices.parquet"))
        except Exception:
            df = pd.read_csv(os.getenv("KATH_CSV", file_path))

    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df = df.drop(columns=["price_avg", "category"], errors="ignore")
    df = df.dropna(subset=["obs_date", "price_mid"])
    return df

def _to_2000(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).apply(lambda x: x.replace(year=2000))

def _fourier_features(dates: pd.Series, k_max=3) -> np.ndarray:
    s = pd.to_datetime(pd.Series(dates))
    doy = s.dt.dayofyear.to_numpy().reshape(-1, 1)
    X = [np.ones_like(doy, dtype=float)]
    for k in range(1, k_max + 1):
        X.append(np.sin(2 * np.pi * k * doy / 365.25))
        X.append(np.cos(2 * np.pi * k * doy / 365.25))
    return np.hstack(X)

@st.cache_data
def eligible_products(d: pd.DataFrame) -> list:
    cnt = d[(d["obs_date"].dt.year.isin(YEARS))].groupby("product_gr")["price_mid"].size()
    return sorted(cnt[cnt >= MIN_OBS].index.tolist())

def plot_overlapped_with_forecast(ax, dd: pd.DataFrame, product_gr: str):
    dsel = dd[(dd["product_gr"] == product_gr) & (dd["obs_date"].dt.year.isin(YEARS))].copy()
    if dsel.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return

    # Train a simple seasonal ridge model on all available points
    last_actual = dsel["obs_date"].max()
    X_train = _fourier_features(dsel["obs_date"])
    y_train = dsel["price_mid"].to_numpy()
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0))
    model.fit(X_train, y_train)

    # Future dates for the rest of 2025
    last_2025 = dsel.loc[dsel["obs_date"].dt.year == 2025, "obs_date"].max()
    start_future = pd.Timestamp("2025-01-01") if pd.isna(last_2025) else (last_2025 + pd.Timedelta(days=1))
    end_future = pd.Timestamp("2025-12-31")
    future_2025 = pd.date_range(start=start_future, end=end_future, freq="D") if start_future <= end_future else pd.DatetimeIndex([])
    X_future = _fourier_features(pd.Series(future_2025)) if len(future_2025) else np.empty((0,1))
    y_future = model.predict(X_future) if len(future_2025) else np.array([])

    # Overlapped lines by year
    for y in YEARS:
        seg = dsel[dsel["obs_date"].dt.year == y].sort_values("obs_date")
        if not seg.empty:
            ax.plot(_to_2000(seg["obs_date"]), seg["price_mid"], label=str(y))

    # Dotted extension for 2025 forecast in the same color as 2025 actuals
    if len(future_2025):
        color_2025 = None
        for line in ax.get_lines():
            if line.get_label() == "2025":
                color_2025 = line.get_color()
                break
        ax.plot(_to_2000(pd.Series(future_2025)), y_future, linestyle="--", label="2025 forecast", color=color_2025)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    ax.set_xlabel("calendar day")
    ax.set_ylabel("price_mid")
    ax.set_title(f"Overlapped years (2021–2025) with 2025 dotted forecast — {product_gr}")
    ax.legend()

def plot_clustered_seasonal(ax, dd: pd.DataFrame, product_gr: str):
    x = dd[(dd["product_gr"] == product_gr) & (dd["obs_date"].dt.year.isin(YEARS))].copy()
    if x.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    x["year"] = x["obs_date"].dt.year
    x["week"] = x["obs_date"].dt.isocalendar().week.astype(int)
    x["week_start"] = x["obs_date"] - pd.to_timedelta(x["obs_date"].dt.weekday, unit="D")
    wk = x.groupby(["year", "week"])["price_mid"].median().reset_index()
    M = wk.pivot_table(index="year", columns="week", values="price_mid", aggfunc="mean")
    M = M.reindex(columns=range(1, 54))
    M = M.apply(lambda r: r.interpolate(limit_direction="both"), axis=1)
    M = M.apply(lambda r: r.fillna(r.mean()), axis=1)

    if len(M) >= 2:
        X = ((M.sub(M.mean(axis=1), axis=0))
             .div(M.std(axis=1).replace(0, np.nan), axis=0)
             .fillna(0))
        k = min(2, len(M))
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        dom = pd.Series(labels).value_counts().idxmax()
        centroid = M[labels == dom].mean(axis=0)
    else:
        centroid = M.mean(axis=0)

    base = pd.to_datetime("2000-01-03")  # Monday
    dates = [base + pd.Timedelta(weeks=w - 1) for w in centroid.index]
    y = pd.Series(centroid.values).rolling(3, center=True).mean().bfill().ffill()

    ax.plot(dates, y.values, label="cluster centroid")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    ax.set_xlabel("calendar week")
    ax.set_ylabel("price_mid")
    ax.set_title(f"Clustered seasonal profile (weekly) — {product_gr}")
    ax.legend()

# ---------- App ----------
df = load_data()
prods = eligible_products(df)
if not prods:
    st.error("No products meet the minimum observation threshold."); st.stop()

product = st.selectbox("product_gr", prods, index=0)

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    plot_overlapped_with_forecast(ax1, df, product)
    st.pyplot(fig1, clear_figure=True)

with col2:
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    plot_clustered_seasonal(ax2, df, product)
    st.pyplot(fig2, clear_figure=True)

# Bottom summary
dd = df[df["product_gr"] == product]
years_present = sorted(dd["obs_date"].dt.year.unique().tolist())
counts_by_year = dd["obs_date"].dt.year.value_counts().sort_index()
st.markdown("---")
st.subheader("Data availability")
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(dd):,}")
c2.metric("Years covered", f"{years_present[0]}–{years_present[-1]}" if years_present else "—")
c3.metric("Unique years", f"{len(years_present)}")
st.write("Counts by year:", counts_by_year.to_frame("n_obs").T)
st.caption(f"Date range: {dd['obs_date'].min().date() if not dd.empty else '—'} → {dd['obs_date'].max().date() if not dd.empty else '—'}")

