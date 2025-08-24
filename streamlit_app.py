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

@st.cache_data
def load_data():
    try:
        import duckdb
        con = duckdb.connect(os.getenv("KATH_DUCKDB", "kath.duckdb"), read_only=True)
        df = con.execute("SELECT * FROM daily_prices").df()
    except Exception:
        try:
            df = pd.read_parquet(os.getenv("KATH_PARQUET", "daily_prices.parquet"))
        except Exception:
            df = pd.read_csv(os.getenv("KATH_CSV", "daily_prices.csv"))
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df = df.drop(columns=["price_avg", "category"], errors="ignore")
    return df.dropna(subset=["obs_date", "price_mid"])

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

def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

@st.cache_data
def eligible_products(d: pd.DataFrame) -> list:
    cnt = d[d["obs_date"].dt.year.isin(YEARS)].groupby("product_gr")["price_mid"].size()
    return sorted(cnt[cnt >= MIN_OBS].index.tolist())

@st.cache_data
def fluctuation_ranking(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("product_gr")["price_mid"]
    n = g.size().rename("n_obs")
    q75 = g.quantile(0.75)
    q25 = g.quantile(0.25)
    robust_std = (0.7413 * (q75 - q25)).rename("robust_std").fillna(0.0)
    w = (1 - np.exp(-n / 180.0)).rename("weight")
    score = (robust_std * w).rename("fluctuation_score")
    out = pd.concat([n, robust_std, w, score], axis=1).reset_index()
    return out.sort_values("fluctuation_score", ascending=False)

def plot_overlapped_with_forecast(ax, dd: pd.DataFrame, product_gr: str, years_on: set):
    dsel = dd[(dd["product_gr"] == product_gr) & (dd["obs_date"].dt.year.isin(YEARS))].copy()
    if dsel.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return

    X_train = _fourier_features(dsel["obs_date"])
    y_train = dsel["price_mid"].to_numpy()
    model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0)).fit(X_train, y_train)

    last_2025 = dsel.loc[dsel["obs_date"].dt.year == 2025, "obs_date"].max()
    start_future = pd.Timestamp("2025-01-01") if pd.isna(last_2025) else (last_2025 + pd.Timedelta(days=1))
    end_future = pd.Timestamp("2025-12-31")
    future_2025 = pd.date_range(start=start_future, end=end_future, freq="D") if start_future <= end_future else pd.DatetimeIndex([])
    X_future = _fourier_features(pd.Series(future_2025)) if len(future_2025) else np.empty((0,1))
    y_future = model.predict(X_future) if len(future_2025) else np.array([])

    for y in YEARS:
        if y not in years_on: 
            continue
        seg = dsel[dsel["obs_date"].dt.year == y].sort_values("obs_date")
        if not seg.empty:
            ax.plot(_to_2000(seg["obs_date"]), seg["price_mid"], label=str(y))

    if 2025 in years_on and len(future_2025):
        c25 = None
        for line in ax.get_lines():
            if line.get_label() == "2025":
                c25 = line.get_color(); break
        ax.plot(_to_2000(pd.Series(future_2025)), y_future, linestyle="--", label="2025 forecast", color=c25)

    _month_axis(ax)
    ax.set_xlabel("month")
    ax.set_ylabel("€ / kg")
    ax.set_title(f"Average price per year – {product_gr}")
    ax.legend()

def plot_clustered_seasonal(ax, dd: pd.DataFrame, product_gr: str):
    x = dd[(dd["product_gr"] == product_gr) & (dd["obs_date"].dt.year.isin(YEARS))].copy()
    if x.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    x["year"] = x["obs_date"].dt.year
    x["week"] = x["obs_date"].dt.isocalendar().week.astype(int)
    wk = x.groupby(["year", "week"])["price_mid"].median().reset_index()
    M = wk.pivot_table(index="year", columns="week", values="price_mid", aggfunc="mean").reindex(columns=range(1,54))
    M = M.apply(lambda r: r.interpolate(limit_direction="both"), axis=1).apply(lambda r: r.fillna(r.mean()), axis=1)

    if len(M) >= 2:
        Xn = ((M.sub(M.mean(axis=1), axis=0)).div(M.std(axis=1).replace(0, np.nan), axis=0)).fillna(0)
        labels = KMeans(n_clusters=min(2, len(M)), n_init=10, random_state=0).fit_predict(Xn)
        centroid = M[labels == pd.Series(labels).value_counts().idxmax()].mean(axis=0)
    else:
        centroid = M.mean(axis=0)

    base = pd.to_datetime("2000-01-03")
    dates = [base + pd.Timedelta(weeks=w-1) for w in centroid.index]
    y = pd.Series(centroid.values).rolling(3, center=True).mean().bfill().ffill()
    ax.plot(dates, y.values, label="cluster centroid")
    _month_axis(ax)
    ax.set_xlabel("month")
    ax.set_ylabel("€ / kg")
    ax.set_title(f"Clustered seasonal profile – {product_gr}")
    ax.legend()

def add_season_column(df_in: pd.DataFrame) -> pd.DataFrame:
    m = df_in["obs_date"].dt.month
    season = pd.Series(np.select(
        [
            m.isin([12,1,2]),
            m.isin([3,4,5]),
            m.isin([6,7,8]),
            m.isin([9,10,11])
        ],
        ["Winter", "Spring", "Summer", "Autumn"],
        default="Unknown"
    ), index=df_in.index, name="season")
    out = df_in.copy()
    out["season"] = season
    return out

# -------- App --------
df = load_data()
prods = eligible_products(df)
if not prods:
    st.error("No products meet the minimum observation threshold."); st.stop()

top_controls = st.columns([3, 1, 3])
with top_controls[0]:
    product = st.selectbox("product_gr", prods, index=0)

# Main row: left (year lines + checkboxes), middle (clustered), right (variance list)
body = st.columns([5, 4, 3])

left_plot_col, year_toggle_col = body[0].columns([6, 1])
years_on = set()
with year_toggle_col:
    st.write("**Years**")
    y2021 = st.checkbox("2021", True)
    y2022 = st.checkbox("2022", True)
    y2023 = st.checkbox("2023", True)
    y2024 = st.checkbox("2024", True)
    y2025 = st.checkbox("2025", True)
    for y, flag in zip([2021, 2022, 2023, 2024, 2025], [y2021, y2022, y2023, y2024, y2025]):
        if flag: years_on.add(y)

with left_plot_col:
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    plot_overlapped_with_forecast(ax1, df, product, years_on)
    st.pyplot(fig1, clear_figure=True)

with body[1]:
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    plot_clustered_seasonal(ax2, df, product)
    st.pyplot(fig2, clear_figure=True)

# ==== RIGHT PANEL: renamed + columns + hide index ====
with body[2]:
    st.subheader("Product Occurrences (obs) & Variance (score)")
    vr = fluctuation_ranking(df)

    # map product_gr -> representative product_en
    map_en = (df.groupby("product_gr")["product_en"]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))
    vr = vr.merge(map_en.rename("product_en"), on="product_gr", how="left")

    vr_small = (vr[["product_en", "n_obs", "fluctuation_score"]]
                  .rename(columns={"n_obs": "obs", "fluctuation_score": "score"}))
    st.dataframe(vr_small, use_container_width=True, height=500, hide_index=True)

# ---- Counts by season + bar plot ----
st.markdown("---")
st.subheader("Counts by season")

season_cols = st.columns([3, 5])
with season_cols[0]:
    st.write("**Years (season section)**")
    ys1 = st.checkbox("2021", True, key="s2021")
    ys2 = st.checkbox("2022", True, key="s2022")
    ys3 = st.checkbox("2023", True, key="s2023")
    ys4 = st.checkbox("2024", True, key="s2024")
    ys5 = st.checkbox("2025", True, key="s2025")
    years_on_season = {y for y, f in zip(YEARS, [ys1, ys2, ys3, ys4, ys5]) if f}

    dd = df[(df["product_gr"] == product) & (df["obs_date"].dt.year.isin(list(years_on_season)))]
    dd = add_season_column(dd)
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    tbl = dd.groupby("season").agg(
        count=("price_mid", "size"),
        price_mid_avg=("price_mid", "mean"),
    ).reindex(seasons).fillna(0)

    styled = (tbl.style
              .format({"price_mid_avg": "{:.3f}", "count": "{:,.0f}"})
              # counts: green high → red low
              .background_gradient(subset=["count"], cmap="RdYlGn")
              # prices: green low → red high
              .background_gradient(subset=["price_mid_avg"], cmap="RdYlGn_r"))
    st.dataframe(styled, use_container_width=True)

with season_cols[1]:
    if dd.empty:
        st.info("No data for selected years.")
    else:
        fig3, ax_price = plt.subplots(figsize=(9, 4.5))
        order = ["Winter", "Spring", "Summer", "Autumn"]
        dd["season"] = pd.Categorical(dd["season"], categories=order, ordered=True)

        avg_price = dd.groupby("season")["price_mid"].mean().reindex(order).fillna(0)
        counts = dd["season"].value_counts().reindex(order).fillna(0).astype(int)

        idx = np.arange(len(order))
        width = 0.4

        # Blue = price (left y-axis)
        ax_price.bar(idx - width/2, avg_price.values, width=width, color="blue", label="avg price (€ / kg)")
        ax_price.set_ylabel("€ / kg")
        ax_price.set_xlabel("season")
        ax_price.set_xticks(idx)
        ax_price.set_xticklabels(order, rotation=0)

        # Green = occurrences (right y-axis)
        ax_cnt = ax_price.twinx()
        ax_cnt.bar(idx + width/2, counts.values, width=width, color="green", label="occurrences")
        ax_cnt.set_ylabel("count")

        h1, l1 = ax_price.get_legend_handles_labels()
        h2, l2 = ax_cnt.get_legend_handles_labels()
        ax_price.legend(h1 + h2, l1 + l2, loc="upper right")

        ax_price.set_title(f"Seasonal bars — {product}")
        st.pyplot(fig3, clear_figure=True)

# Bottom summary
dd_all = df[df["product_gr"] == product]
years_present = sorted(dd_all["obs_date"].dt.year.unique().tolist())
counts_by_year = dd_all["obs_date"].dt.year.value_counts().sort_index()
st.markdown("---")
st.subheader("Data availability")
c1, c2, c3 = st.columns(3)
c1.metric("Observations", f"{len(dd_all):,}")
c2.metric("Years covered", f"{years_present[0]}–{years_present[-1]}" if years_present else "—")
c3.metric("Unique years", f"{len(years_present)}")
st.write("Counts by year:", counts_by_year.to_frame("n_obs").T)
st.caption(f"Date range: {dd_all['obs_date'].min().date() if not dd_all.empty else '—'} → {dd_all['obs_date'].max().date() if not dd_all.empty else '—'}")
