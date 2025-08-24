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
    q75, q25 = g.quantile(0.75), g.quantile(0.25)
    robust_std = (0.7413 * (q75 - q25)).rename("robust_std").fillna(0.0)
    weight = (1 - np.exp(-n / 180.0)).rename("weight")
    base_score = (robust_std * weight).rename("fluctuation_score")
    n_max = max(float(n.max()), 1.0)
    obs_boost = (np.log1p(n) / np.log1p(n_max)).rename("obs_boost")
    variance_score = (base_score * obs_boost).rename("variance_score")
    out = pd.concat([n, robust_std, weight, base_score, obs_boost, variance_score], axis=1).reset_index()
    return out.sort_values("variance_score", ascending=False)

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

def _text_color_green_to_red(series: pd.Series):
    vmin, vmax = series.min(), series.max()
    rng = (vmax - vmin) if pd.notna(vmax) and pd.notna(vmin) else 0.0
    styles = []
    for v in series:
        if rng == 0 or pd.isna(v):
            styles.append("color: inherit")
        else:
            t = float((v - vmin) / rng)
            r = int(220 * t)
            g = int(153 * (1 - t))
            styles.append(f"color: rgb({r},{g},0)")
    return styles

# -------- App --------
df = load_data()
prods = eligible_products(df)
if not prods:
    st.error("No products meet the minimum observation threshold."); st.stop()

# Selector
product = st.selectbox("product_gr", prods, index=0)

# Main layout: [years checkboxes] | [avg price per year] | [clustered seasonal] | [right panel table]
cols = st.columns([2, 7, 7, 3])

# Left: years checkboxes
with cols[0]:
    st.markdown("**Years**")
    years_on = set()
    y2021 = st.checkbox("2021", True)
    y2022 = st.checkbox("2022", True)
    y2023 = st.checkbox("2023", True)
    y2024 = st.checkbox("2024", True)
    y2025 = st.checkbox("2025", True)
    for y, flag in zip([2021, 2022, 2023, 2024, 2025], [y2021, y2022, y2023, y2024, y2025]):
        if flag: years_on.add(y)

# Middle-left: overlapped plot
with cols[1]:
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.8))
    plot_overlapped_with_forecast(ax1, df, product, years_on)
    st.pyplot(fig1, clear_figure=True)

# Middle-right: clustered plot
with cols[2]:
    fig2, ax2 = plt.subplots(figsize=(10.5, 5.8))
    plot_clustered_seasonal(ax2, df, product)
    st.pyplot(fig2, clear_figure=True)

# Right panel: compact with combined variance score
with cols[3]:
    st.markdown("### Product Occurrences (obs) & Variance (score)")
    vr = fluctuation_ranking(df)
    vr_small = (
        vr[["product_gr", "variance_score"]]
          .rename(columns={"product_gr": "product_gr", "variance_score": "variance score"})
    )
    st.dataframe(vr_small, use_container_width=True, height=380, hide_index=True)

# ---- Counts by season + monthly table + seasonal average price bar plot ----
st.markdown("---")
st.subheader("Counts by season")

season_cols = st.columns([5, 7])

# Left side: Years controls (left) and tables (right)
with season_cols[0]:
    left_controls, table_col = st.columns([1, 4])
    with left_controls:
        st.markdown("**Years:**")
        ys1 = st.checkbox("2021", True, key="s2021")
        ys2 = st.checkbox("2022", True, key="s2022")
        ys3 = st.checkbox("2023", True, key="s2023")
        ys4 = st.checkbox("2024", True, key="s2024")
        ys5 = st.checkbox("2025", True, key="s2025")
        years_on_season = {y for y, f in zip(YEARS, [ys1, ys2, ys3, ys4, ys5]) if f}

    with table_col:
        dd = df[(df["product_gr"] == product) & (df["obs_date"].dt.year.isin(list(years_on_season)))]
        dd = add_season_column(dd)

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        tbl = dd.groupby("season").agg(
            count=("price_mid", "size"),
            price_mid_avg=("price_mid", "mean"),
        ).reindex(seasons).fillna(0)

        styled_season = (tbl.style
                         .format({"price_mid_avg": "{:.3f}", "count": "{:,.0f}"})
                         .apply(_text_color_green_to_red, subset=["price_mid_avg"]))
        st.dataframe(styled_season, use_container_width=True)

        # Monthly table below
        if dd.empty:
            st.info("No monthly data for selected years.")
        else:
            dd["month_num"] = dd["obs_date"].dt.month
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            mon_tbl = dd.groupby("month_num").agg(
                count=("price_mid", "size"),
                price_mid_avg=("price_mid", "mean"),
            ).reindex(range(1,13))
            mon_tbl.index = [month_names[m] for m in mon_tbl.index]
            styled_month = (mon_tbl.style
                            .format({"price_mid_avg": "{:.3f}", "count": "{:,.0f}"})
                            .apply(_text_color_green_to_red, subset=["price_mid_avg"]))
            st.dataframe(styled_month, use_container_width=True)

# Right side: Seasonal average price bar plot with custom colors + outline
with season_cols[1]:
    dd_plot = dd.copy()
    if dd_plot.empty:
        st.info("No data for selected years.")
    else:
        fig3, ax_price = plt.subplots(figsize=(10.5, 4.8))
        order = ["Winter", "Spring", "Summer", "Autumn"]
        dd_plot["season"] = pd.Categorical(dd_plot["season"], categories=order, ordered=True)

        avg_price = dd_plot.groupby("season")["price_mid"].mean().reindex(order).fillna(0)

        colors_map = {
            "Winter": "#ADD8E6",   # light blue
            "Spring": "#90EE90",   # light green
            "Summer": "#FF7F7F",   # light red
            "Autumn": "#FFD580",   # light orange
        }
        colors = [colors_map[s] for s in order]
        idx = np.arange(len(order))

        ax_price.bar(idx, avg_price.values, color=colors, edgecolor="black", linewidth=1.0)
        ax_price.set_ylabel("€ / kg")
        ax_price.set_xlabel("season")
        ax_price.set_xticks(idx)
        ax_price.set_xticklabels(order, rotation=0)
        ax_price.set_title(f"Seasonal average price — {product}")

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
