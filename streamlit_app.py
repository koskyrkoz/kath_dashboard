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

# Improved variance score: penalize step-like series with few distinct changes
@st.cache_data
def fluctuation_ranking(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("product_gr")["price_mid"]
    n = g.size().rename("n_obs")
    q75, q25 = g.quantile(0.75), g.quantile(0.25)
    robust_std = (0.7413 * (q75 - q25)).rename("robust_std").fillna(0.0)
    weight = (1 - np.exp(-n / 180.0)).rename("weight")

    def change_ratio(series: pd.Series) -> float:
        x = series.dropna().values
        if len(x) <= 1:
            return 0.0
        med = float(np.nanmedian(x)) if len(x) else 0.0
        tol = max(0.01, 0.01 * med)  # ~1% step tolerance
        diffs = np.abs(np.diff(x))
        changes = int((diffs > tol).sum())
        return changes / max(len(x) - 1, 1)

    df_sorted = d.sort_values(["product_gr", "obs_date"])
    churn = df_sorted.groupby("product_gr")["price_mid"].apply(change_ratio).rename("change_ratio")
    churn_weight = (churn.clip(0, 1) ** 0.7).rename("churn_weight")

    base_score = (robust_std * weight).rename("fluctuation_score")
    n_max = max(float(n.max()), 1.0)
    obs_boost = (np.log1p(n) / np.log1p(n_max)).rename("obs_boost")
    variance_score = (base_score * obs_boost * churn_weight).rename("variance_score")

    out = pd.concat([n, robust_std, weight, base_score, obs_boost, churn, churn_weight, variance_score], axis=1).reset_index()
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

def plot_clustered_seasonal(ax, dd: pd.DataFrame, product_gr: str, years_on: set):
    x = dd[(dd["product_gr"] == product_gr) & (dd["obs_date"].dt.year.isin(list(years_on)))].copy()
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
        [m.isin([12,1,2]), m.isin([3,4,5]), m.isin([6,7,8]), m.isin([9,10,11])],
        ["Winter", "Spring", "Summer", "Autumn"], default="Unknown"
    ), index=df_in.index, name="season")
    out = df_in.copy()
    out["season"] = season
    return out

# ---------- Top Movers helpers ----------
def _period_bounds(last_date: pd.Timestamp, mode: str):
    if mode == "Week":
        p = last_date.to_period("W-MON")
        return p.start_time.normalize(), p.end_time.normalize()
    else:
        p = last_date.to_period("M")
        return p.start_time.normalize(), p.end_time.normalize()

def _prev_period_bounds(start: pd.Timestamp, mode: str):
    if mode == "Week":
        prev_end = start - pd.Timedelta(days=1)
        p = prev_end.to_period("W-MON")
        return p.start_time.normalize(), p.end_time.normalize()
    else:
        prev_end = (start - pd.Timedelta(days=1))
        p = prev_end.to_period("M")
        return p.start_time.normalize(), p.end_time.normalize()

def compute_top_movers(data: pd.DataFrame, mode: str):
    last_date = data["obs_date"].max().normalize()
    cur_start, cur_end = _period_bounds(last_date, mode)
    prev_start, prev_end = _prev_period_bounds(cur_start, mode)

    cur = (data[(data["obs_date"] >= cur_start) & (data["obs_date"] <= cur_end)]
           .groupby("product_gr")["price_mid"].mean().rename("cur_avg"))
    prev = (data[(data["obs_date"] >= prev_start) & (data["obs_date"] <= prev_end)]
           .groupby("product_gr")["price_mid"].mean().rename("prev_avg"))

    m = pd.concat([cur, prev], axis=1).dropna()
    m = m[m["prev_avg"] > 0]
    m["pct_change"] = (m["cur_avg"] - m["prev_avg"]) / m["prev_avg"] * 100.0

    risers = (m[m["pct_change"] > 0].sort_values("pct_change", ascending=False).head(5).reset_index())
    risers = risers[["product_gr", "pct_change", "cur_avg"]]
    droppers = (m[m["pct_change"] < 0].sort_values("pct_change", ascending=True).head(5).reset_index())
    droppers["pct_change"] = -droppers["pct_change"]
    droppers = droppers[["product_gr", "pct_change", "cur_avg"]]
    return droppers, risers, (cur_start.date(), cur_end.date()), (prev_start.date(), prev_end.date())

# ---------- styling helpers ----------
def _text_color_green_to_red(series: pd.Series):
    vmin, vmax = series.min(), series.max()
    rng = (vmax - vmin) if pd.notna(vmax) and pd.notna(vmin) else 0.0
    out = []
    for v in series:
        if rng == 0 or pd.isna(v): out.append("color: inherit")
        else:
            t = float((v - vmin) / rng)
            r = int(220 * t); g = int(153 * (1 - t))
            out.append(f"color: rgb({r},{g},0)")
    return out

def _text_color_red_to_green(series: pd.Series):
    vmin, vmax = series.min(), series.max()
    rng = (vmax - vmin) if pd.notna(vmax) and pd.notna(vmin) else 0.0
    out = []
    for v in series:
        if rng == 0 or pd.isna(v): out.append("color: inherit")
        else:
            t = float((v - vmin) / rng)          # 0→low, 1→high
            r = int(220 * (1 - t)); g = int(153 * t)
            out.append(f"color: rgb({r},{g},0)")  # high=green, low=red
    return out

# ================== APP ==================
df = load_data()

# Compute eligible products and sort the dropdown by variance score (DESC)
prods_eligible = eligible_products(df)
vr_global = fluctuation_ranking(df)
ranked = vr_global["product_gr"].tolist()
prods = [p for p in ranked if p in prods_eligible]  # ordered by variance score

if not prods:
    st.error("No products meet the minimum observation threshold."); st.stop()

# ---------- TOP SEGMENT ----------
st.markdown("## Top Movers")
mode = st.radio("Period", ["Month", "Week"], horizontal=True)
droppers, risers, cur_range, prev_range = compute_top_movers(df, mode)

top_cols = st.columns([4, 4, 3.5])

with top_cols[0]:
    st.markdown(f"**Biggest % drops ({mode.lower()})**")
    d_disp = droppers.rename(columns={"pct_change":"drop %","cur_avg":"avg price (€)"})
    styled = (d_disp.style
              .format({"drop %":"{:.1f}%","avg price (€)":"{:.3f}"})
              .apply(lambda s: ["color: green"]*len(s), subset=["product_gr"])
              .apply(_text_color_red_to_green, subset=["drop %"])
              .apply(_text_color_green_to_red, subset=["avg price (€)"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)

with top_cols[1]:
    st.markdown(f"**Biggest % rises ({mode.lower()})**")
    r_disp = risers.rename(columns={"pct_change":"rise %","cur_avg":"avg price (€)"})
    styled = (r_disp.style
              .format({"rise %":"{:.1f}%","avg price (€)":"{:.3f}"})
              .apply(lambda s: ["color: red"]*len(s), subset=["product_gr"])
              .apply(_text_color_red_to_green, subset=["rise %"])
              .apply(_text_color_green_to_red, subset=["avg price (€)"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)

with top_cols[2]:
    st.markdown("### Variance score")
    vr_small = vr_global[["product_gr", "variance_score"]].rename(columns={"variance_score":"score"}).head(12)
    smin, smax = float(vr_small["score"].min()), float(vr_small["score"].max())
    def _score_color(series: pd.Series):
        if smax == smin:
            return ["color: inherit"] * len(series)
        styles = []
        for v in series:
            t = (float(v) - smin) / (smax - smin)
            r, g = int(220 * (1 - t)), int(153 * t)
            styles.append(f"color: rgb({r},{g},0)")
        return styles
    styled_vs = (vr_small.style
                 .format({"score": "{:.4f}"})
                 .apply(_score_color, subset=["score"]))
    st.dataframe(styled_vs, use_container_width=True, hide_index=True, height=300)

st.caption(f"Comparing {mode.lower()} averages: current {cur_range[0]} → {cur_range[1]} vs previous {prev_range[0]} → {prev_range[1]}")

# -------- product selector (sorted by variance score)
if "product_select" not in st.session_state or st.session_state["product_select"] not in prods:
    st.session_state["product_select"] = prods[0]
product = st.selectbox("product_gr (sorted by variance score)", prods,
                       index=prods.index(st.session_state["product_select"]),
                       key="product_select")

# -------- Main plots row: [years] | [overlapped+forecast] | [clustered seasonal]
cols = st.columns([2, 7, 7])
with cols[0]:
    st.markdown("**Years**")
    years_on = {y for y, f in zip(YEARS, [
        st.checkbox("2021", True),
        st.checkbox("2022", True),
        st.checkbox("2023", True),
        st.checkbox("2024", True),
        st.checkbox("2025", True),
    ]) if f}

with cols[1]:
    fig1, ax1 = plt.subplots(figsize=(10.5, 5.8))
    plot_overlapped_with_forecast(ax1, df, product, years_on)
    st.pyplot(fig1, clear_figure=True)

with cols[2]:
    fig2, ax2 = plt.subplots(figsize=(10.5, 5.8))
    plot_clustered_seasonal(ax2, df, product, years_on)  # respects year toggles
    st.pyplot(fig2, clear_figure=True)

# ---- Counts by season + monthly + seasonal avg bar ----
st.markdown("---")
st.subheader("Counts by season")
season_cols = st.columns([5, 7])

with season_cols[0]:
    left_controls, table_col = st.columns([1, 4])
    with left_controls:
        st.markdown("**Years:**")
        years_on_season = {y for y, f in zip(YEARS, [
            st.checkbox("2021", True, key="s2021"),
            st.checkbox("2022", True, key="s2022"),
            st.checkbox("2023", True, key="s2023"),
            st.checkbox("2024", True, key="s2024"),
            st.checkbox("2025", True, key="s2025"),
        ]) if f}

    with table_col:
        dd = df[(df["product_gr"] == product) & (df["obs_date"].dt.year.isin(list(years_on_season)))]
        dd = add_season_column(dd)

        seasons = ["Winter", "Spring", "Summer", "Autumn"]
        tbl = dd.groupby("season").agg(count=("price_mid", "size"), price_mid_avg=("price_mid", "mean")).reindex(seasons).fillna(0)
        styled_season = (tbl.style
                         .format({"price_mid_avg": "{:.3f}", "count": "{:,.0f}"})
                         .apply(_text_color_green_to_red, subset=["price_mid_avg"]))
        st.dataframe(styled_season, use_container_width=True)

        if dd.empty:
            st.info("No monthly data for selected years.")
        else:
            dd["month_num"] = dd["obs_date"].dt.month
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            mon_tbl = dd.groupby("month_num").agg(count=("price_mid","size"), price_mid_avg=("price_mid","mean")).reindex(range(1,13)).reset_index()
            mon_tbl["month"] = mon_tbl["month_num"].map(month_names)
            mon_tbl = mon_tbl[["month","count","price_mid_avg"]]

            def _month_text_colors(s: pd.Series):
                season_color = {"Jan":"#ADD8E6","Feb":"#ADD8E6","Dec":"#ADD8E6",
                                "Mar":"#90EE90","Apr":"#90EE90","May":"#90EE90",
                                "Jun":"#FF7F7F","Jul":"#FF7F7F","Aug":"#FF7F7F",
                                "Sep":"#FFD580","Oct":"#FFD580","Nov":"#FFD580"}
                return [f"color: {season_color.get(v, 'inherit')}" for v in s]

            styled_month = (mon_tbl.style
                            .format({"price_mid_avg":"{:.3f}","count":"{:,.0f}"})
                            .apply(_month_text_colors, subset=["month"])
                            .apply(_text_color_green_to_red, subset=["price_mid_avg"]))
            st.dataframe(styled_month, use_container_width=True, height=420, hide_index=True)

with season_cols[1]:
    dd_plot = dd.copy()
    if dd_plot.empty:
        st.info("No data for selected years.")
    else:
        fig3, ax_price = plt.subplots(figsize=(10.5, 4.8))
        order = ["Winter", "Spring", "Summer", "Autumn"]
        dd_plot["season"] = pd.Categorical(dd_plot["season"], categories=order, ordered=True)
        avg_price = dd_plot.groupby("season")["price_mid"].mean().reindex(order).fillna(0)
        colors_map = {"Winter":"#ADD8E6","Spring":"#90EE90","Summer":"#FF7F7F","Autumn":"#FFD580"}
        colors = [colors_map[s] for s in order]
        idx = np.arange(len(order))
        ax_price.bar(idx, avg_price.values, color=colors, edgecolor="black", linewidth=1.0)
        ax_price.set_ylabel("€ / kg"); ax_price.set_xlabel("season")
        ax_price.set_xticks(idx); ax_price.set_xticklabels(order, rotation=0)
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

