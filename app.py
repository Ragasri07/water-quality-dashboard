# ============================================================
# Water Quality Management & Rainfall Dashboard (Final Project)
"""
Clean, modular Streamlit app rebuilt from scratch.

Features implemented here:
- Robust data loading from `data/water_quality.csv` and `data/rainfall.csv` (daily merge)
- Preprocessing: numeric coercion, outlier capping, median imputation
- WQI calculation with per-parameter contributions
- Nykaa-inspired hero UI + product-style site cards
- Key visualizations: KPI tiles, time-series with rolling mean, stacked WQI contributions, Spearman correlations with pairwise counts
- Simple ML demo (RandomForest) with feature alignment and sample input
- Data export

To run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
from datetime import timedelta
import math

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    seasonal_decompose = None
from scipy import stats
import altair as alt
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="Water Quality Management", page_icon="💧", layout="wide")


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
WATER_FILE = os.path.join(DATA_DIR, "water_quality.csv")
RAIN_FILE = os.path.join(DATA_DIR, "rainfall.csv")


# ----------------------------
# Minimal CSS (Nykaa-like cards)
# ----------------------------
st.markdown(
        """
        <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

        :root{
            --bg-overlay: rgba(6,10,26,0.55);
            --card-bg: rgba(255,255,255,0.78);
            --glass-border: rgba(255,255,255,0.08);
            --accent-1: linear-gradient(90deg,#667eea 0%, #764ba2 100%);
            --muted: #64748b;
        }

        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            background-image: linear-gradient(120deg, rgba(6,10,26,0.55), rgba(10,25,47,0.55)), url('https://images.unsplash.com/photo-1505483531331-3a4f0b6b1a3a?auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            color: #0f172a;
        }

        /* Make Streamlit main container look like a centered website card */
        .main .block-container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding: 16px 24px 24px 24px;
            background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
            box-shadow: 0 10px 40px rgba(2,6,23,0.35);
            border-radius: 14px;
            border: 1px solid var(--glass-border);
            backdrop-filter: blur(6px) saturate(120%);
        }

        .stSidebar { font-family: 'Inter', sans-serif; }

        .hero {
            background: var(--accent-1);
            color: white;
            padding: 28px 28px;
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 12px 40px rgba(18,25,50,0.28);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }

        .hero h1 { margin: 0; font-size: 2.4rem; letter-spacing: -0.02em; }
        .hero p { margin: 4px 0 0 0; opacity: 0.95; font-size:1.05rem; }

        .kpi { background: var(--card-bg); border-radius: 12px; padding: 18px; box-shadow: 0 8px 30px rgba(2,6,23,0.08); border: 1px solid var(--glass-border); }

        .card-compact { padding: 12px; border-radius: 10px; background: rgba(255,255,255,0.9); box-shadow: 0 6px 18px rgba(2,6,23,0.06); }

        .site-image { width:100%; height:140px; object-fit:cover; border-radius:10px; }

        .wqi-badge { padding:8px 12px; border-radius:999px; color:white; font-weight:800; font-size:0.95rem; }
        .wqi-excellent { background:#10b981 }
        .wqi-good { background:#06b6d4 }
        .wqi-poor { background:#f59e0b }
        .wqi-very-poor { background:#ef4444 }

        /* Tabs styling - blend with header */
        .stTabs [role="tab"] { font-weight:600; font-size:14px; padding:8px 16px; }
        .stTabs [data-baseweb="tab-list"] { gap:4px; margin-top:-4px; }
        .stTabs [data-baseweb="tab-panel"] { padding-top:16px; }

        /* Modern Dashboard Styling */
        
        /* Remove default Streamlit padding */
        .main .block-container {
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
            padding: 0 !important;
            background: transparent;
            box-shadow: none;
            border-radius: 0;
            border: none;
        }
        
        /* Animated gradient header */
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
            background-size: 200% 200%;
            animation: gradientShift 8s ease infinite;
            padding: 40px 40px;
            margin: -70px -100px 40px -100px;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating particles effect */
        .dashboard-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(255,255,255,0.1) 0%, transparent 50%);
            animation: float 6s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 8px;
            position: relative;
            z-index: 1;
        }
        
        .header-text h1 {
            margin: 0;
            font-size: 48px;
            font-weight: 800;
            color: white;
            letter-spacing: -1.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
            animation: fadeInDown 0.8s ease-out;
        }
        
        .header-text p {
            margin: 0;
            font-size: 16px;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 400;
            letter-spacing: 0.5px;
            animation: fadeInUp 0.8s ease-out;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .hero { flex-direction: column; text-align: center; }
            .main .block-container { padding: 20px; }
            .hero h1 { font-size: 1.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
)


# ----------------------------
# Utilities: loading & preprocessing
# ----------------------------

@st.cache_data
def load_and_process(water_path=WATER_FILE, rain_path=RAIN_FILE, use_synthetic: bool = True):
    # Load
    if not os.path.exists(water_path):
        raise FileNotFoundError("water_quality.csv not found in data/")
    water = pd.read_csv(water_path)

    # Coerce timestamps
    # Accept common names: Timestamp, timestamp, Date, date_time
    ts_cols = [c for c in water.columns if c.lower() in ("timestamp", "date", "datetime", "date_time")]
    if ts_cols:
        water["Timestamp"] = pd.to_datetime(water[ts_cols[0]], errors="coerce")
    else:
        # try to find first column resembling a date
        for c in water.columns:
            try:
                tmp = pd.to_datetime(water[c], errors="coerce")
                if tmp.notna().sum() > 5:
                    water["Timestamp"] = tmp
                    break
            except Exception:
                continue

    water = water.loc[~water.get("Timestamp").isna()].copy()

    # normalize column names (strip)
    water.columns = [c.strip() for c in water.columns]

    # load rainfall if present and merge daily
    if os.path.exists(rain_path):
        rain = pd.read_csv(rain_path)
        # try parse date
        if "date" in rain.columns:
            rain["date"] = pd.to_datetime(rain["date"], errors="coerce")
        else:
            # try first col
            rain.iloc[:, 0] = pd.to_datetime(rain.iloc[:, 0], errors="coerce")
            rain.columns.values[0] = "date"
        rain["date"] = rain["date"].dt.floor("D")
        rain_daily = rain.groupby("date").agg({
            c: ("sum" if np.issubdtype(rain[c].dtype, np.number) else "first") for c in rain.columns if c != "date"
        }).reset_index()
        # If rainfall dates do not overlap water dates, generate synthetic daily
        try:
            water_min = pd.to_datetime(water.iloc[:, 0], errors='coerce').min() if 'Timestamp' not in water.columns else water['Timestamp'].min()
            water_max = pd.to_datetime(water.iloc[:, 0], errors='coerce').max() if 'Timestamp' not in water.columns else water['Timestamp'].max()
            if pd.isna(water_min) or pd.isna(water_max):
                # fallback to Timestamp parse if available
                water_min = water['Timestamp'].min()
                water_max = water['Timestamp'].max()

            # simple overlap check
            if rain_daily.empty:
                need_synth = True
            else:
                rain_min = rain_daily['date'].min()
                rain_max = rain_daily['date'].max()
                need_synth = (rain_max < water_min) or (rain_min > water_max)

            if need_synth and use_synthetic:
                # Build reproducible synthetic rainfall + temperature for each day in water range
                rng = np.random.RandomState(42)
                date_range = pd.date_range(start=water_min.floor('D'), end=water_max.floor('D'), freq='D')

                # Estimate monthly (seasonal) statistics from raw rain file if present
                rain_stats = None
                try:
                    rain_tmp = rain.copy()
                    # find candidate rainfall and temp columns
                    rain_col_candidates = [c for c in rain_tmp.columns if 'rain' in c.lower()]
                    temp_col_candidates = [c for c in rain_tmp.columns if 'temp' in c.lower()]

                    if 'date' in rain_tmp.columns:
                        rain_tmp['month'] = pd.to_datetime(rain_tmp['date'], errors='coerce').dt.month
                    else:
                        rain_tmp['month'] = pd.to_datetime(rain_tmp.iloc[:, 0], errors='coerce').dt.month

                    # monthly stats for rainfall
                    if rain_col_candidates:
                        rcol = rain_col_candidates[0]
                        rain_tmp[rcol] = pd.to_numeric(rain_tmp[rcol], errors='coerce')
                        rain_stats = rain_tmp.groupby('month')[rcol].agg(['mean', 'std']).to_dict(orient='index')
                    else:
                        rain_stats = {}

                    # monthly stats for temperature
                    if temp_col_candidates:
                        tcol = temp_col_candidates[0]
                        rain_tmp[tcol] = pd.to_numeric(rain_tmp[tcol], errors='coerce')
                        temp_stats = rain_tmp.groupby('month')[tcol].agg(['mean', 'std']).to_dict(orient='index')
                    else:
                        temp_stats = {}
                except Exception:
                    rain_stats = {}
                    temp_stats = {}

                # Default fallbacks if no stats present
                overall_mean_r = None
                overall_std_r = None
                try:
                    if rain_col_candidates:
                        overall_mean_r = pd.to_numeric(rain[rcol], errors='coerce').mean()
                        overall_std_r = pd.to_numeric(rain[rcol], errors='coerce').std()
                except Exception:
                    overall_mean_r, overall_std_r = 1.0, 1.0
                if overall_mean_r is None or np.isnan(overall_mean_r):
                    overall_mean_r, overall_std_r = 1.0, 1.0

                overall_mean_t = None
                overall_std_t = None
                try:
                    if temp_col_candidates:
                        overall_mean_t = pd.to_numeric(rain[tcol], errors='coerce').mean()
                        overall_std_t = pd.to_numeric(rain[tcol], errors='coerce').std()
                except Exception:
                    overall_mean_t, overall_std_t = 25.0, 5.0
                if overall_mean_t is None or np.isnan(overall_mean_t):
                    overall_mean_t, overall_std_t = 25.0, 5.0

                # Generate per-day synthetic values using month-specific stats when available
                synth_rows = []
                for d in date_range:
                    m = int(d.month)
                    # rainfall stats for month
                    if m in rain_stats and not np.isnan(rain_stats[m]['mean']):
                        mean_r = rain_stats[m]['mean']
                        std_r = rain_stats[m]['std'] if not np.isnan(rain_stats[m]['std']) else max(0.1, overall_std_r)
                    else:
                        mean_r = overall_mean_r
                        std_r = overall_std_r

                    # temperature stats for month
                    if 'temp_stats' in locals() and m in temp_stats and not np.isnan(temp_stats[m]['mean']):
                        mean_t = temp_stats[m]['mean']
                        std_t = temp_stats[m]['std'] if not np.isnan(temp_stats[m]['std']) else max(0.1, overall_std_t)
                    else:
                        mean_t = overall_mean_t
                        std_t = overall_std_t

                    rain_val = float(max(0.0, rng.normal(loc=mean_r, scale=max(0.1, std_r))))
                    temp_val = float(rng.normal(loc=mean_t, scale=max(0.1, std_t)))

                    synth_rows.append({'date': d, 'rainfall': rain_val, 'temperature': temp_val, 'rain_synthetic': True})

                rain_daily = pd.DataFrame(synth_rows)
            else:
                # If user opted out of synthetic generation, keep original rain_daily (may be empty)
                if need_synth and not use_synthetic:
                    print("ℹ️ Synthetic rainfall generation skipped by user preference; rainfall columns will remain missing.")

                # Persist synthetic rainfall for inspection
                try:
                    synth_path = os.path.join(DATA_DIR, 'rainfall_synthetic.csv')
                    rain_daily.to_csv(synth_path, index=False)
                    print(f"✓ Saved synthetic rainfall to {synth_path}")
                except Exception as _e:
                    print(f"Unable to persist synthetic rainfall CSV: {_e}")
                # Informative print for local debugging
                print(f"⚠️ Generated synthetic rainfall/temperature for {len(rain_daily)} days ({date_range[0].date()} to {date_range[-1].date()})")
        except Exception as _e:
            # If anything goes wrong, continue without synthetic generation
            print(f"Synthetic rainfall generation skipped due to error: {_e}")
    else:
        rain_daily = pd.DataFrame()

    # make a date column and merge
    water["date"] = water["Timestamp"].dt.floor("D")
    if not rain_daily.empty:
        df = pd.merge(water, rain_daily, on="date", how="left")
    else:
        df = water.copy()

    # detect numeric columns and coerce
    for c in df.columns:
        if df[c].dtype == object:
            # try numeric conversion
            try:
                df[c] = pd.to_numeric(df[c].str.replace(',', '').replace('', np.nan), errors="coerce")
            except Exception:
                continue

    # basic outlier cap: per-column clamp at 0.1%/99.9% if numeric
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numcols:
        try:
            if df[c].dropna().empty:
                continue
            lo = df[c].quantile(0.001)
            hi = df[c].quantile(0.999)
            df[c] = df[c].clip(lower=lo, upper=hi)
        except Exception:
            continue

    # fill numeric with median
    for c in numcols:
        try:
            if df[c].dropna().empty:
                continue
            df[c] = df[c].fillna(df[c].median())
        except Exception:
            continue

    # add month/season
    df["month"] = df["Timestamp"].dt.month
    df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})

    # compute WQI
    df = compute_wqi(df)

    return df


def compute_wqi(df_in):
    df = df_in.copy()
    # default standards (tweakable)
    standards = {
        "pH": {"ideal": 7.0, "standard": 8.5, "weight": 0.25},
        "Turbidity": {"ideal": 0.0, "standard": 5.0, "weight": 0.35},
        "Specific Conductance": {"ideal": 0.0, "standard": 1500.0, "weight": 0.25},
        "Dissolved Oxygen": {"ideal": 8.0, "standard": 4.0, "weight": 0.15}
    }

    w_cols = []
    # per-row compute
    wqi_vals = []
    for idx, row in df.iterrows():
        num = 0.0
        den = 0.0
        for param, meta in standards.items():
            if param not in df.columns:
                df.at[idx, param + "_Q"] = np.nan
                df.at[idx, param + "_contrib"] = np.nan
                continue
            v = row[param]
            Vio = meta["ideal"]
            Si = meta["standard"]
            w = meta["weight"]
            # compute Q — normalized distance
            denom = Si - Vio if Si != Vio else 1.0
            Q = abs((v - Vio) / denom) * 100
            df.at[idx, param + "_Q"] = Q
            df.at[idx, param + "_contrib"] = Q * w
            num += Q * w
            den += w
        wqi = num / den if den > 0 else np.nan
        wqi_vals.append(wqi)
    df["WQI"] = wqi_vals

    def classify(w):
        if pd.isna(w):
            return "Unknown"
        if w <= 25:
            return "Excellent"
        if w <= 50:
            return "Good"
        if w <= 75:
            return "Poor"
        return "Very Poor"

    df["WQI_Class"] = df["WQI"].apply(classify)
    return df


# ----------------------------
# UI helpers
# ----------------------------

def find_site_col(df):
    candidates = ["Site", "site", "Location", "location", "Station", "station", "Station Name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def render_top_nav(df):
    """Render a modern dashboard header."""
    try:
        st.markdown(
            """
            <div class='dashboard-header'>
              <div class='header-content'>
                <div class='header-text'>
                  <h1>Water Quality Management</h1>
                  <p>Real-time Water Quality Intelligence & Monitoring System</p>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.write("Water Quality Management — Real-time Water Quality Intelligence")


def render_top_sites(df, n=6):
    site_col = find_site_col(df)
    if site_col:
        # compute most recent WQI per site
        latest = df.sort_values("Timestamp").groupby(site_col).tail(1)
        if "WQI" in latest.columns:
            top = latest.sort_values("WQI", ascending=False).head(n)
        else:
            top = latest.head(n)
        rows = []
        for _, r in top.iterrows():
            name = r.get(site_col, "Unknown")
            wqi = r.get("WQI")
            ts = r.get("Timestamp")
            rows.append((name, wqi, ts))
    else:
        # fallback: worst dates
        if "WQI" in df.columns:
            # resample only numeric columns to avoid aggregating string/object dtypes
            numcols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numcols:
                rows = []
            else:
                daily = df.set_index("Timestamp")[numcols].resample("D").mean().reset_index()
                rows = [(d.get("Timestamp").strftime("%Y-%m-%d"), d.get("WQI"), d.get("Timestamp")) for _, d in daily.sort_values("WQI", ascending=False).head(n).iterrows()]
        else:
            rows = [("Record", None, None) for _ in range(n)]

    # render bootstrap-like cards in columns
    cols = st.columns(3)
    i = 0
    for name, wqi, ts in rows:
        with cols[i % 3]:
            st.markdown("<div class='kpi'>", unsafe_allow_html=True)
            img = "https://source.unsplash.com/collection/190727/800x600?water"
            st.image(img, use_container_width=True)
            st.markdown(f"<strong>{name}</strong>", unsafe_allow_html=True)
            st.markdown(f"<div style='color:#64748b;font-size:12px'>{ts}</div>", unsafe_allow_html=True)
            if wqi is None or (isinstance(wqi, float) and math.isnan(wqi)):
                st.markdown("<div style='margin-top:6px'>WQI: N/A</div>", unsafe_allow_html=True)
            else:
                badge = "wqi-good"
                if wqi <= 25:
                    badge = "wqi-excellent"
                elif wqi <= 50:
                    badge = "wqi-good"
                elif wqi <= 75:
                    badge = "wqi-poor"
                else:
                    badge = "wqi-very-poor"
                st.markdown(f"<div style='margin-top:8px'><span class='wqi-badge {badge}'>{wqi:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        i += 1


# ----------------------------
# Visualizations
# ----------------------------

def kpi_row(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    if "WQI" in df.columns:
        c2.metric("Avg WQI", f"{df['WQI'].mean():.1f}")
    else:
        c2.metric("Avg WQI", "N/A")
    if "Turbidity" in df.columns:
        c3.metric("Avg Turbidity", f"{df['Turbidity'].mean():.2f}")
    else:
        c3.metric("Avg Turbidity", "N/A")
    if "rainfall" in df.columns:
        c4.metric("Avg Rainfall", f"{df['rainfall'].mean():.1f} mm")
    else:
        c4.metric("Avg Rainfall", "N/A")


def time_series(df, param="Turbidity", span_days=7):
    if param not in df.columns or df[param].dropna().empty:
        st.info("No data for selected parameter/time range.")
        return
    s = df.set_index("Timestamp").sort_index()[param].dropna()
    s_daily = s.resample("D").mean()
    if s_daily.dropna().empty:
        st.info("No data for selected parameter/time range.")
        return
    s_daily = s_daily.interpolate()
    roll = s_daily.rolling(window=min(span_days, max(1, len(s_daily)))).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_daily.index, y=s_daily.values, mode="lines", name=param, line=dict(color="#7c4dff")))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name=f"{span_days}-day mean", line=dict(color="#ff7eb3")))
    fig.update_layout(margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def wqi_contributions(df):
    contrib_cols = [c for c in df.columns if c.endswith("_contrib")]
    if not contrib_cols:
        st.info("WQI contributions not available.")
        return
    contrib_df = df.set_index("Timestamp")[contrib_cols].resample('D').mean().fillna(0).reset_index()
    if contrib_df.empty:
        st.info("Not enough data for WQI contributions.")
        return
    melted = contrib_df.melt(id_vars='Timestamp', value_vars=contrib_cols)
    melted['component'] = melted['variable'].str.replace('_contrib', '')
    fig = px.area(melted, x='Timestamp', y='value', color='component', labels={'value': 'Contribution'})
    st.plotly_chart(fig, use_container_width=True)


def correlation_panel(df, cols=None):
    if cols is None:
        cols = [c for c in df.columns if c.lower() in ("ph", "turbidity", "specific conductance", "temperature", "dissolved oxygen", "salinity", "rainfall")]
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        st.info("Select at least two numeric parameters for correlation.")
        return
    sub = df[cols]
    presence = sub.notna().astype(int)
    pair_counts = pd.DataFrame(presence.T.dot(presence), index=cols, columns=cols)
    st.markdown("**Pairwise non-missing counts**")
    st.dataframe(pair_counts)
    if pair_counts.values.max() < 5:
        st.info("Too few pairwise observations for reliable correlations.")
        return
    corr = sub.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='vlag', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)


# ----------------------------
# ML demo (simple random forest)
# ----------------------------

def train_rf(df, target='Turbidity'):
    if target not in df.columns:
        return None, None
    df2 = df.dropna(subset=[target]).copy()
    if len(df2) < 30:
        return None, None
    df2['date_num'] = df2['Timestamp'].map(pd.Timestamp.toordinal)
    features = [c for c in ['pH', 'Specific Conductance', 'Temperature', 'Dissolved Oxygen', 'Salinity', 'rainfall', 'date_num'] if c in df2.columns]
    X = df2[features]
    y = df2[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return model, rmse



# ----------------------------
# Advanced Visualizations
# ----------------------------

def seasonal_decomposition_viz(df, param='Turbidity', period=7):
    """Display seasonal decomposition chart for a parameter."""
    if param not in df.columns:
        st.warning(f"Parameter '{param}' not found")
        return
    
    # Prepare data: daily aggregation
    water_sorted = df.sort_values('Timestamp').set_index('Timestamp')
    daily = water_sorted[[param]].resample('D').mean()
    
    if len(daily) < 2 * period:
        st.info(f"Need at least {2*period} daily records for decomposition. Current: {len(daily)}")
        return
    
    # Fill NaN for decomposition
    series = daily[param].ffill().bfill()
    
    if not HAS_STATSMODELS:
        st.warning("Seasonal decomposition requires statsmodels. Install with: `pip install statsmodels`")
        return
    
    try:
        decomposition = seasonal_decompose(series, model='additive', period=period)
        
        # Create subplots
        fig = go.Figure()
        
        # Observed
        fig.add_trace(go.Scatter(
            x=decomposition.observed.index, y=decomposition.observed.values,
            mode='lines', name='Observed', line=dict(color='steelblue', width=2)
        ))
        
        fig.update_layout(
            title=f'{param} — Seasonal Decomposition (Period: {period}d)',
            xaxis_title='Date', yaxis_title='Value',
            height=500, hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual components
        col1, col2 = st.columns(2)
        
        with col1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=decomposition.trend.index, y=decomposition.trend.values,
                mode='lines', name='Trend', line=dict(color='orange', width=2)
            ))
            fig_trend.update_layout(title='Trend Component', xaxis_title='Date', yaxis_title='Value', height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            fig_seasonal = go.Figure()
            fig_seasonal.add_trace(go.Scatter(
                x=decomposition.seasonal.index, y=decomposition.seasonal.values,
                mode='lines', name='Seasonal', line=dict(color='green', width=2)
            ))
            fig_seasonal.update_layout(title='Seasonal Component', xaxis_title='Date', yaxis_title='Value', height=400)
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Residual
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=decomposition.resid.index, y=decomposition.resid.values,
            mode='markers', name='Residual', marker=dict(color='red', size=4)
        ))
        fig_residual.update_layout(title='Residual Component', xaxis_title='Date', yaxis_title='Value', height=400)
        st.plotly_chart(fig_residual, use_container_width=True)
        
    except Exception as e:
        st.error(f"Decomposition failed: {e}")


def animated_timeseries(df, param='Turbidity', window_days=7):
    """Create animated time-series with Plotly."""
    if param not in df.columns:
        st.warning(f"Parameter '{param}' not found")
        return
    
    df_sorted = df.sort_values('Timestamp').copy()
    df_sorted['RollingMean'] = df_sorted[param].rolling(window=window_days, center=True).mean()
    
    # Sample data to avoid too many points
    sample_rate = max(1, len(df_sorted) // 500)
    df_sample = df_sorted.iloc[::sample_rate].copy()
    
    fig = px.scatter(
        df_sample,
        x='Timestamp',
        y=param,
        title=f'{param} Over Time',
        labels={'Timestamp': 'Date', param: 'Value'},
        hover_data={'Timestamp': '|%Y-%m-%d %H:%M', param: ':.3f'}
    )
    
    # Add rolling mean as line
    fig.add_trace(go.Scatter(
        x=df_sorted['Timestamp'],
        y=df_sorted['RollingMean'],
        mode='lines',
        name=f'{window_days}d Moving Average',
        line=dict(color='red', width=3),
        hovertemplate='<b>Rolling Mean</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.3f}'
    ))
    
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def correlation_heatmap_enhanced(df, method='spearman'):
    """Enhanced correlation heatmap with pairwise counts."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation")
        return

    # Filter out internal or helper columns to avoid clutter
    exclude_tokens = ['_', 'Q', 'contrib', 'WQI', 'date']
    candidate_cols = [c for c in numeric_cols if not any(tok in c for tok in exclude_tokens)]

    if len(candidate_cols) > 12:
        # Prioritize columns by non-missing count and then variance
        col_stats = []
        for c in candidate_cols:
            non_missing = df[c].notna().sum()
            var = df[c].var() if df[c].notna().sum() > 1 else 0.0
            col_stats.append((c, non_missing, var))
        # sort by non_missing desc, var desc
        col_stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
        selected = [c for c, _, _ in col_stats[:12]]
        st.markdown(f"**Showing top {len(selected)} parameters for clarity (by data availability & variance):** {', '.join(selected)}")
    else:
        selected = candidate_cols

    if len(selected) < 2:
        st.info("Not enough suitable numeric parameters for correlation visualization.")
        return

    # Compute correlation for selected set
    corr_matrix = df[selected].corr(method=method)

    # Pairwise counts (non-missing pairs)
    pairwise_counts = df[selected].notna().astype(int).T.dot(df[selected].notna().astype(int))

    # Create heatmap with annotations
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{corr_matrix.iloc[i,j]:.2f}<br>n={pairwise_counts.iloc[i,j]}" 
               for j in range(len(corr_matrix.columns))] 
              for i in range(len(corr_matrix.columns))],
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{y} vs %{x}<br>Corr: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Correlation')
    ))

    fig.update_layout(
        title=f'{method.capitalize()} Correlation Matrix',
        xaxis_title='Parameter', yaxis_title='Parameter',
        height=600, width=800
    )

    st.plotly_chart(fig, use_container_width=True)


def parameter_boxplot_by_weekday(df, param='Turbidity'):
    """Boxplot of parameter by day-of-week."""
    if param not in df.columns:
        st.warning(f"Parameter '{param}' not found")
        return
    
    df_plot = df.copy()
    df_plot['DayOfWeek'] = df_plot['Timestamp'].dt.day_name()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_plot['DayOfWeek'] = pd.Categorical(df_plot['DayOfWeek'], categories=day_order, ordered=True)
    
    fig = px.box(
        df_plot,
        x='DayOfWeek',
        y=param,
        title=f'{param} Distribution by Day of Week',
        points='outliers',
        hover_data=[param]
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def wqi_trend_with_bands(df):
    """WQI trend with confidence bands."""
    df_sorted = df.sort_values('Timestamp').copy()
    
    # Compute rolling stats
    window = 7
    df_sorted['WQI_MA'] = df_sorted['WQI'].rolling(window=window, center=True).mean()
    df_sorted['WQI_Std'] = df_sorted['WQI'].rolling(window=window, center=True).std()
    df_sorted['Upper_Band'] = df_sorted['WQI_MA'] + 1.96 * df_sorted['WQI_Std']
    df_sorted['Lower_Band'] = df_sorted['WQI_MA'] - 1.96 * df_sorted['WQI_Std']
    
    fig = go.Figure()
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=df_sorted['Timestamp'],
        y=df_sorted['Upper_Band'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        name='Upper Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_sorted['Timestamp'],
        y=df_sorted['Lower_Band'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% CI',
        fillcolor='rgba(68, 68, 68, 0.2)'
    ))
    
    # WQI Moving Average
    fig.add_trace(go.Scatter(
        x=df_sorted['Timestamp'],
        y=df_sorted['WQI_MA'],
        mode='lines',
        name=f'WQI {window}d MA',
        line=dict(color='darkblue', width=2)
    ))
    
    # Add classification zones
    fig.add_hline(y=80, line_dash='dash', line_color='green', annotation_text='Excellent')
    fig.add_hline(y=60, line_dash='dash', line_color='blue', annotation_text='Good')
    fig.add_hline(y=40, line_dash='dash', line_color='orange', annotation_text='Poor')
    
    fig.update_layout(
        title='WQI Trend with Confidence Bands & Classification Zones',
        xaxis_title='Date',
        yaxis_title='WQI',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Pages
# ----------------------------

def page_home(df):
    """Modern admin-style dashboard matching professional UI standards."""
    
    show_rainfall_coverage_warning(df)
    
    # Top Row - Key Metrics Cards
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_wqi = df['WQI'].iloc[-1] if not df.empty else 0
        prev_wqi = df['WQI'].iloc[-7] if len(df) > 7 else current_wqi
        delta = ((current_wqi - prev_wqi) / prev_wqi * 100) if prev_wqi != 0 else 0
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; right: -10px; top: -10px; font-size: 4em; opacity: 0.2;">💧</div>
            <div style="font-size: 0.85em; font-weight: 500; opacity: 0.9; margin-bottom: 8px;">CURRENT WQI</div>
            <div style="font-size: 2.5em; font-weight: 700; margin-bottom: 8px;">{current_wqi:.1f}</div>
            <div style="font-size: 0.85em; opacity: 0.9;">
                <span style="color: {'#4ade80' if delta >= 0 else '#f87171'};">
                    {'▲' if delta >= 0 else '▼'} {abs(delta):.1f}%
                </span> vs last week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_records = len(df)
        avg_daily = total_records / ((df['Timestamp'].max() - df['Timestamp'].min()).days + 1)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; right: -10px; top: -10px; font-size: 4em; opacity: 0.2;">📊</div>
            <div style="font-size: 0.85em; font-weight: 500; opacity: 0.9; margin-bottom: 8px;">TOTAL RECORDS</div>
            <div style="font-size: 2.5em; font-weight: 700; margin-bottom: 8px;">{total_records:,}</div>
            <div style="font-size: 0.85em; opacity: 0.9;">{avg_daily:.0f} records/day</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_ph = df['pH'].mean() if 'pH' in df.columns else 7.0
        ph_status = "Optimal" if 6.5 <= avg_ph <= 8.5 else "Off-range"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; right: -10px; top: -10px; font-size: 4em; opacity: 0.2;">⚗️</div>
            <div style="font-size: 0.85em; font-weight: 500; opacity: 0.9; margin-bottom: 8px;">AVERAGE pH</div>
            <div style="font-size: 2.5em; font-weight: 700; margin-bottom: 8px;">{avg_ph:.2f}</div>
            <div style="font-size: 0.85em; opacity: 0.9;">Status: {ph_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_turb = df['Turbidity'].mean() if 'Turbidity' in df.columns else 0
        turb_trend = "Low" if avg_turb < 5 else "Moderate" if avg_turb < 10 else "High"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; right: -10px; top: -10px; font-size: 4em; opacity: 0.2;">🌊</div>
            <div style="font-size: 0.85em; font-weight: 500; opacity: 0.9; margin-bottom: 8px;">AVG TURBIDITY</div>
            <div style="font-size: 2.5em; font-weight: 700; margin-bottom: 8px;">{avg_turb:.2f}</div>
            <div style="font-size: 0.85em; opacity: 0.9;">Level: {turb_trend}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second Row - Classification with Circular Progress
    st.markdown("### 🎯 Water Quality Distribution")
    col1, col2, col3, col4 = st.columns(4)
    
    classifications = [
        ('Excellent', '#10b981', '🟢'),
        ('Good', '#3b82f6', '🔵'),
        ('Poor', '#f59e0b', '🟠'),
        ('Very Poor', '#ef4444', '🔴')
    ]
    
    for col, (class_name, color, emoji) in zip([col1, col2, col3, col4], classifications):
        with col:
            count = (df['WQI_Class'] == class_name).sum()
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            
            st.markdown(f"""
            <div style="background: white; padding: 30px 20px; border-radius: 12px; 
                        text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                        border: 1px solid #f0f0f0;">
                <div style="position: relative; width: 120px; height: 120px; margin: 0 auto 15px auto;">
                    <svg width="120" height="120" style="transform: rotate(-90deg);">
                        <circle cx="60" cy="60" r="50" fill="none" stroke="#f0f0f0" stroke-width="10"/>
                        <circle cx="60" cy="60" r="50" fill="none" stroke="{color}" stroke-width="10"
                                stroke-dasharray="{314 * pct / 100} 314" 
                                stroke-linecap="round"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                font-size: 1.8em; font-weight: 700; color: {color};">
                        {pct:.0f}%
                    </div>
                </div>
                <div style="font-size: 1.3em; margin-bottom: 5px;">{emoji}</div>
                <div style="font-weight: 600; color: #333; font-size: 1.1em; margin-bottom: 5px;">{class_name}</div>
                <div style="color: #666; font-size: 0.9em;">{count:,} records</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    st.markdown("### 📈 Analytics Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### WQI Trend Analysis")
        recent_df = df[df['Timestamp'] >= df['Timestamp'].max() - timedelta(days=30)].copy()
        if len(recent_df) > 0:
            fig = px.line(
                recent_df.sort_values('Timestamp'),
                x='Timestamp',
                y='WQI',
                title='',
                template='plotly_white'
            )
            fig.update_traces(line_color='#667eea', line_width=3)
            fig.add_hline(y=80, line_dash='dash', line_color='#10b981', opacity=0.3)
            fig.add_hline(y=60, line_dash='dash', line_color='#3b82f6', opacity=0.3)
            fig.add_hline(y=40, line_dash='dash', line_color='#f59e0b', opacity=0.3)
            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis_title="",
                yaxis_title="WQI Score",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='white',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data")
    
    with col2:
        st.markdown("#### Classification Distribution")
        class_counts = df['WQI_Class'].value_counts()
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title='',
            color=class_counts.index,
            color_discrete_map={
                'Excellent': '#10b981',
                'Good': '#3b82f6',
                'Poor': '#f59e0b',
                'Very Poor': '#ef4444'
            },
            template='plotly_white'
        )
        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            xaxis_title="",
            yaxis_title="Count",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Parameter Heatmap
    st.markdown("---")
    st.markdown("### 🔥 Parameter Correlation Heatmap")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_params = [c for c in numeric_cols if '_' not in c and c not in ['WQI', 'Record number', 'date']][:6]
    
    if len(key_params) >= 2:
        col1, col2 = st.columns([2, 1])
        with col1:
            corr_matrix = df[key_params].corr(method='spearman')
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                text_auto='.2f',
                aspect='auto',
                title='Spearman Correlation between Key Parameters'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🌍 Monitoring Overview")
            
            # Site information
            site_col = find_site_col(df)
            if site_col:
                total_sites = df[site_col].nunique()
                active_sites = df[df['Timestamp'] >= df['Timestamp'].max() - timedelta(days=7)][site_col].nunique()
                st.metric("Total Sites", total_sites, delta=f"{active_sites} active (7d)")
            else:
                st.metric("Monitoring Points", "Single Location")
            
            # Data coverage
            date_range = (df['Timestamp'].max() - df['Timestamp'].min()).days
            st.metric("Data Coverage", f"{date_range} days", delta=f"{len(df):,} records")
            
            # Average samples per day
            avg_samples = len(df) / max(date_range, 1)
            st.metric("Sampling Rate", f"{avg_samples:.1f}/day")
            
            # Latest update
            latest = df['Timestamp'].max()
            st.info(f"📅 Latest update:\n{latest.strftime('%Y-%m-%d %H:%M')}")
    
    # Recent Anomalies/Alerts
    st.markdown("---")
    st.markdown("### ⚠️ Recent Anomalies")
    
    recent_alerts = df[df['Timestamp'] >= df['Timestamp'].max() - timedelta(days=7)].copy()
    
    anomaly_count = 0
    alerts_content = []
    
    # Check for very poor WQI
    very_poor_recent = recent_alerts[recent_alerts['WQI_Class'] == 'Very Poor']
    if len(very_poor_recent) > 0:
        anomaly_count += len(very_poor_recent)
        alerts_content.append(f"🔴 **{len(very_poor_recent)} Very Poor WQI readings** in last 7 days")
    
    # Check for high turbidity
    if 'Turbidity' in df.columns:
        high_turb = recent_alerts[recent_alerts['Turbidity'] > df['Turbidity'].quantile(0.95)]
        if len(high_turb) > 0:
            anomaly_count += len(high_turb)
            alerts_content.append(f"🟠 **{len(high_turb)} High Turbidity events** (>95th percentile)")
    
    # Check for pH out of range
    if 'pH' in df.columns:
        bad_ph = recent_alerts[(recent_alerts['pH'] < 6.5) | (recent_alerts['pH'] > 8.5)]
        if len(bad_ph) > 0:
            anomaly_count += len(bad_ph)
            alerts_content.append(f"🟡 **{len(bad_ph)} pH out-of-range** readings")
    
    if anomaly_count == 0:
        st.success("✅ No anomalies detected in the last 7 days!")
    else:
        for alert in alerts_content:
            st.warning(alert)
    
    # Advanced Visualizations Section
    st.markdown("---")
    st.markdown("### 📊 Advanced Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge Chart for Current WQI
        st.markdown("#### 🎯 Current WQI Gauge")
        current_wqi = df['WQI'].iloc[-1] if not df.empty else 0
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_wqi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Water Quality Index"},
            delta={'reference': df['WQI'].mean(), 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#fee2e2'},
                    {'range': [40, 60], 'color': '#fef3c7'},
                    {'range': [60, 80], 'color': '#dbeafe'},
                    {'range': [80, 100], 'color': '#dcfce7'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Radar Chart for Parameter Comparison
        st.markdown("#### 🕸️ Parameter Profile (Current vs Ideal)")
        
        params_for_radar = ['pH', 'Dissolved Oxygen', 'Turbidity', 'Specific Conductance']
        params_available = [p for p in params_for_radar if p in df.columns]
        
        if len(params_available) >= 3:
            current_values = []
            ideal_values = []
            param_names = []
            
            for param in params_available:
                current_val = df[param].iloc[-1]
                param_range = df[param].max() - df[param].min()
                normalized_current = ((current_val - df[param].min()) / param_range * 100) if param_range > 0 else 50
                
                current_values.append(normalized_current)
                param_names.append(param.split()[0] if ' ' in param else param)
                
                # Ideal values (normalized to 80-90 range for "good")
                ideal_values.append(85)
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=current_values,
                theta=param_names,
                fill='toself',
                name='Current',
                line_color='#667eea'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=ideal_values,
                theta=param_names,
                fill='toself',
                name='Target',
                line_color='#10b981',
                opacity=0.5
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=300,
                margin=dict(l=50, r=50, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    # Sankey Diagram for WQI Flow
    st.markdown("#### 🌊 Water Quality Flow Analysis")
    st.caption("Shows how water quality transitions between classifications over time")
    
    # Simplified approach: compare first half vs second half
    df_sorted = df.sort_values('Timestamp').copy()
    midpoint = len(df_sorted) // 2
    
    first_half = df_sorted.iloc[:midpoint]
    second_half = df_sorted.iloc[midpoint:]
    
    # Count classifications in each half
    first_counts = first_half['WQI_Class'].value_counts()
    second_counts = second_half['WQI_Class'].value_counts()
    
    all_classes = ['Excellent', 'Good', 'Poor', 'Very Poor']
    
    # Build cleaner Sankey with just two time periods
    node_labels = [f"Start: {cls}" for cls in all_classes] + [f"End: {cls}" for cls in all_classes]
    node_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'] * 2
    
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Calculate proportional transitions
    for i, class_from in enumerate(all_classes):
        count_from = first_counts.get(class_from, 0)
        if count_from > 0:
            for j, class_to in enumerate(all_classes):
                count_to = second_counts.get(class_to, 0)
                if count_to > 0:
                    # Proportional flow based on both counts
                    flow = min(count_from, count_to) // 3
                    if flow > 0:
                        sources.append(i)
                        targets.append(j + 4)  # Offset by 4 for second period
                        values.append(flow)
                        
                        # Color based on quality change
                        if i == j:  # Same class
                            link_colors.append('rgba(150, 150, 150, 0.3)')
                        elif i < j:  # Improving
                            link_colors.append('rgba(16, 185, 129, 0.4)')
                        else:  # Declining
                            link_colors.append('rgba(239, 68, 68, 0.4)')
    
    if len(sources) > 0:
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,
                thickness=35,
                line=dict(color="black", width=1),
                label=[f"<b>{label}</b>" for label in node_labels],  # Bold labels
                color=node_colors,
                customdata=[f"Count: {first_counts.get(cls, 0)}" for cls in all_classes] + 
                          [f"Count: {second_counts.get(cls, 0)}" for cls in all_classes],
                hovertemplate='%{label}<br>%{customdata}<extra></extra>'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate='Flow: %{value}<extra></extra>'
            ),
            textfont=dict(color="black", size=16, family="Arial Black")
        )])
        
        fig_sankey.update_layout(
            title={
                'text': "<b>Water Quality Flow: First Half → Second Half of Dataset</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#1e293b'}
            },
            height=550,
            font=dict(size=16, family="Arial", color='black'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Add legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **Green flows**: Quality improving")
        with col2:
            st.markdown("⚪ **Gray flows**: Quality stable")
        with col3:
            st.markdown("🔴 **Red flows**: Quality declining")
    else:
        st.info("Not enough data variation to show flow transitions")
    
    # Waterfall Chart for WQI Components
    st.markdown("#### 💧 WQI Component Breakdown (Waterfall Analysis)")
    
    components = []
    values = []
    
    if 'pH_Contribution' in df.columns:
        components.append('pH')
        values.append(df['pH_Contribution'].iloc[-1])
    if 'Turbidity_Contribution' in df.columns:
        components.append('Turbidity')
        values.append(df['Turbidity_Contribution'].iloc[-1])
    if 'DO_Contribution' in df.columns:
        components.append('DO')
        values.append(df['DO_Contribution'].iloc[-1])
    if 'SC_Contribution' in df.columns:
        components.append('Conductance')
        values.append(df['SC_Contribution'].iloc[-1])
    
    if len(components) > 0:
        fig_waterfall = go.Figure(go.Waterfall(
            name="WQI",
            orientation="v",
            measure=["relative"] * len(components) + ["total"],
            x=components + ["Total WQI"],
            textposition="outside",
            text=[f"{v:.1f}" for v in values] + [f"{sum(values):.1f}"],
            y=values + [sum(values)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#667eea"}}
        ))
        
        fig_waterfall.update_layout(
            title="WQI Component Contributions",
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
        <p>Last updated: <strong>{}</strong></p>
        <p>Data source: Water Quality Monitoring Network</p>
    </div>
    """.format(df['Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


def show_rainfall_coverage_warning(df, threshold: float = 0.05):
    """Show a banner warning when rainfall-related columns have very low coverage."""
    rain_cols = [c for c in df.columns if c.lower().startswith('rain') or c.lower() == 'rainfall']
    if not rain_cols:
        return
    coverage = df[rain_cols].notna().mean()
    # overall fraction of columns that are essentially empty
    low = coverage[coverage <= 0.01].index.tolist()
    if len(low) == len(rain_cols):
        st.info("📊 **Note:** Rainfall data overlaps water data using synthetically generated rainfall (based on 2022 statistical patterns). This allows model training with rainfall features.")
    elif (coverage <= threshold).any():
        bad = coverage[coverage <= threshold].index.tolist()
        st.info(f"Note: Low coverage for rainfall features: {', '.join(bad)} (<= {threshold*100:.0f}% non-missing)")


def page_explore(df):
    st.title("🔍 Explore — Interactive Data Discovery")
    
    # Summary overview cards at top
    st.markdown("### 📊 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Measurements", f"{len(df):,}")
    with col2:
        date_range_days = (df['Timestamp'].max() - df['Timestamp'].min()).days
        st.metric("Date Range", f"{date_range_days} days")
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Parameters", len([c for c in numeric_cols if '_' not in c]))
    with col4:
        site_col = find_site_col(df)
        if site_col:
            st.metric("Monitoring Sites", df[site_col].nunique())
        else:
            st.metric("Data Quality", f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])):.1%}")
    
    st.markdown("---")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series Explorer", "📊 Distribution Analysis", "🔍 Parameter Inspector", "🌊 3D Visualization"])
    
    with tab1:
        st.markdown("#### Interactive Time Series Analysis")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['WQI', 'Record number']]
            param = st.selectbox("Select parameter:", options=params, index=min(2, len(params)-1), key="explore_param")
        with col2:
            window = st.slider("Moving avg (days):", 1, 30, 7, key="explore_window")
        
        # Date range filter
        date_range = st.date_input("Date range:", [df['Timestamp'].min().date(), df['Timestamp'].max().date()], key="explore_dates")
        if len(date_range) == 2:
            start, end = date_range
            mask = (df['Timestamp'].dt.date >= start) & (df['Timestamp'].dt.date <= end)
            dff = df.loc[mask]
        else:
            dff = df
        
        # Display statistics for selected period
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{dff[param].mean():.2f}")
        with col2:
            st.metric("Median", f"{dff[param].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{dff[param].std():.2f}")
        with col4:
            st.metric("Range", f"{dff[param].max() - dff[param].min():.2f}")
        
        animated_timeseries(dff, param=param, window_days=window)
        
        # Add comparison feature
        st.markdown("##### Compare Multiple Parameters")
        compare_params = st.multiselect("Select parameters to compare:", params, default=[params[0]] if params else [], key="compare_params")
        if compare_params:
            fig = go.Figure()
            for p in compare_params:
                fig.add_trace(go.Scatter(
                    x=dff.sort_values('Timestamp')['Timestamp'],
                    y=dff.sort_values('Timestamp')[p],
                    mode='lines',
                    name=p,
                    line=dict(width=2)
                ))
            fig.update_layout(
                title="Multi-Parameter Comparison",
                xaxis_title="Time",
                yaxis_title="Value",
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Distribution & Statistical Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            params = st.multiselect("Select parameters:", 
                                   options=[c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['WQI', 'Record number']], 
                                   default=[p for p in ['Turbidity','pH'] if p in df.columns],
                                   key="dist_params")
        with col2:
            chart_type = st.radio("Chart type:", ["Histogram + Box", "Violin Plot", "Density"], horizontal=True)
        
        date_range = st.date_input("Date range:", [df['Timestamp'].min().date(), df['Timestamp'].max().date()], key="dist_dates")
        if len(date_range) == 2:
            start, end = date_range
            mask = (df['Timestamp'].dt.date >= start) & (df['Timestamp'].dt.date <= end)
            dff = df.loc[mask]
        else:
            dff = df
        
        if params:
            for p in params:
                st.markdown(f"##### {p}")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if chart_type == "Histogram + Box":
                        fig = px.histogram(dff, x=p, nbins=40, marginal='box', title=f"{p} Distribution")
                    elif chart_type == "Violin Plot":
                        fig = px.violin(dff, y=p, box=True, title=f"{p} Violin Plot")
                    else:
                        from scipy.stats import gaussian_kde
                        data = dff[p].dropna()
                        if len(data) > 0:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(data.min(), data.max(), 200)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), fill='tozeroy', name='Density'))
                            fig.update_layout(title=f"{p} Density Plot", xaxis_title=p, yaxis_title="Density")
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Statistical summary
                    st.markdown("**Statistics:**")
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', '25%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                        'Value': [
                            len(dff[p].dropna()),
                            dff[p].mean(),
                            dff[p].median(),
                            dff[p].std(),
                            dff[p].min(),
                            dff[p].quantile(0.25),
                            dff[p].quantile(0.75),
                            dff[p].max(),
                            dff[p].skew(),
                            dff[p].kurtosis()
                        ]
                    })
                    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
    
    with tab3:
        st.markdown("#### Parameter Deep Dive Inspector")
        
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['WQI', 'Record number']]
        selected_param = st.selectbox("Select parameter to inspect:", params, key="inspect_param")
        
        if selected_param:
            # Time-based analysis
            st.markdown("##### Temporal Patterns")
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly pattern
                if 'Hour' not in df.columns:
                    df['Hour'] = df['Timestamp'].dt.hour
                hourly_avg = df.groupby('Hour')[selected_param].mean()
                fig = px.line(x=hourly_avg.index, y=hourly_avg.values, 
                             title="Average by Hour of Day",
                             labels={'x': 'Hour', 'y': selected_param})
                fig.update_traces(line_color='#667eea', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Day of week pattern
                if 'DayOfWeek' not in df.columns:
                    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_avg = df.groupby('DayOfWeek')[selected_param].mean().reindex(day_order)
                fig = px.bar(x=dow_avg.index, y=dow_avg.values,
                            title="Average by Day of Week",
                            labels={'x': 'Day', 'y': selected_param})
                fig.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig, use_container_width=True)
            
            # Outlier detection
            st.markdown("##### Outlier Detection")
            Q1 = df[selected_param].quantile(0.25)
            Q3 = df[selected_param].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[selected_param] < (Q1 - 1.5 * IQR)) | (df[selected_param] > (Q3 + 1.5 * IQR))]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outliers Detected", len(outliers))
            with col2:
                st.metric("% of Data", f"{len(outliers)/len(df)*100:.1f}%")
            with col3:
                st.metric("IQR", f"{IQR:.2f}")
            
            if len(outliers) > 0:
                st.dataframe(outliers[['Timestamp', selected_param]].head(10), use_container_width=True)
    
    with tab4:
        st.markdown("#### 3D Interactive Visualization")
        st.info("💡 Explore relationships between three parameters in 3D space")
        
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number']]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_param = st.selectbox("X-axis:", params, index=0 if len(params) > 0 else 0, key="3d_x")
        with col2:
            y_param = st.selectbox("Y-axis:", params, index=1 if len(params) > 1 else 0, key="3d_y")
        with col3:
            z_param = st.selectbox("Z-axis:", params, index=2 if len(params) > 2 else 0, key="3d_z")
        
        color_by = st.selectbox("Color by:", ['WQI', 'WQI_Class'] + params, key="3d_color")
        
        # Sample data for performance
        df_sample = df.sample(min(1000, len(df)))
        
        fig = px.scatter_3d(
            df_sample,
            x=x_param,
            y=y_param,
            z=z_param,
            color=color_by,
            title=f"3D Scatter: {x_param} vs {y_param} vs {z_param}",
            hover_data=['Timestamp'],
            height=600
        )
        fig.update_traces(marker=dict(size=4))
        st.plotly_chart(fig, use_container_width=True)


def page_analysis(df):
    st.title("🧭 Analysis — Advanced Water Quality Analytics")
    # Rainfall coverage note
    show_rainfall_coverage_warning(df)
    
    # Summary cards at top
    st.markdown("### 📊 Analysis Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        wqi_trend_value = df['WQI'].iloc[-7:].mean() - df['WQI'].iloc[-14:-7].mean() if len(df) >= 14 else 0
        trend_icon = "📈" if wqi_trend_value > 0 else "📉"
        st.metric("WQI Trend (7d)", f"{trend_icon} {abs(wqi_trend_value):.2f}", delta=f"{'Improving' if wqi_trend_value > 0 else 'Declining'}")
    
    with col2:
        corr_strength = df[[c for c in df.columns if df[c].dtype.kind in 'fiu']].corr().abs().mean().mean()
        st.metric("Avg Correlation", f"{corr_strength:.2f}", delta="Parameter strength")
    
    with col3:
        variance_score = df['WQI'].std() / df['WQI'].mean() * 100 if df['WQI'].mean() != 0 else 0
        st.metric("WQI Variability", f"{variance_score:.1f}%", delta="Coefficient of variation")
    
    with col4:
        recent_quality = (df['WQI'].iloc[-100:] >= 60).sum() / 100 * 100 if len(df) >= 100 else 0
        st.metric("Recent Quality", f"{recent_quality:.0f}%", delta="Last 100 samples")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 WQI Trends", 
        "🔄 Decomposition", 
        "🔗 Correlations", 
        "📅 Temporal Patterns",
        "🎯 Anomaly Detection",
        "📊 Statistical Tests"
    ])
    
    with tab1:
        st.subheader("WQI Trend Analysis with Confidence Bands")
        wqi_trend_with_bands(df)
        
        # Additional trend insights
        st.markdown("##### Trend Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # WQI change over time periods
            periods = {
                'Last 7 days': df['WQI'].iloc[-7:].mean() if len(df) >= 7 else 0,
                'Last 30 days': df['WQI'].iloc[-30:].mean() if len(df) >= 30 else 0,
                'Last 90 days': df['WQI'].iloc[-90:].mean() if len(df) >= 90 else 0,
                'Overall': df['WQI'].mean()
            }
            period_df = pd.DataFrame(list(periods.items()), columns=['Period', 'Average WQI'])
            fig = px.bar(period_df, x='Period', y='Average WQI', 
                        title='Average WQI by Time Period',
                        color='Average WQI',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # WQI volatility over time
            df_sorted = df.sort_values('Timestamp')
            window = 30
            if len(df_sorted) >= window:
                df_sorted['WQI_Rolling_Std'] = df_sorted['WQI'].rolling(window=window).std()
                fig = px.line(df_sorted, x='Timestamp', y='WQI_Rolling_Std',
                             title=f'WQI Volatility ({window}-day Rolling Std Dev)',
                             labels={'WQI_Rolling_Std': 'Standard Deviation'})
                fig.update_traces(line_color='#f59e0b')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Seasonal Decomposition")
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu']
        param = st.selectbox("Select parameter for decomposition:", options=params, index=min(2, len(params)-1))
        period = st.slider("Decomposition period (days):", min_value=3, max_value=14, value=7)
        seasonal_decomposition_viz(df, param=param, period=period)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            method = st.radio("Correlation method:", ['spearman', 'pearson'], horizontal=True)
        with col2:
            min_corr = st.slider("Min correlation to display:", 0.0, 1.0, 0.3, 0.05)
        
        correlation_heatmap_enhanced(df, method=method)
        
        # Top correlations table
        st.markdown("##### Strongest Correlations")
        numeric_cols = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number']]
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Extract upper triangle correlations
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'Parameter 1': corr_matrix.columns[i],
                    'Parameter 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df[abs(corr_df['Correlation']) >= min_corr].sort_values('Correlation', key=abs, ascending=False)
        
        if len(corr_df) > 0:
            st.dataframe(corr_df.head(10), use_container_width=True, hide_index=True)
        else:
            st.info(f"No correlations found above threshold {min_corr}")
    
    with tab4:
        st.subheader("Temporal Pattern Analysis")
        
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu']
        param = st.selectbox("Select parameter:", options=params, index=min(2, len(params)-1), key="temporal_param")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week patterns
            parameter_boxplot_by_weekday(df, param=param)
        
        with col2:
            # Monthly patterns
            st.markdown("##### Monthly Variation")
            if 'month' in df.columns or 'Timestamp' in df.columns:
                df_temp = df.copy()
                if 'month' not in df_temp.columns:
                    df_temp['month'] = df_temp['Timestamp'].dt.month
                
                monthly_avg = df_temp.groupby('month')[param].agg(['mean', 'std']).reset_index()
                month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                              7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
                monthly_avg['Month'] = monthly_avg['month'].map(month_names)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_avg['Month'],
                    y=monthly_avg['mean'],
                    error_y=dict(type='data', array=monthly_avg['std']),
                    name=param,
                    marker_color='#667eea'
                ))
                fig.update_layout(title=f'{param} - Monthly Average with Std Dev',
                                 xaxis_title='Month', yaxis_title=param, height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        # Hour of day heatmap
        st.markdown("##### Hourly Patterns")
        if 'Timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['hour'] = df_temp['Timestamp'].dt.hour
            df_temp['day_name'] = df_temp['Timestamp'].dt.day_name()
            
            pivot_table = df_temp.pivot_table(values=param, index='day_name', columns='hour', aggfunc='mean')
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_table = pivot_table.reindex(day_order)
            
            fig = px.imshow(pivot_table, 
                           labels=dict(x="Hour of Day", y="Day of Week", color=param),
                           title=f'{param} Heatmap - Day vs Hour',
                           color_continuous_scale='Viridis',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("🎯 Anomaly Detection & Outlier Analysis")
        
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number']]
        selected_param = st.selectbox("Select parameter for anomaly detection:", params, key="anomaly_param")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            method = st.radio("Detection method:", 
                            ["IQR (Interquartile Range)", "Z-Score", "Isolation Forest"],
                            key="anomaly_method")
            
            if method == "Z-Score":
                threshold = st.slider("Z-score threshold:", 1.5, 4.0, 3.0, 0.5)
            elif method == "IQR (Interquartile Range)":
                multiplier = st.slider("IQR multiplier:", 1.0, 3.0, 1.5, 0.5)
        
        with col2:
            if method == "IQR (Interquartile Range)":
                Q1 = df[selected_param].quantile(0.25)
                Q3 = df[selected_param].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                anomalies = df[(df[selected_param] < lower_bound) | (df[selected_param] > upper_bound)]
                
            elif method == "Z-Score":
                mean = df[selected_param].mean()
                std = df[selected_param].std()
                z_scores = np.abs((df[selected_param] - mean) / std)
                anomalies = df[z_scores > threshold]
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            
            else:  # Isolation Forest
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.05, random_state=42)
                predictions = iso_forest.fit_predict(df[[selected_param]].dropna())
                anomalies = df.loc[df[[selected_param]].dropna().index[predictions == -1]]
                lower_bound = df[selected_param].min()
                upper_bound = df[selected_param].max()
            
            # Visualization
            fig = go.Figure()
            
            # Normal data
            normal_data = df[~df.index.isin(anomalies.index)].sort_values('Timestamp')
            fig.add_trace(go.Scatter(
                x=normal_data['Timestamp'],
                y=normal_data[selected_param],
                mode='markers',
                name='Normal',
                marker=dict(color='#3b82f6', size=4, opacity=0.6)
            ))
            
            # Anomalies
            fig.add_trace(go.Scatter(
                x=anomalies['Timestamp'],
                y=anomalies[selected_param],
                mode='markers',
                name='Anomaly',
                marker=dict(color='#ef4444', size=8, symbol='x')
            ))
            
            # Bounds
            if method != "Isolation Forest":
                fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                             annotation_text="Upper Bound")
                fig.add_hline(y=lower_bound, line_dash="dash", line_color="red",
                             annotation_text="Lower Bound")
            
            fig.update_layout(title=f'Anomaly Detection: {selected_param}',
                            xaxis_title='Time', yaxis_title=selected_param,
                            height=400, hovermode='closest')
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Anomalies", len(anomalies))
        with col2:
            st.metric("% of Data", f"{len(anomalies)/len(df)*100:.2f}%")
        with col3:
            st.metric("Mean Value", f"{df[selected_param].mean():.2f}")
        with col4:
            st.metric("Anomaly Mean", f"{anomalies[selected_param].mean():.2f}" if len(anomalies) > 0 else "N/A")
        
        # Show anomaly records
        if len(anomalies) > 0:
            st.markdown("##### Recent Anomalies")
            display_cols = ['Timestamp', selected_param, 'WQI', 'WQI_Class'] if 'WQI' in anomalies.columns else ['Timestamp', selected_param]
            st.dataframe(anomalies[display_cols].sort_values('Timestamp', ascending=False).head(10), 
                        use_container_width=True, hide_index=True)
    
    with tab6:
        st.subheader("📊 Statistical Hypothesis Testing")
        st.info("💡 Compare parameter distributions across different groups or time periods")
        
        params = [c for c in df.columns if df[c].dtype.kind in 'fiu']
        test_param = st.selectbox("Select parameter to test:", params, key="test_param")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Compare by Quality Classification")
            
            # Group by WQI class
            if 'WQI_Class' in df.columns:
                groups = df.groupby('WQI_Class')[test_param].apply(list).to_dict()
                
                # ANOVA test
                from scipy.stats import f_oneway
                group_data = [data for data in groups.values() if len(data) > 0]
                
                if len(group_data) >= 2:
                    f_stat, p_value = f_oneway(*group_data)
                    
                    st.markdown(f"""
                    **ANOVA Test Results:**
                    - F-statistic: `{f_stat:.4f}`
                    - P-value: `{p_value:.4f}`
                    - Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'} (α=0.05)
                    """)
                    
                    if p_value < 0.05:
                        st.success(f"{test_param} shows significant differences across WQI classifications")
                    else:
                        st.warning(f"{test_param} shows no significant differences across WQI classifications")
                    
                    # Box plot comparison
                    fig = px.box(df, x='WQI_Class', y=test_param,
                                title=f'{test_param} Distribution by WQI Class',
                                color='WQI_Class',
                                color_discrete_map={
                                    'Excellent': '#10b981',
                                    'Good': '#3b82f6',
                                    'Poor': '#f59e0b',
                                    'Very Poor': '#ef4444'
                                })
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Time Period Comparison")
            
            # Compare recent vs historical
            if len(df) >= 60:
                recent = df[test_param].iloc[-30:]
                historical = df[test_param].iloc[-60:-30]
                
                # T-test
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(recent.dropna(), historical.dropna())
                
                st.markdown(f"""
                **T-Test Results (Last 30 vs Previous 30 days):**
                - T-statistic: `{t_stat:.4f}`
                - P-value: `{p_value:.4f}`
                - Significant: {'✅ Yes' if p_value < 0.05 else '❌ No'} (α=0.05)
                """)
                
                mean_recent = recent.mean()
                mean_historical = historical.mean()
                
                if p_value < 0.05:
                    if mean_recent > mean_historical:
                        st.success(f"📈 {test_param} significantly increased recently")
                    else:
                        st.warning(f"📉 {test_param} significantly decreased recently")
                else:
                    st.info(f"➡️ No significant change in {test_param}")
                
                # Comparison plot
                comparison_df = pd.DataFrame({
                    'Period': ['Recent (Last 30)'] * len(recent) + ['Historical (30-60 days ago)'] * len(historical),
                    'Value': list(recent) + list(historical)
                })
                
                fig = px.violin(comparison_df, x='Period', y='Value',
                               title=f'{test_param} - Time Period Comparison',
                               box=True, color='Period')
                st.plotly_chart(fig, use_container_width=True)


def page_ml(df):
    """Advanced ML Demo with multi-model training, evaluation, and explainability."""
    st.title("🧠 ML Demo — Advanced Model Training & Prediction")
    
    models_dir = Path(BASE_DIR) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize session state for active tab
    if 'ml_active_tab' not in st.session_state:
        st.session_state.ml_active_tab = 0
    
    # Tabs for different sections
    tab_train, tab_compare, tab_predict, tab_explain = st.tabs(
        ["🎯 Train Models", "📊 Model Comparison", "🔮 Make Predictions", "🔍 Explainability"]
    )
    
    # =====================================================================
    # TAB 1: TRAIN MODELS
    # =====================================================================
    with tab_train:
        st.markdown("### Model Training Control")
        
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox(
                "Select target variable:",
                options=['Turbidity', 'pH', 'Dissolved Oxygen', 'Specific Conductance'],
                key="target_select"
            )
        with col2:
            test_size = st.slider("Test set size (%):", 10, 40, 20, 5)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔵 Train RandomForest", use_container_width=True, key="train_rf"):
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    from sklearn.model_selection import train_test_split
                    
                    # Quick data prep with expanded features
                    progress = st.progress(0, "Preparing data...")
                    
                    # All available features
                    all_features = ['Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)', 
                                   'pH', 'Salinity', 'Specific Conductance', 'Turbidity',
                                   'Chlorophyll', 'Average Water Speed', 'Average Water Direction']
                    
                    # Filter to only features that exist (exclude target)
                    available_features = [f for f in all_features if f in df.columns and f != target]
                    
                    # Sample last 2000 rows first
                    df_subset = df.iloc[-2000:] if len(df) > 2000 else df.copy()
                    
                    # Fill missing values with median instead of dropping rows
                    df_clean = df_subset[[target] + available_features].copy()
                    df_clean = df_clean.dropna(subset=[target])  # Only drop if target is missing
                    
                    # Fill feature NaN with median
                    for feat in available_features:
                        if df_clean[feat].isna().any():
                            df_clean[feat] = df_clean[feat].fillna(df_clean[feat].median())
                    
                    X = df_clean[available_features].values
                    y = df_clean[target].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)
                    feature_names = available_features
                    
                    progress.progress(33, f"Data ready: {len(X_train)} samples, {len(feature_names)} features")
                    
                    if len(X_train) < 10:
                        st.error("Insufficient data for training (need >= 10 samples).")
                    else:
                        progress.progress(50, "Training model...")
                        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        progress.progress(75, "Calculating metrics...")
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        progress.progress(90, "Saving model...")
                        model_file = models_dir / f'rf_{target.lower().replace(" ", "_")}.pkl'
                        metrics_file = models_dir / f'rf_{target.lower().replace(" ", "_")}_metrics.txt'
                        
                        joblib.dump(model, model_file)
                        with open(metrics_file, 'w') as f:
                            f.write(f"RandomForest — {target}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}\nTrain: {len(X_train)} | Test: {len(X_test)}\n")
                        
                        progress.progress(100, "Complete!")
                        progress.empty()
                        st.success(f"✓ RandomForest trained!\n\n**RMSE:** {rmse:.4f} | **R²:** {r2:.4f} | **MAE:** {mae:.4f}\n\n📊 Check 'Saved Models' below or go to 'Make Predictions' tab to use it!")
                        st.balloons()
                except Exception as e:
                    st.error(f"Training error: {str(e)}\n\n{type(e).__name__}")

        
        with col2:
            if st.button("🟠 Train XGBoost", use_container_width=True, key="train_xgb"):
                try:
                    try:
                        import xgboost as xgb
                        HAS_XGB = True
                    except Exception:
                        HAS_XGB = False
                    
                    if not HAS_XGB:
                        st.info("ℹ️ XGBoost is not available in this deployment. Using Random Forest and LightGBM instead.")
                    else:
                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                        from sklearn.model_selection import train_test_split
                        
                        # Quick data prep with expanded features
                        progress = st.progress(0, "Preparing data...")
                        
                        # All available features
                        all_features = ['Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)', 
                                       'pH', 'Salinity', 'Specific Conductance', 'Turbidity',
                                       'Chlorophyll', 'Average Water Speed', 'Average Water Direction']
                        
                        # Filter to only features that exist (exclude target)
                        available_features = [f for f in all_features if f in df.columns and f != target]
                        
                        # Sample last 2000 rows first
                        df_subset = df.iloc[-2000:] if len(df) > 2000 else df.copy()
                        
                        # Fill missing values with median instead of dropping rows
                        df_clean = df_subset[[target] + available_features].copy()
                        df_clean = df_clean.dropna(subset=[target])  # Only drop if target is missing
                        
                        # Fill feature NaN with median
                        for feat in available_features:
                            if df_clean[feat].isna().any():
                                df_clean[feat] = df_clean[feat].fillna(df_clean[feat].median())
                        
                        X = df_clean[available_features].values
                        y = df_clean[target].values
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)
                        feature_names = available_features
                        
                        progress.progress(33, f"Data ready: {len(X_train)} samples, {len(feature_names)} features")
                        
                        if len(X_train) < 10:
                            st.error("Insufficient data for training.")
                        else:
                            progress.progress(50, "Training XGBoost...")
                            model = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42, verbosity=0, n_jobs=-1)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            progress.progress(75, "Calculating metrics...")
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            progress.progress(90, "Saving model...")
                            model_file = models_dir / f'xgb_{target.lower().replace(" ", "_")}.pkl'
                            metrics_file = models_dir / f'xgb_{target.lower().replace(" ", "_")}_metrics.txt'
                            
                            joblib.dump(model, model_file)
                            with open(metrics_file, 'w') as f:
                                f.write(f"XGBoost — {target}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}\nTrain: {len(X_train)} | Test: {len(X_test)}\n")
                            
                            progress.progress(100, "Complete!")
                            progress.empty()
                            st.success(f"✓ XGBoost trained!\n\n**RMSE:** {rmse:.4f} | **R²:** {r2:.4f} | **MAE:** {mae:.4f}\n\n📊 Check 'Saved Models' below or go to 'Make Predictions' tab to use it!")
                            st.balloons()
                except Exception as e:
                    st.error(f"Training error: {str(e)}\n\n{type(e).__name__}")

        
        with col3:
            if st.button("🟢 Train LightGBM", use_container_width=True, key="train_lgbm"):
                try:
                    try:
                        import lightgbm as lgb
                        HAS_LGB = True
                    except ImportError:
                        HAS_LGB = False
                    
                    if not HAS_LGB:
                        st.info("ℹ️ LightGBM is not available in this deployment. Using Random Forest instead.")
                    else:
                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                        from sklearn.model_selection import train_test_split
                        
                        # Quick data prep with expanded features
                        progress = st.progress(0, "Preparing data...")
                        
                        # All available features
                        all_features = ['Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)', 
                                       'pH', 'Salinity', 'Specific Conductance', 'Turbidity',
                                       'Chlorophyll', 'Average Water Speed', 'Average Water Direction']
                        
                        # Filter to only features that exist (exclude target)
                        available_features = [f for f in all_features if f in df.columns and f != target]
                        
                        # Sample last 2000 rows first
                        df_subset = df.iloc[-2000:] if len(df) > 2000 else df.copy()
                        
                        # Fill missing values with median instead of dropping rows
                        df_clean = df_subset[[target] + available_features].copy()
                        df_clean = df_clean.dropna(subset=[target])  # Only drop if target is missing
                        
                        # Fill feature NaN with median
                        for feat in available_features:
                            if df_clean[feat].isna().any():
                                df_clean[feat] = df_clean[feat].fillna(df_clean[feat].median())
                        
                        X = df_clean[available_features].values
                        y = df_clean[target].values
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, shuffle=False)
                        feature_names = available_features
                        
                        progress.progress(33, f"Data ready: {len(X_train)} samples, {len(feature_names)} features")
                        
                        if len(X_train) < 10:
                            st.error("Insufficient data for training.")
                        else:
                            progress.progress(50, "Training LightGBM...")
                            model = lgb.LGBMRegressor(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            progress.progress(75, "Calculating metrics...")
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            progress.progress(90, "Saving model...")
                            model_file = models_dir / f'lgbm_{target.lower().replace(" ", "_")}.pkl'
                            metrics_file = models_dir / f'lgbm_{target.lower().replace(" ", "_")}_metrics.txt'
                            
                            joblib.dump(model, model_file)
                            with open(metrics_file, 'w') as f:
                                f.write(f"LightGBM — {target}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}\nTrain: {len(X_train)} | Test: {len(X_test)}\n")
                            
                            progress.progress(100, "Complete!")
                            progress.empty()
                            st.success(f"✓ LightGBM trained!\n\n**RMSE:** {rmse:.4f} | **R²:** {r2:.4f} | **MAE:** {mae:.4f}\n\n📊 Check 'Saved Models' below or go to 'Make Predictions' tab to use it!")
                            st.balloons()
                except Exception as e:
                    st.error(f"Training error: {str(e)}\n\n{type(e).__name__}")

        
        st.divider()
        st.markdown("### Saved Models")
        
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.h5"))
        if model_files:
            for mf in sorted(model_files):
                metric_file = mf.parent / f"{mf.stem}_metrics.txt"
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📦 **{mf.name}**")
                    if metric_file.exists():
                        with open(metric_file) as f:
                            st.code(f.read(), language="text")
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{mf.name}", use_container_width=True):
                        mf.unlink()
                        if metric_file.exists():
                            metric_file.unlink()
                        st.rerun()
        else:
            st.info("No saved models yet. Train a model to get started.")
    
    # =====================================================================
    # TAB 2: MODEL COMPARISON
    # =====================================================================
    with tab_compare:
        st.markdown("### Compare Model Performance")
        
        # Load all saved models and their metrics
        metrics_dict = {}
        for metric_file in models_dir.glob("*_metrics.txt"):
            try:
                with open(metric_file) as f:
                    content = f.read()
                metrics_dict[metric_file.stem] = content
            except Exception:
                continue
        
        if metrics_dict:
            # Parse metrics and create comparison table
            comparison_data = []
            for model_name, metrics_text in metrics_dict.items():
                try:
                    lines = metrics_text.strip().split('\n')
                    header = lines[0]
                    
                    # Parse header
                    if ' — ' in header:
                        model_type, target_var = header.split(' — ')
                    else:
                        model_type = header
                        target_var = 'Unknown'
                    
                    # Parse metrics
                    metrics = {}
                    for line in lines[1:]:
                        if ':' in line:
                            k, v = line.split(':', 1)
                            try:
                                metrics[k.strip()] = float(v.strip().split()[0])  # Handle "value | ..." format
                            except (ValueError, IndexError):
                                pass
                    
                    if metrics.get('RMSE') or metrics.get('R²'):
                        comparison_data.append({
                            'Model': model_name,
                            'Type': model_type,
                            'Target': target_var,
                            'RMSE': metrics.get('RMSE', np.nan),
                            'R²': metrics.get('R²', np.nan),
                            'MAE': metrics.get('MAE', np.nan)
                        })
                except Exception:
                    continue
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    if 'RMSE' in comp_df.columns and comp_df['RMSE'].notna().any():
                        fig_rmse = px.bar(comp_df, x='Model', y='RMSE', color='Type', 
                                          title='RMSE Comparison (Lower is Better)',
                                          hover_data=['Target'])
                        st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col2:
                    if 'R²' in comp_df.columns and comp_df['R²'].notna().any():
                        fig_r2 = px.bar(comp_df, x='Model', y='R²', color='Type',
                                        title='R² Score Comparison (Higher is Better)',
                                        hover_data=['Target'])
                        st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.info("No valid metrics found. Train models first.")
        else:
            st.info("No models trained yet. Go to 'Train Models' tab to train models.")

    
    # =====================================================================
    # TAB 3: MAKE PREDICTIONS
    # =====================================================================
    with tab_predict:
        st.markdown("### Interactive Predictions")
        
        # Model selection
        pkl_models = list(models_dir.glob("*.pkl"))
        if pkl_models:
            model_choice = st.selectbox("Select a trained model:", [m.stem for m in pkl_models])
            model = joblib.load(models_dir / f"{model_choice}.pkl")
            
            st.markdown("#### Input Parameters")
            
            # Get feature names from the model
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, 'n_features_in_'):
                # Fallback to all available features
                all_features = ['Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)', 
                               'pH', 'Salinity', 'Specific Conductance', 'Turbidity',
                               'Chlorophyll', 'Average Water Speed', 'Average Water Direction']
                feature_names = [f for f in all_features if f in df.columns][:model.n_features_in_]
            else:
                st.error("Cannot determine model features")
                feature_names = []
            
            # Create sliders for each feature
            input_vals = {}
            cols = st.columns(3)
            for i, feat in enumerate(feature_names):
                with cols[i % 3]:
                    if feat in df.columns and df[feat].dtype.kind in 'fiu':
                        lo = float(df[feat].quantile(0.05))
                        hi = float(df[feat].quantile(0.95))
                        mid = float(df[feat].median())
                        # Ensure min < max to avoid slider errors
                        if lo >= hi:
                            lo, hi = float(df[feat].min()), float(df[feat].max())
                        if lo >= hi:
                            lo, hi = mid - 1, mid + 1
                        input_vals[feat] = st.slider(f"{feat}", min_value=lo, max_value=hi, value=mid, key=f"pred_{feat}")
                    else:
                        input_vals[feat] = st.number_input(f"{feat}", value=0.0, key=f"pred_{feat}")
            
            # Make prediction
            if st.button("🔮 Predict", use_container_width=True):
                try:
                    # Ensure features are in correct order
                    sample_df = pd.DataFrame([input_vals])
                    # Reorder columns to match model's expected feature order
                    sample_df = sample_df[feature_names]
                    prediction = model.predict(sample_df)[0]
                    
                    st.success(f"### Prediction Result")
                    st.metric("Predicted Value", f"{prediction:.3f}", delta=f"Based on {len(feature_names)} features")
                    
                    # Show input summary
                    st.markdown("#### Input Summary")
                    st.dataframe(sample_df.T, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.info("No trained models available. Train a model first in the 'Train Models' tab.")
    
    # =====================================================================
    # TAB 4: EXPLAINABILITY (SHAP)
    # =====================================================================
    with tab_explain:
        st.markdown("### Model Explainability & Insights")
        
        pkl_models = list(models_dir.glob("*.pkl"))
        if pkl_models:
            model_choice = st.selectbox("Select model for explainability:", [m.stem for m in pkl_models], key="explain_model")
            model = joblib.load(models_dir / f"{model_choice}.pkl")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                try:
                    feature_importance = model.feature_importances_
                    # Get feature names from model
                    if hasattr(model, 'feature_names_in_'):
                        feature_names = list(model.feature_names_in_)
                    else:
                        # Fallback
                        all_features = ['Temperature', 'Dissolved Oxygen', 'Dissolved Oxygen (%Saturation)', 
                                       'pH', 'Salinity', 'Specific Conductance', 'Turbidity',
                                       'Chlorophyll', 'Average Water Speed', 'Average Water Direction']
                        feature_names = all_features[:len(feature_importance)]
                    
                    # Create importance dataframe
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(feature_importance)],
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=False)
                    
                    # Add percentage contribution
                    importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
                    
                    # Main visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                    title='Feature Importance Scores',
                                    color='Importance', color_continuous_scale='Viridis',
                                    text='Percentage')
                        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Pie chart - dynamically adjust based on number of features
                        num_features = len(importance_df)
                        top_n = min(5, num_features)
                        top_features_pie = importance_df.head(top_n)
                        title = f'Top {top_n} Feature{"s" if top_n > 1 else ""}' if top_n < num_features else 'All Features'
                        fig_pie = px.pie(top_features_pie, values='Importance', names='Feature',
                                        title=title)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Feature importance table
                    st.markdown("#### 📊 Detailed Feature Importance")
                    display_df = importance_df.copy()
                    display_df['Importance'] = display_df['Importance'].round(4)
                    display_df['Percentage'] = display_df['Percentage'].round(2).astype(str) + '%'
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Key insights
                    st.markdown("---")
                    top_features = importance_df.head(3)['Feature'].tolist()
                    top_importance = importance_df.head(3)['Importance'].tolist()
                    top_percentage = importance_df.head(3)['Percentage'].tolist()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🥇 Most Important", top_features[0], f"{top_percentage[0]:.1f}%")
                    with col2:
                        st.metric("🥈 Second Most", top_features[1] if len(top_features) > 1 else "N/A", 
                                 f"{top_percentage[1]:.1f}%" if len(top_features) > 1 else "")
                    with col3:
                        st.metric("🥉 Third Most", top_features[2] if len(top_features) > 2 else "N/A",
                                 f"{top_percentage[2]:.1f}%" if len(top_features) > 2 else "")
                    
                    # Interpretation
                    st.markdown("#### 💡 Key Insights")
                    cumulative_top3 = sum(top_percentage[:3])
                    insights = f"""
                    - The **top 3 features** account for **{cumulative_top3:.1f}%** of the model's decision-making
                    - **{top_features[0]}** is the dominant predictor with {top_percentage[0]:.1f}% importance
                    - Model uses all {len(feature_names)} features, but some contribute more than others
                    """
                    st.info(insights)
                    
                    # Feature correlations with target (if available)
                    st.markdown("#### 🔗 Feature Statistics")
                    stats_data = []
                    for feat in feature_names:
                        if feat in df.columns:
                            stats_data.append({
                                'Feature': feat,
                                'Mean': df[feat].mean(),
                                'Std Dev': df[feat].std(),
                                'Min': df[feat].min(),
                                'Max': df[feat].max()
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        for col in ['Mean', 'Std Dev', 'Min', 'Max']:
                            stats_df[col] = stats_df[col].round(3)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error computing feature importance: {e}")
            else:
                st.info("Feature importance not available for this model type.")
        else:
            st.info("No trained models available. Train a model first in the 'Train Models' tab.")


def page_data(df):
    """Enhanced data explorer with statistics, filtering, and visualizations."""
    st.title("📁 Data Explorer & Statistics")
    
    # Summary statistics at the top
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        date_range = (df['Timestamp'].max() - df['Timestamp'].min()).days
        st.metric("Date Range (days)", f"{date_range:,}")
    with col4:
        missing_pct = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Missing Data %", f"{missing_pct:.1f}%")
    
    # Data Quality Report
    st.markdown("---")
    st.markdown("### 🔍 Data Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values by Column")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isna().sum().values,
            'Missing %': (df.isna().sum().values / len(df) * 100).round(2)
        }).sort_values('Missing', ascending=False)
        missing_data = missing_data[missing_data['Missing'] > 0]
        
        if len(missing_data) > 0:
            st.dataframe(missing_data, use_container_width=True, hide_index=True)
        else:
            st.success("✓ No missing values detected!")
    
    with col2:
        st.markdown("#### Data Types")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index.astype(str), 
                     title='Column Data Types')
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Summary
    st.markdown("---")
    st.markdown("### 📈 Statistical Summary (Numeric Columns)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary_stats = df[numeric_cols].describe().T
        summary_stats['median'] = df[numeric_cols].median()
        summary_stats = summary_stats[['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']]
        summary_stats = summary_stats.round(3)
        st.dataframe(summary_stats, use_container_width=True)
    
    # Filter and search
    st.markdown("---")
    st.markdown("### 🔎 Filter & Search Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Date range filter
        if 'Timestamp' in df.columns:
            min_date = df['Timestamp'].min().date()
            max_date = df['Timestamp'].max().date()
            date_range = st.date_input(
                "Filter by date range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                df_filtered = df[(df['Timestamp'].dt.date >= date_range[0]) & 
                                (df['Timestamp'].dt.date <= date_range[1])]
            else:
                df_filtered = df
        else:
            df_filtered = df
    
    with col2:
        # Row limit
        row_limit = st.selectbox("Rows to display:", [100, 500, 1000, 5000, "All"])
    
    with col3:
        # Show all columns toggle
        show_all_cols = st.checkbox("Show all columns", value=False)
    
    # Column selector (full width, below the filters)
    if not show_all_cols:
        # Default: show key columns
        key_cols = ['Timestamp', 'Temperature', 'Dissolved Oxygen', 'pH', 
                   'Salinity', 'Specific Conductance', 'Turbidity', 'Chlorophyll']
        default_cols = [col for col in key_cols if col in df.columns][:8]
        
        selected_cols = st.multiselect(
            "Select specific columns (or check 'Show all columns' above):",
            options=df.columns.tolist(),
            default=default_cols
        )
        if selected_cols:
            df_display = df_filtered[selected_cols]
        else:
            df_display = df_filtered[default_cols]
    else:
        df_display = df_filtered
        selected_cols = df.columns.tolist()
    
    # Apply row limit
    if row_limit != "All":
        df_display = df_display.head(row_limit)
    
    st.markdown(f"**Showing {len(df_display):,} of {len(df):,} rows**")
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Download options
    st.markdown("---")
    st.markdown("### 💾 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📥 Download Full Dataset (CSV)",
            df.to_csv(index=False),
            file_name='water_quality_full.csv',
            mime='text/csv'
        )
    
    with col2:
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            df_display.to_csv(index=False),
            file_name='water_quality_filtered.csv',
            mime='text/csv'
        )
    
    with col3:
        st.download_button(
            "📥 Download Statistics (CSV)",
            summary_stats.to_csv() if len(numeric_cols) > 0 else "No numeric data",
            file_name='water_quality_statistics.csv',
            mime='text/csv'
        )
    
    # Quick visualizations
    st.markdown("---")
    st.markdown("### 📊 Quick Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Distribution plot
        st.markdown("#### Distribution Plot")
        numeric_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_list:
            selected_col = st.selectbox("Select parameter:", numeric_cols_list, key="dist_plot")
            fig = px.histogram(df, x=selected_col, nbins=50, title=f'{selected_col} Distribution',
                              marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Time series plot
        st.markdown("#### Time Series")
        if 'Timestamp' in df.columns and numeric_cols_list:
            selected_ts = st.selectbox("Select parameter:", numeric_cols_list, key="ts_plot")
            fig = px.line(df.sort_values('Timestamp'), x='Timestamp', y=selected_ts, 
                         title=f'{selected_ts} Over Time')
            st.plotly_chart(fig, use_container_width=True)


def page_insights(df):
    """Automated insights and anomaly alerts."""
    st.title("📊 Insights & Intelligence")
    
    # Calculate key metrics
    recent_days = 7
    recent_df = df[df['Timestamp'] >= df['Timestamp'].max() - timedelta(days=recent_days)]
    
    if len(recent_df) == 0:
        st.warning("No recent data available")
        return
    
    # ===== HEALTH SCORE DASHBOARD =====
    st.subheader("🎯 Water Quality Health Score")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Overall health score (0-100)
        excellent_pct = (df['WQI_Class'] == 'Excellent').sum() / len(df) * 100
        good_pct = (df['WQI_Class'] == 'Good').sum() / len(df) * 100
        health_score = excellent_pct * 1.0 + good_pct * 0.8
        
        score_color = '#10b981' if health_score >= 80 else '#3b82f6' if health_score >= 60 else '#f59e0b' if health_score >= 40 else '#ef4444'
        st.markdown(f"""
        <div style="background: white; padding: 30px; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 3.5em; font-weight: 700; color: {score_color}; margin-bottom: 10px;">
                {health_score:.0f}
            </div>
            <div style="font-size: 1.1em; color: #64748b; font-weight: 500;">Overall Health Score</div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                <div style="font-size: 0.9em; color: #64748b;">Based on {len(df):,} measurements</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Compliance rate
        compliance = ((df['WQI'] >= 60).sum() / len(df) * 100) if len(df) > 0 else 0
        st.markdown(f"""
        <div style="background: white; padding: 30px; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 3.5em; font-weight: 700; color: #3b82f6; margin-bottom: 10px;">
                {compliance:.0f}%
            </div>
            <div style="font-size: 1.1em; color: #64748b; font-weight: 500;">Compliance Rate</div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                <div style="font-size: 0.9em; color: #64748b;">WQI ≥ 60 (Good or better)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Stability index (lower std = more stable)
        wqi_std = df['WQI'].std()
        stability = max(0, 100 - (wqi_std * 2))  # Lower std = higher stability
        st.markdown(f"""
        <div style="background: white; padding: 30px; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.08);">
            <div style="font-size: 3.5em; font-weight: 700; color: #8b5cf6; margin-bottom: 10px;">
                {stability:.0f}%
            </div>
            <div style="font-size: 1.1em; color: #64748b; font-weight: 500;">Stability Index</div>
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
                <div style="font-size: 0.9em; color: #64748b;">σ = {wqi_std:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== PARAMETER HEALTH CARDS =====
    st.subheader("🔬 Parameter Health Status")
    
    param_health = []
    
    # pH Health
    if 'pH' in df.columns:
        ph_mean = df['pH'].mean()
        ph_in_range = ((df['pH'] >= 6.5) & (df['pH'] <= 8.5)).sum() / len(df) * 100
        param_health.append({
            'name': 'pH Level',
            'icon': '⚗️',
            'value': f'{ph_mean:.2f}',
            'health': ph_in_range,
            'status': 'Optimal' if ph_in_range >= 80 else 'Acceptable' if ph_in_range >= 60 else 'Concerning'
        })
    
    # Dissolved Oxygen
    do_cols = [c for c in df.columns if 'Dissolved Oxygen' in c and '%' not in c and '_' not in c]
    if do_cols:
        do_mean = df[do_cols[0]].mean()
        do_health = ((df[do_cols[0]] >= 5).sum() / len(df) * 100) if len(df) > 0 else 0
        param_health.append({
            'name': 'Dissolved O₂',
            'icon': '🫧',
            'value': f'{do_mean:.2f}',
            'health': do_health,
            'status': 'Good' if do_health >= 70 else 'Moderate' if do_health >= 50 else 'Low'
        })
    
    # Turbidity
    if 'Turbidity' in df.columns:
        turb_mean = df['Turbidity'].mean()
        turb_health = ((df['Turbidity'] <= 5).sum() / len(df) * 100) if len(df) > 0 else 0
        param_health.append({
            'name': 'Turbidity',
            'icon': '🌊',
            'value': f'{turb_mean:.2f}',
            'health': turb_health,
            'status': 'Clear' if turb_health >= 70 else 'Moderate' if turb_health >= 50 else 'Cloudy'
        })
    
    # Conductivity
    if 'Specific Conductance' in df.columns:
        sc_mean = df['Specific Conductance'].mean()
        sc_health = ((df['Specific Conductance'] <= 1500).sum() / len(df) * 100) if len(df) > 0 else 0
        param_health.append({
            'name': 'Conductivity',
            'icon': '⚡',
            'value': f'{sc_mean:.0f}',
            'health': sc_health,
            'status': 'Normal' if sc_health >= 70 else 'Elevated' if sc_health >= 50 else 'High'
        })
    
    # Display parameter cards
    if param_health:
        cols = st.columns(len(param_health))
        for col, param in zip(cols, param_health):
            with col:
                health_color = '#10b981' if param['health'] >= 70 else '#f59e0b' if param['health'] >= 50 else '#ef4444'
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 10px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center;">
                    <div style="font-size: 2.5em; margin-bottom: 10px;">{param['icon']}</div>
                    <div style="font-size: 0.9em; color: #64748b; margin-bottom: 8px;">{param['name']}</div>
                    <div style="font-size: 1.8em; font-weight: 700; color: #1e293b; margin-bottom: 8px;">{param['value']}</div>
                    <div style="background: {health_color}; color: white; padding: 4px 12px; 
                                border-radius: 20px; font-size: 0.85em; font-weight: 600; display: inline-block;">
                        {param['status']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== WQI ALERTS =====
    st.subheader("🚨 WQI Status & Alerts")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_wqi = df['WQI'].iloc[-1] if not df.empty else 0
        avg_wqi = recent_df['WQI'].mean()
        color = 'green' if current_wqi >= 80 else 'blue' if current_wqi >= 60 else 'orange' if current_wqi >= 40 else 'red'
        st.metric("Current WQI", f"{current_wqi:.1f}", delta=f"{current_wqi - avg_wqi:.1f}")
    
    with col2:
        min_wqi_recent = recent_df['WQI'].min()
        max_wqi_recent = recent_df['WQI'].max()
        st.metric("7-day Range", f"{min_wqi_recent:.1f}-{max_wqi_recent:.1f}", delta=f"Δ {max_wqi_recent - min_wqi_recent:.1f}")
    
    with col3:
        excellent_pct = (recent_df['WQI_Class'] == 'Excellent').sum() / len(recent_df) * 100
        good_pct = (recent_df['WQI_Class'] == 'Good').sum() / len(recent_df) * 100
        st.metric("Excellent%", f"{excellent_pct:.1f}%", delta=f"Good: {good_pct:.1f}%")
    
    with col4:
        poor_pct = (recent_df['WQI_Class'].isin(['Poor', 'Very Poor'])).sum() / len(recent_df) * 100
        st.metric("Alert Days%", f"{poor_pct:.1f}%", delta="Below threshold")
    
    # Anomalies
    st.markdown("#### Detected Anomalies")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies_found = False
    
    for col in numeric_cols[:5]:  # Check first 5 numeric columns
        if col in ['WQI', 'Timestamp'] or 'quality' in col.lower() or '_Q' in col or 'Contribution' in col:
            continue
        
        mean = df[col].mean()
        std = df[col].std()
        threshold = mean + 3 * std
        
        anomaly_mask = df[col] > threshold
        if anomaly_mask.any():
            anomalies_found = True
            count = anomaly_mask.sum()
            st.warning(f"⚠️ **{col}**: {count} extreme values detected (>{threshold:.2f})")
    
    if not anomalies_found:
        st.success("✓ No major anomalies detected in recent data")
    
    # ===== TREND ANALYSIS =====
    st.markdown("#### Trend Analysis (7-day)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wqi_trend = recent_df['WQI'].iloc[-1] - recent_df['WQI'].iloc[0]
        trend_dir = "📈 Improving" if wqi_trend > 0 else "📉 Declining" if wqi_trend < 0 else "➡️ Stable"
        st.info(f"**WQI Trend**: {trend_dir}\nChange: {wqi_trend:.2f} points")
    
    with col2:
        turb_cols = [c for c in df.columns if 'Turbidity' in c and '_' not in c]
        if turb_cols:
            turb_col = turb_cols[0]
            turb_recent = recent_df[turb_col]
            turb_trend = turb_recent.iloc[-1] - turb_recent.iloc[0]
            st.info(f"**Turbidity**: {'Decreasing' if turb_trend < 0 else 'Increasing' if turb_trend > 0 else 'Stable'}\nChange: {turb_trend:.2f}")
    
    with col3:
        ph_cols = [c for c in df.columns if c == 'pH']
        if ph_cols:
            ph_col = ph_cols[0]
            ph_recent = recent_df[ph_col]
            ph_mean = ph_recent.mean()
            optimal = abs(ph_mean - 7.0) < 0.5
            status = "✓ Optimal" if optimal else "⚠️ Off-range"
            st.info(f"**pH Balance**: {status}\nAvg: {ph_mean:.2f}")
    
    # ===== RECOMMENDATIONS =====
    st.markdown("#### 💡 Recommendations")
    
    recommendations = []
    
    # WQI-based
    if current_wqi < 40:
        recommendations.append("🔴 **URGENT**: Water quality is Very Poor. Immediate intervention required.")
    elif current_wqi < 60:
        recommendations.append("🟠 **WARNING**: Water quality below Good threshold. Increase monitoring frequency.")
    
    # Turbidity-based
    if turb_cols:
        turb_col = turb_cols[0]
        turb_mean = recent_df[turb_col].mean()
        if turb_mean > 5:
            recommendations.append("🔹 High turbidity detected. Consider implementing sediment filtration.")
    
    # pH-based
    if ph_cols:
        ph_col = ph_cols[0]
        ph_mean = recent_df[ph_col].mean()
        if ph_mean < 6.5:
            recommendations.append("🔹 Acidic conditions detected. Add alkaline buffers to stabilize pH.")
        elif ph_mean > 8.0:
            recommendations.append("🔹 Basic conditions detected. Consider pH adjustment through aeration.")
    
    # Temporal
    if wqi_trend < -5:
        recommendations.append("🔹 Declining water quality trend. Schedule investigation for pollution sources.")
    elif wqi_trend > 5:
        recommendations.append("✅ Water quality improving! Continue current treatment protocols.")
    
    if not recommendations:
        recommendations.append("✅ Water quality is stable and within acceptable ranges.")
    
    for rec in recommendations:
        st.write(rec)
    
    # ===== STATISTICS SUMMARY =====
    st.markdown("#### 📈 7-Day Statistics")
    
    stats_summary = pd.DataFrame({
        'Parameter': [],
        'Mean': [],
        'Min': [],
        'Max': [],
        'Std Dev': []
    })
    
    key_params = [c for c in recent_df.columns if c not in ['Timestamp', 'WQI', 'WQI_Class', 'DayOfWeek', 'Season', 'month', 'season', 'Year', 'Hour', 'IsWeekend', 'date'] and '_' not in c][:6]
    
    rows = []
    for param in key_params:
        if param in recent_df.columns and recent_df[param].dtype.kind in 'fiu':
            rows.append({
                'Parameter': param,
                'Mean': f"{recent_df[param].mean():.2f}",
                'Min': f"{recent_df[param].min():.2f}",
                'Max': f"{recent_df[param].max():.2f}",
                'Std Dev': f"{recent_df[param].std():.2f}"
            })
    
    stats_df = pd.DataFrame(rows)
    st.dataframe(stats_df, use_container_width=True)
    
    # ===== ADVANCED VISUALIZATIONS FOR DATA VIZ PROJECT =====
    st.markdown("---")
    st.markdown("### 🎨 Advanced Visualization Showcase")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Bullet Chart for Performance vs Target
        st.markdown("#### 🎯 Performance Bullet Chart")
        st.caption("Compare current performance against targets")
        
        params_for_bullet = ['pH', 'Dissolved Oxygen', 'Turbidity']
        params_available = [p for p in params_for_bullet if p in df.columns]
        
        if len(params_available) > 0:
            fig_bullet = go.Figure()
            
            for i, param in enumerate(params_available[:3]):
                current_val = recent_df[param].mean()
                
                # Define targets based on parameter
                if param == 'pH':
                    target = 7.0
                    good_range = [6.5, 8.5]
                    acceptable_range = [6.0, 9.0]
                elif param == 'Dissolved Oxygen':
                    target = 8.0
                    good_range = [6.0, 10.0]
                    acceptable_range = [4.0, 12.0]
                else:  # Turbidity
                    target = 2.0
                    good_range = [0, 5.0]
                    acceptable_range = [0, 10.0]
                
                fig_bullet.add_trace(go.Indicator(
                    mode="number+gauge+delta",
                    value=current_val,
                    delta={'reference': target},
                    title={'text': param},
                    domain={'x': [0.1, 1], 'y': [i * 0.33, (i + 1) * 0.33]},
                    gauge={
                        'shape': "bullet",
                        'axis': {'range': [None, acceptable_range[1]]},
                        'threshold': {
                            'line': {'color': "red", 'width': 2},
                            'thickness': 0.75,
                            'value': target
                        },
                        'steps': [
                            {'range': acceptable_range, 'color': "#fef3c7"},
                            {'range': good_range, 'color': "#dcfce7"}
                        ],
                        'bar': {'color': "#667eea"}
                    }
                ))
            
            fig_bullet.update_layout(height=350, margin=dict(l=100, r=50, t=30, b=30))
            st.plotly_chart(fig_bullet, use_container_width=True)
    
    with viz_col2:
        # Treemap for Data Coverage
        st.markdown("#### 🌳 Data Coverage Treemap")
        st.caption("Hierarchical view of data completeness")
        
        # Calculate completeness for each parameter
        numeric_params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number']][:10]
        
        treemap_data = []
        for param in numeric_params:
            completeness = (1 - df[param].isna().sum() / len(df)) * 100
            category = 'Excellent' if completeness >= 95 else 'Good' if completeness >= 80 else 'Fair' if completeness >= 60 else 'Poor'
            treemap_data.append({
                'Parameter': param,
                'Category': category,
                'Completeness': completeness,
                'Count': df[param].notna().sum()
            })
        
        if len(treemap_data) > 0:
            treemap_df = pd.DataFrame(treemap_data)
            
            fig_treemap = px.treemap(
                treemap_df,
                path=['Category', 'Parameter'],
                values='Count',
                color='Completeness',
                color_continuous_scale='RdYlGn',
                title='',
                hover_data={'Completeness': ':.1f'}
            )
            fig_treemap.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_treemap, use_container_width=True)
    
    # Ridgeline Plot (using multiple violin plots)
    st.markdown("#### 🏔️ Parameter Distribution Ridge Plot")
    st.caption("Compare distributions of multiple parameters over time")
    
    params_for_ridge = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number', 'WQI']][:5]
    
    if len(params_for_ridge) >= 3:
        df_ridge = df.copy()
        df_ridge['Month'] = df_ridge['Timestamp'].dt.to_period('M').astype(str)
        recent_months = df_ridge['Month'].unique()[-4:]
        
        fig_ridge = go.Figure()
        
        for i, param in enumerate(params_for_ridge[:4]):
            for j, month in enumerate(recent_months):
                month_data = df_ridge[df_ridge['Month'] == month][param].dropna()
                
                if len(month_data) > 5:
                    fig_ridge.add_trace(go.Violin(
                        x=month_data,
                        y=[f"{param}<br>{month}"] * len(month_data),
                        name=f"{param}-{month}",
                        orientation='h',
                        side='positive',
                        line_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)],
                        showlegend=False
                    ))
        
        fig_ridge.update_layout(
            title="Parameter Distributions Across Recent Months",
            xaxis_title="Value",
            yaxis_title="",
            height=500,
            violingap=0,
            violinmode='overlay'
        )
        st.plotly_chart(fig_ridge, use_container_width=True)
    
    # Contour Density Plot
    st.markdown("#### 🗺️ 2D Density Contour Plot")
    st.caption("Explore relationship density between two parameters")
    
    col1, col2 = st.columns(2)
    numeric_params = [c for c in df.columns if df[c].dtype.kind in 'fiu' and c not in ['Record number']]
    
    with col1:
        x_contour = st.selectbox("X-axis parameter:", numeric_params, index=0, key="contour_x")
    with col2:
        y_contour = st.selectbox("Y-axis parameter:", numeric_params, index=1 if len(numeric_params) > 1 else 0, key="contour_y")
    
    if x_contour != y_contour:
        df_contour = df[[x_contour, y_contour]].dropna().sample(min(1000, len(df)))
        
        fig_contour = go.Figure()
        
        # Add scatter
        fig_contour.add_trace(go.Scatter(
            x=df_contour[x_contour],
            y=df_contour[y_contour],
            mode='markers',
            marker=dict(size=3, color='rgba(102, 126, 234, 0.5)'),
            name='Data Points'
        ))
        
        # Add contour
        fig_contour.add_trace(go.Histogram2dContour(
            x=df_contour[x_contour],
            y=df_contour[y_contour],
            colorscale='Viridis',
            showscale=True,
            name='Density'
        ))
        
        fig_contour.update_layout(
            title=f"Density Contour: {x_contour} vs {y_contour}",
            xaxis_title=x_contour,
            yaxis_title=y_contour,
            height=450
        )
        st.plotly_chart(fig_contour, use_container_width=True)


def main():
    # Initialize session state before any widgets are created (prevents warning)
    if 'use_synthetic_rainfall' not in st.session_state:
        st.session_state['use_synthetic_rainfall'] = False  # Disabled for performance
    
    # Initialize active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    try:
        df = load_and_process(use_synthetic=False)  # Force disable synthetic for speed
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    # Render compact top navigation
    try:
        render_top_nav(df)
    except Exception:
        pass
    
    # Add synthetic rainfall toggle in sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.checkbox(
            "Use synthetic rainfall",
            key='use_synthetic_rainfall',
            help="Toggle synthetic rainfall when data doesn't overlap",
        )
        st.markdown("---")
    
    # Top navigation with session state tracking
    tab_names = ["Home", "Explore", "Analysis", "Insights", "ML Demo", "Data"]
    icons = {
        "Home": "🏠",
        "Explore": "🔍",
        "Analysis": "🧭",
        "Insights": "💡",
        "ML Demo": "🧠",
        "Data": "📁",
    }

    # Professional navigation styling
    st.markdown("""
    <style>
        /* Hide default streamlit header elements */
        header[data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* Hide radio button label */
        div[data-testid="stRadio"] > label:first-child {
            display: none !important;
        }
        
        /* Navigation container */
        div[data-testid="stRadio"] {
            background: transparent;
            padding: 0;
            margin: 0 0 30px 0;
            border: none;
            box-shadow: none;
        }
        
        div[data-testid="stRadio"] > div {
            flex-direction: row;
            gap: 8px;
            justify-content: flex-start;
            background: white;
            padding: 8px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }
        
        /* Navigation buttons */
        div[data-testid="stRadio"] label {
            background: transparent !important;
            color: #64748b !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            border: none !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            box-shadow: none !important;
        }
        
        div[data-testid="stRadio"] label:hover {
            background: #f1f5f9 !important;
            color: #334155 !important;
        }
        
        /* Active/selected tab */
        div[data-testid="stRadio"] label[data-checked="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25) !important;
        }
        
        div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation radio buttons (styled to look like header tabs)
    selected_tab = st.radio(
        "nav",
        tab_names,
        index=st.session_state.active_tab,
        horizontal=True,
        label_visibility="collapsed",
        key="main_tab_selector",
        format_func=lambda x: f"{icons[x]} {x}"
    )
    
    # Update active tab based on selection
    st.session_state.active_tab = tab_names.index(selected_tab)
    
    # Render the selected page
    if selected_tab == "Home":
        page_home(df)
    elif selected_tab == "Explore":
        page_explore(df)
    elif selected_tab == "Analysis":
        page_analysis(df)
    elif selected_tab == "Insights":
        page_insights(df)
    elif selected_tab == "ML Demo":
        page_ml(df)
    elif selected_tab == "Data":
        page_data(df)


if __name__ == '__main__':
    main()

