# WaterWatch — Water Quality Analytics Dashboard

> *Interactive monitoring and storytelling for water quality management.*

A Streamlit-based analytics platform for exploring water quality trends, identifying pollution patterns, and forecasting turbidity using machine learning. Built with a Nykaa-inspired product design for intuitive navigation and beautiful visualizations.

---

## 🎯 Project Overview

**Goal**: Analyze, monitor, and visualize pollution trends and water quality variations across time, providing insights to support sustainable water resource management.

**Target Users**:
- Environmental scientists & water quality managers
- Policy makers & sustainability officers
- Data analysts & engineers
- Public health agencies

**Key Deliverables**:
- Interactive web dashboard (Streamlit)
- Statistical analysis (correlations, trends, seasonal patterns)
- ML-based forecasting (turbidity prediction)
- Exportable insights & reports

---

## 📊 Features

### Current (v1)

✅ **Home Page**
- Hero banner (Nykaa-inspired gradient)
- Top polluted sites as product-style cards
- Quick KPIs (records count, avg WQI, avg turbidity, avg rainfall)

✅ **Explore Page**
- Parameter selection (multi-select)
- Date range filtering
- Time-series visualization with rolling mean
- Distribution histograms and boxplots

✅ **Analysis Page**
- WQI contributions breakdown (stacked area chart)
- Spearman correlation heatmap with pairwise non-missing counts
- Automated guidance when data is insufficient

✅ **ML Demo Page**
- RandomForest model training on historical data
- Turbidity prediction with interactive sliders
- Model performance metrics (RMSE)

✅ **Data Explorer**
- Full dataset table view (up to 1000 rows)
- CSV download functionality

### Upcoming (v2)

🔄 **EDA Notebook** — Reproducible analysis with summary stats, missingness maps, seasonal decomposition, and distribution plots.

🔄 **Preprocessing Module** — Centralized `src/preprocessing.py` for unit standardization, aggregation, and WQI calculation.

🔄 **Insights & Alerts** — Auto-generated narrative pages highlighting worsening trends, anomalies, and actionable recommendations.

🔄 **Advanced Forecasting** — XGBoost/LSTM models with cross-validation and SHAP explainability.

🔄 **PDF Report Export** — Automated report generation with charts, summary statistics, and conclusions.

🔄 **API & Deployment** — FastAPI backend + Docker + Streamlit Cloud deployment.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- `pip` or `conda`

### Installation

```bash
# Clone or navigate to project directory
cd /Users/raga/Desktop/water_quality_project

# Create and activate virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
water_quality_project/
├── app.py                      # Main Streamlit app (all pages + logic)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   ├── water_quality.csv       # Primary sensor dataset (30,894 rows)
│   ├── rainfall.csv            # Daily weather data (54 rows)
│   ├── processed.csv           # (Generated) Preprocessed dataset
│   └── README.md               # Data inventory & validation
│
├── src/
│   └── preprocessing.py        # (Planned) Centralized preprocessing pipeline
│
├── notebooks/
│   └── eda.ipynb               # (Planned) Exploratory data analysis
│
└── .venv/                      # Virtual environment (created by pip)
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.50+ |
| **Data Processing** | Pandas 2.3+, NumPy 2.0+ |
| **Visualization** | Plotly 5.10+, Seaborn 0.13+, Altair 5.0+ |
| **Statistical Analysis** | SciPy 1.13+, StatsModels 0.14+ |
| **Machine Learning** | Scikit-learn 1.6+ (RandomForest) |
| **Backend** | Python 3.9 |
| **Deployment** | Streamlit Cloud (planned) |

---

## 📊 Data Summary

### Water Quality (`data/water_quality.csv`)
- **30,894 records** spanning ~5 months (Jan–May 2024)
- **20 columns** including physicochemical parameters:
  - pH, Turbidity, Specific Conductance, Salinity
  - Dissolved Oxygen (mg/L and % saturation)
  - Temperature, Chlorophyll
  - Each parameter includes a quality flag
- **Temporal**: Sub-hourly to hourly measurements
- **Missingness**: 3–19% depending on parameter (sensor downtime)

### Rainfall (`data/rainfall.csv`)
- **54 daily records** (subset of water quality date range)
- **6 columns**: date, rainfall (mm), temperature, humidity, wind_speed, weather_condition
- **Complete data** (no nulls) — suitable for daily aggregation

### Key Constraint
⚠️ **No site/location identifier** — All water quality records are aggregated from a single monitoring point. Geographic analysis not possible without external data.

For full data validation details, see `data/README.md`.

---

## 📈 Water Quality Index (WQI)

The app computes a **weighted WQI** combining four parameters:

| Parameter | Ideal | Standard | Weight |
|-----------|-------|----------|--------|
| pH | 7.0 | 8.5 | 25% |
| Turbidity (NTU) | 0.0 | 5.0 | 35% |
| Specific Conductance (µS/cm) | 0 | 1500 | 25% |
| Dissolved Oxygen (mg/L) | 8.0 | 4.0 | 15% |

**WQI Classification**:
- **Excellent**: 0–25 (green) ✓
- **Good**: 26–50 (blue)
- **Poor**: 51–75 (orange) ⚠️
- **Very Poor**: 76–100+ (red) ⛔

---

## 🔍 Key Analysis Features

### Correlations
- **Spearman correlation** (robust to non-linear relationships)
- **Pairwise-complete** analysis (handles missing data)
- Heatmap with correlation matrix and pair counts

### Time-Series
- Daily resampling for trend clarity
- 7-day rolling mean for smoothing
- Interactive Plotly charts with zoom/pan

### Distributions
- Histograms with marginal boxplots
- Identify outliers and data gaps
- Seasonal breakdown options

### ML Prediction
- RandomForest trained on historical turbidity
- Interactive sliders for pH, rainfall, conductance, temperature
- Real-time turbidity forecasts with RMSE accuracy

---

## 📝 Usage Examples

### 1. Explore Time-Series Trends
1. Go to **Explore** page
2. Select "Turbidity" and "pH" parameters
3. Set date range (e.g., Jan 1 – May 30, 2024)
4. View time-series with rolling mean; identify trend direction

### 2. Check Parameter Correlations
1. Go to **Analysis** page
2. Review WQI contributions stacked area
3. Scroll to correlation matrix
4. Compare parameters; note pairwise observation counts

### 3. Predict Turbidity
1. Go to **ML Demo** page
2. Adjust sliders for your scenario (e.g., high rainfall → expect high turbidity)
3. Model predicts turbidity; compare to historical baseline

### 4. Export Data
1. Go to **Data** page
2. Click **Download processed CSV**
3. Use for further analysis in R, Excel, or Python

---

## 🐛 Troubleshooting

### "No data for selected parameter/time range"
- Your date filter or parameter choice is too narrow
- Expand the date range or select all parameters with fewer nulls

### "Too few pairwise observations for reliable correlations"
- Some parameters have high missingness (see `data/README.md`)
- Select fewer parameters or expand the date range

### Streamlit takes a long time to load
- First load caches the data; subsequent loads are fast
- If data file is very large, consider preprocessing to `processed.csv`

### Models show NaN predictions
- Not enough training data after filtering (need ≥30 records)
- Ensure target parameter (Turbidity) has sufficient non-null values

---

## 📚 Next Steps

1. **Run EDA Notebook** — Generate reproducible exploratory analysis (notebooks/eda.ipynb)
2. **Create Preprocessing Module** — Centralize logic into src/preprocessing.py; save processed.csv
3. **Build Insights Pages** — Auto-generate alerts and recommendations from data
4. **Enhance Models** — Add XGBoost, LSTM, and SHAP explainability
5. **Deploy** — Containerize with Docker; push to Streamlit Cloud

---

## 👥 Contributing

To extend the project:

1. **Add a new page**: Create a `def page_name(df):` function in `app.py`; add to menu
2. **New visualization**: Import Plotly/Seaborn; create viz function; call from a page
3. **ML model**: Implement in `train_rf()` or `src/preprocessing.py`; integrate into ML Demo page
4. **Quality improvements**: Fix warnings, optimize performance, improve UI

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 📧 Contact & Questions

For questions or feedback, refer to the project documentation or reach out to the development team.

---

## 🎨 Design Inspiration

The app UI is inspired by **Nykaa** (beauty e-commerce platform):
- **Hero banner** with gradient background and call-to-action
- **Product-style cards** for displaying top sites/items
- **Dark theme** for professional, modern look
- **Responsive layout** with clear hierarchy and navigation

---

*Last updated: 29 November 2025*  
*Status: MVP complete (v1); EDA & preprocessing modules in progress (v2)*
