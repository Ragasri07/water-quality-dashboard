# 💧 Water Quality Management Dashboard

> *Advanced data visualization and analytics for water quality monitoring*

An interactive Streamlit dashboard featuring **10+ advanced visualization techniques** for comprehensive water quality analysis. Built for academic demonstration of data visualization methods, featuring cutting-edge charts, statistical analysis, and machine learning models.

---

## 🎯 Project Overview

**Goal**: Demonstrate advanced data visualization techniques through comprehensive water quality monitoring and analysis.

**Project Type**: Academic Data Visualization Project

**Key Features**:
- 🎨 **10+ Advanced Visualization Types** (Gauge, Radar, Sankey, Waterfall, Sunburst, Treemap, etc.)
- 📊 **Interactive Dashboards** with real-time filtering and exploration
- 🤖 **Machine Learning Models** (RandomForest, XGBoost, LightGBM)
- 📈 **Statistical Analysis** (Anomaly Detection, Hypothesis Testing, Correlations)
- 🎯 **Professional UI/UX** with animated gradients and responsive design

**Target Audience**:
- Data visualization students and educators
- Environmental data analysts
- Data science portfolio projects
- Academic project demonstrations

---

## 📊 Features & Visualizations

### 🏠 Home Page
✅ **Core Metrics Dashboard**
- KPI cards with gradient backgrounds (Total Records, Avg WQI, pH, Turbidity)
- Circular progress indicators for WQI classifications
- Time series trends and distribution charts

✅ **Advanced Visualizations**
- 📊 **Gauge Chart**: Real-time WQI with color-coded thresholds and delta indicators
- 🕸️ **Radar Chart**: Current vs ideal parameter comparison (multi-dimensional)
- 🌊 **Sankey Diagram**: Water quality flow analysis (Start → End states with color-coded improvements)
- 📉 **Waterfall Chart**: WQI component breakdown showing contributions

### 🔍 Explore Page (4 Comprehensive Tabs)
✅ **Time Series Explorer**
- Multi-parameter comparison with interactive date ranges
- Rolling mean smoothing with statistical summaries
- Dynamic chart updates with parameter selection

✅ **Distribution Analysis**
- Histogram + Box plots with detailed statistics
- Violin plots for distribution shape visualization
- Kernel Density Estimation (KDE) plots

✅ **Parameter Inspector**
- Hourly pattern analysis with line charts
- Day-of-week aggregations
- Outlier detection using IQR method

✅ **Advanced Multi-Dimensional Views**
- 📐 **Parallel Coordinates**: High-dimensional data visualization (500 sample points)
- ☀️ **Sunburst Chart**: Hierarchical Season → WQI classification breakdown

### 📈 Analysis Page (6 Advanced Tabs)
✅ **WQI Trends Analysis**
- Period comparison (7, 30, 90 days)
- Volatility analysis with rolling standard deviation
- Trend direction indicators

✅ **Seasonal Decomposition**
- Trend, seasonal, and residual components
- Time series decomposition using statsmodels

✅ **Correlation Analysis**
- Spearman correlation heatmap with threshold filtering
- Top correlations table with strength indicators
- Pairwise observation counts

✅ **Temporal Patterns**
- Monthly aggregation bar charts
- Hourly heatmaps showing patterns over time
- Day-of-week analysis

✅ **Anomaly Detection (3 Methods)**
- 🔍 **IQR Method**: Interquartile range outlier detection
- 📊 **Z-Score Method**: Statistical deviation analysis
- 🤖 **Isolation Forest**: ML-based anomaly detection

✅ **Statistical Hypothesis Testing**
- ANOVA tests for WQI class comparisons
- T-tests for temporal period comparisons
- P-value significance reporting

### 💡 Insights Page
✅ **Health Score Dashboard**
- Overall health score with color-coded indicators
- Compliance rate tracking
- Stability index calculations

✅ **Parameter Health Cards**
- Individual parameter status (pH, DO, Turbidity, Conductivity)
- Color-coded health badges (Excellent/Good/Poor)
- Recent trend indicators

✅ **Advanced Visualizations**
- 🎯 **Bullet Charts**: Performance vs targets with gauge indicators
- 🗂️ **Treemap**: Hierarchical data completeness visualization
- 🏔️ **Ridge Plots**: Monthly parameter distribution comparison
- 🌡️ **2D Density Contour**: Scatter + contour overlay for relationships

### 🤖 ML Demo Page
✅ **3 Machine Learning Models**
- **RandomForest**: Ensemble learning with 50 trees
- **XGBoost**: Gradient boosting (depth 5, 50 estimators)
- **LightGBM**: Light gradient boosting machine

✅ **Interactive Predictions**
- 9 feature inputs with dynamic sliders
- Real-time predictions with confidence metrics
- Model performance comparison (RMSE, R², MAE)

✅ **Model Explainability**
- Feature importance charts (bar & pie)
- Medal system for feature ranking
- Model saving and loading functionality

### 📋 Data Page
✅ **Data Explorer**
- Full dataset table view with pagination
- Advanced filtering and sorting
- Column selection and search

✅ **Data Export**
- CSV download functionality
- Filtered data export options
- Summary statistics display

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- `pip` or `conda`

### Installation

```bash
# Clone the repository
git clone https://github.com/Ragasri07/water-quality-dashboard.git
cd water-quality-dashboard

# Create and activate virtual environment (recommended)
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
| **Framework** | Streamlit 1.50.0 |
| **Data Processing** | Pandas 2.3.3, NumPy 2.0.2 |
| **Visualization** | Plotly 6.5.0, Seaborn, Matplotlib |
| **Machine Learning** | Scikit-learn 1.6.1, XGBoost 2.1.4, LightGBM 4.6.0 |
| **Statistical Analysis** | SciPy 1.13.1, StatsModels 0.14.5 |
| **Language** | Python 3.9+ |
| **Deployment** | Streamlit Cloud |

### Advanced Visualization Library
All charts built with **Plotly** for:
- ✨ Interactive zoom, pan, and hover functionality
- 📱 Responsive design for all screen sizes
- 🎨 Professional color schemes and themes
- 💾 Export to PNG/SVG functionality

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
- 3 models trained on 9 features (pH, Turbidity, Conductance, DO, Temperature, Salinity, Chlorophyll, Water Speed, Direction)
- Interactive sliders for all parameters
- Real-time predictions with performance metrics (RMSE, R², MAE)
- Feature importance visualization and explainability

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

## 🎓 Visualization Techniques Demonstrated

This project showcases **10+ advanced data visualization types**:

1. **Gauge Charts** (`plotly.graph_objects.Indicator`) - Performance metrics with thresholds
2. **Radar/Spider Charts** (`go.Scatterpolar`) - Multi-dimensional comparisons
3. **Sankey Diagrams** (`go.Sankey`) - Flow analysis and state transitions
4. **Waterfall Charts** (`go.Waterfall`) - Component contribution breakdown
5. **Parallel Coordinates** (`plotly.express.parallel_coordinates`) - High-dimensional data
6. **Sunburst Charts** (`px.sunburst`) - Hierarchical data visualization
7. **Bullet Charts** (`go.Indicator` with bullet gauge) - Target vs actual performance
8. **Treemaps** (`px.treemap`) - Hierarchical proportions
9. **Ridge Plots** (Multiple `go.Violin`) - Distribution comparison across categories
10. **2D Density Contours** (`go.Histogram2dContour`) - Bivariate distributions
11. **3D Scatter Plots** (`px.scatter_3d`) - Three-dimensional relationships
12. **Heatmaps** - Temporal patterns and correlations
13. **Interactive Time Series** - Multi-parameter trends with brushing/linking

### Statistical Methods Applied
- Anomaly detection (IQR, Z-Score, Isolation Forest)
- Hypothesis testing (ANOVA, T-tests)
- Correlation analysis (Spearman)
- Seasonal decomposition
- Rolling statistics and volatility

---

## 👥 Contributing

To extend the project:

1. **Add a new page**: Create a `def page_name(df):` function in `app.py`; add to menu
2. **New visualization**: Import Plotly/Seaborn; create viz function; call from a page
3. **ML model**: Implement in `train_rf()` or `src/preprocessing.py`; integrate into ML Demo page
4. **Quality improvements**: Fix warnings, optimize performance, improve UI

---

## 🚀 Live Demo

**Deployed App**: [https://water-quality-dashboard-data-vizualization.streamlit.app](https://water-quality-dashboard-data-vizualization.streamlit.app)

**GitHub Repository**: [Ragasri07/water-quality-dashboard](https://github.com/Ragasri07/water-quality-dashboard)

---

## 📸 Screenshots

![Home Dashboard](https://via.placeholder.com/800x400?text=Advanced+Visualizations)
*Home page featuring Gauge, Radar, Sankey, and Waterfall charts*

![Explore Page](https://via.placeholder.com/800x400?text=Interactive+Exploration)
*Multi-dimensional data exploration with parallel coordinates and sunburst charts*

---

## 🎨 UI/UX Design

**Professional Features**:
- 🎭 Animated gradient header with smooth transitions
- 🎯 Clean navigation with pill-style buttons
- 📱 Fully responsive layout
- 🎨 Consistent color scheme and typography
- ✨ Interactive visualizations with hover tooltips
- 🖼️ White background with professional spacing

---

## 📄 License

This project is provided for educational and portfolio purposes.

---

## 👤 Author

**Ragasri07**
- GitHub: [@Ragasri07](https://github.com/Ragasri07)
- Project: Water Quality Management Dashboard

---

## 🙏 Acknowledgments

- Data visualization techniques inspired by modern data science practices
- Built with Streamlit and Plotly for interactive analytics
- Designed for academic demonstration of advanced visualization methods

---

*Last Updated: December 3, 2025*  
*Version: 1.0 - Production Ready*  
*Status: ✅ Deployed on Streamlit Cloud*
