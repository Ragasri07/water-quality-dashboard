# Water Quality Project — Data Inventory & Validation

This document describes the structure, quality, and constraints of the water quality and rainfall datasets used in the WaterWatch project.

---

## Dataset Overview

### 1. Water Quality Data (`water_quality.csv`)

**File size:** ~30,900 records × 20 columns

**Temporal coverage:** Jan 2024 to May 2024 (approximately 152 days)

**Data source:** Sensor measurements (likely from a water monitoring station or network)

#### Column Inventory

| Column | Type | Null Count | Null % | Description |
|--------|------|-----------|--------|-------------|
| **Timestamp** | datetime | 0 | 0% | Observation timestamp (YYYY-MM-DD HH:MM:SS) |
| **Record number** | int | 0 | 0% | Sequential record identifier |
| **Average Water Speed** | float | 20 | 0.06% | Water flow/current speed (units unknown) |
| **Average Water Direction** | float | 1 | 0.003% | Water flow direction (degrees?) |
| **Chlorophyll** | float | 585 | 1.9% | Chlorophyll concentration (mg/m³?) |
| **Chlorophyll [quality]** | object | 808 | 2.6% | Quality flag for chlorophyll measurement |
| **Temperature** | float | 5,164 | 16.7% | Water temperature (°C) |
| **Temperature [quality]** | object | 5,344 | 17.3% | Quality flag for temperature |
| **Dissolved Oxygen** | float | 4,300 | 13.9% | Dissolved oxygen (mg/L) |
| **Dissolved Oxygen [quality]** | object | 4,524 | 14.6% | Quality flag for dissolved oxygen |
| **Dissolved Oxygen (%Saturation)** | float | 5,749 | 18.6% | DO saturation percentage (%) |
| **DO (%Saturation) [quality]** | object | 5,950 | 19.3% | Quality flag for DO saturation |
| **pH** | float | 1,084 | 3.5% | pH value (0–14 scale) |
| **pH [quality]** | object | 1,308 | 4.2% | Quality flag for pH |
| **Salinity** | float | 3,958 | 12.8% | Water salinity (PSU or ppt) |
| **Salinity [quality]** | object | 4,182 | 13.5% | Quality flag for salinity |
| **Specific Conductance** | float | 1,367 | 4.4% | Electrical conductivity (µS/cm) |
| **Specific Conductance [quality]** | object | 1,591 | 5.2% | Quality flag for conductance |
| **Turbidity** | float | 2,000 | 6.5% | Water turbidity (NTU) |
| **Turbidity [quality]** | object | 2,224 | 7.2% | Quality flag for turbidity |

#### Key Observations

- **No site/location identifier**: The dataset lacks a `Site`, `Location`, `Station`, or `StationID` column, meaning all records are aggregated from a single monitoring point or pre-aggregated across sites.
- **No geographic data**: Latitude and longitude are not present; geographic analysis is not possible without external site mapping.
- **Sparse parameters**: Temperature, Dissolved Oxygen, and DO saturation have ~17–19% missing values, indicating sensor downtime or poor data quality during those periods.
- **Quality flags**: Parallel "[quality]" columns suggest sensor QA/QC codes (likely flags like 'Good', 'Questionable', 'Bad'). These are not currently used in preprocessing but could inform confidence weighting.

---

### 2. Rainfall Data (`rainfall.csv`)

**File size:** 54 records × 6 columns

**Temporal coverage:** ~54 days (sparse, subset of water quality date range)

**Data source:** Weather station or meteorological database

#### Column Inventory

| Column | Type | Non-null | Description |
|--------|------|----------|-------------|
| **date** | datetime | 54 | Daily observation date (YYYY-MM-DD) |
| **rainfall** | float | 54 | Daily precipitation (mm) |
| **temperature** | float | 54 | Daily air temperature (°C) |
| **humidity** | float | 54 | Daily relative humidity (%) |
| **wind_speed** | float | 54 | Daily wind speed (km/h or m/s) |
| **weather_condition** | object | 54 | Categorical weather description (e.g., 'Sunny', 'Rainy') |

#### Key Observations

- **Much smaller dataset**: Only 54 daily records vs. 30,894 water quality records.
- **Temporal mismatch**: Rainfall data spans a subset of the water quality date range; daily merging required.
- **Complete (no nulls)**: All 54 rows are complete, making it clean for daily aggregation.
- **Weather condition** is categorical; not used in correlation but useful for storytelling.

---

## Data Quality Issues & Constraints

### Critical Limitations

1. **Single monitoring location**: No site/station information prevents per-site analysis, trend comparison, or geospatial visualization.
   - *Workaround*: The app aggregates all records as a single time-series or shows daily worst/best values.

2. **Missing geographic coordinates**: Latitude/longitude not available.
   - *Workaround*: Map view cannot be implemented without external data enrichment.

3. **Temporal data sparsity**:
   - Water quality: ~6.5–19% missing across parameters (sensor downtime/maintenance).
   - Rainfall: Only 54 daily records vs. 152 days of water data.
   - *Workaround*: Median imputation for water parameters; pairwise-complete correlations; daily aggregation of rainfall.

4. **Quality flags not standardized**: "[quality]" columns are present but not documented or interpreted in the app.
   - *Workaround*: Currently ignored; could be enhanced to weight measurements by confidence.

### Data Processing Strategy

The `app.py` and future `src/preprocessing.py` module apply:

1. **Timestamp parsing**: Robust parsing with fallback to detect date columns.
2. **Numeric coercion**: Convert string columns to numeric with error handling.
3. **Outlier capping**: Clamp values to 0.1%–99.9% quantiles to remove sensor errors/spikes.
4. **Median imputation**: Fill remaining nulls with per-column medians to maintain sample size.
5. **Daily aggregation**: Resample water quality to daily means for cleaner trends and alignment with rainfall.
6. **WQI computation**: Weighted combination of pH, Turbidity, Specific Conductance, and Dissolved Oxygen using standardized sub-indices.

---

## Data Statistics

### Water Quality Parameters (Post-Processing Estimates)

After numeric conversion and filtering:

- **pH**: ~29,810 non-null values (range: 0–14)
- **Turbidity**: ~28,894 non-null values (range: 0–∞, typically 0–100+ NTU)
- **Specific Conductance**: ~29,527 non-null values (range: 0–10,000+ µS/cm)
- **Dissolved Oxygen**: ~26,594 non-null values (range: 0–15+ mg/L)
- **Temperature**: ~25,730 non-null values (range: ~5–35°C)
- **Salinity**: ~26,936 non-null values (range: 0–35+ PSU)

### Rainfall Statistics

- **Total rainfall**: 54 daily measurements, mean ~4–5 mm/day (seasonal dependent)
- **Air temperature**: ~10–30°C range (seasonal)
- **Humidity**: ~30–90% range
- **Wind speed**: ~0–20 km/h typical

---

## Recommendations for Data Enhancement

To improve analysis, consider:

1. **Add site/location identifiers**: If multiple stations exist, include StationID, Site Name, Latitude, Longitude, and Site Type (urban, rural, industrial, etc.).
2. **Document quality flags**: Provide a mapping for "[quality]" column codes and apply confidence weights in modeling.
3. **Expand rainfall data**: Collect daily records matching the full water quality date range for better correlation analysis.
4. **Include metadata**: Add units, sensor model, maintenance logs, and data gaps documentation.
5. **Align timestamps**: If multiple monitoring points exist, ensure consistent time zones and sampling frequencies.

---

## File Manifest

- `water_quality.csv`: Primary water sensor dataset (30,894 rows)
- `rainfall.csv`: Daily weather/rainfall data (54 rows)
- `processed.csv` (generated): Output of preprocessing pipeline (to be created)

---

## Data Access Notes

- **Load in Python**: `pd.read_csv('water_quality.csv')`
- **Streamlit app**: Automatically loads, preprocesses, and displays via `app.py`
- **Preprocessing module**: `src/preprocessing.py` will centralize all transformations (to be implemented)

---

*Last updated: 29 November 2025*
*Next review: After EDA notebook completion*
