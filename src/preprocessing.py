"""
Water Quality Data Preprocessing Module

This module provides centralized data transformation functions for water quality
and rainfall data. It handles loading, numeric coercion, outlier detection,
missing value imputation, WQI computation, and feature engineering.

Functions:
    load_water_data() - Load and parse water quality CSV
    load_rainfall_data() - Load and parse rainfall CSV
    coerce_numeric() - Convert columns to numeric, suppressing errors
    cap_outliers() - Cap extreme values at quantile bounds
    impute_missing() - Fill missing values with median
    compute_wqi() - Calculate Water Quality Index
    engineer_features() - Create derived features (season, day-of-week, etc.)
    process_water_data() - Complete pipeline for water data
    process_rainfall_data() - Complete pipeline for rainfall data
    save_processed_data() - Export processed data to CSV
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional, List


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_water_data(filepath: str) -> pd.DataFrame:
    """
    Load water quality CSV with proper dtype handling.
    
    Args:
        filepath: Path to water_quality.csv
        
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(filepath)
    
    # Parse timestamp column
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    return df


def load_rainfall_data(filepath: str) -> pd.DataFrame:
    """
    Load rainfall CSV with proper dtype handling.
    
    Args:
        filepath: Path to rainfall.csv
        
    Returns:
        DataFrame with parsed dates
    """
    df = pd.read_csv(filepath)
    
    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df


# ============================================================================
# NUMERIC COERCION & OUTLIER HANDLING
# ============================================================================

def coerce_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Coerce specified columns to numeric, setting non-convertible values to NaN.
    
    Args:
        df: Input DataFrame
        columns: List of column names to coerce. If None, coerce all except
                 non-numeric-looking columns (Timestamp, date, etc.)
        
    Returns:
        DataFrame with coerced numeric columns
    """
    df = df.copy()
    
    if columns is None:
        # Auto-identify columns to coerce (exclude datetime columns)
        exclude = ['Timestamp', 'date', 'Record number', 'quality', 'Quality']
        columns = [col for col in df.columns if col not in exclude and not col.endswith('[quality]')]
    
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def cap_outliers(df: pd.DataFrame, quantile_range: Tuple[float, float] = (0.001, 0.999)) -> pd.DataFrame:
    """
    Cap extreme outliers at specified quantile bounds for numeric columns.
    
    Args:
        df: Input DataFrame
        quantile_range: Tuple of (lower_quantile, upper_quantile) for capping
        
    Returns:
        DataFrame with capped outliers
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:  # Only if column has non-null values
            q_low = df[col].quantile(quantile_range[0])
            q_high = df[col].quantile(quantile_range[1])
            df[col] = df[col].clip(lower=q_low, upper=q_high)
    
    return df


def impute_missing(df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
    """
    Impute missing values in numeric columns.
    
    Args:
        df: Input DataFrame
        method: 'median', 'mean', or 'forward_fill'
        
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if method == 'median':
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
            elif method == 'mean':
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
            elif method == 'forward_fill':
                df[col] = df[col].ffill().bfill()
            else:
                raise ValueError(f"Unknown imputation method: {method}")
    
    return df


# ============================================================================
# WQI COMPUTATION
# ============================================================================

def compute_wqi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Water Quality Index (WQI) using weighted parameter formula.
    
    WQI Formula:
        WQI = (pH_weight * pH_Q) + (Turbidity_weight * Turbidity_Q) + 
              (SC_weight * SC_Q) + (DO_weight * DO_Q)
    
    Where:
        - pH: 25% weight, ideal = 7.0, std dev = 0.5
        - Turbidity: 35% weight, ideal = 0.1, std dev = 2.0
        - Specific Conductance: 25% weight, ideal = 500, std dev = 100
        - Dissolved Oxygen: 15% weight, ideal = 8.0, std dev = 1.5
    
    Q values range from 0-100 (0=poor, 100=excellent)
    
    Args:
        df: Input DataFrame with water quality parameters
        
    Returns:
        DataFrame with added WQI, Q values, and contributions columns
    """
    df = df.copy()
    
    # Parameter ideals and standard deviations
    params = {
        'pH': {'ideal': 7.0, 'std': 0.5, 'weight': 0.25},
        'Turbidity': {'ideal': 0.1, 'std': 2.0, 'weight': 0.35},
        'Specific Conductance': {'ideal': 500, 'std': 100, 'weight': 0.25},
        'Dissolved Oxygen (%Saturation)': {'ideal': 8.0, 'std': 1.5, 'weight': 0.15}
    }
    
    # Compute Q values (deviation from ideal)
    wqi = 0.0
    
    for param_name, config in params.items():
        if param_name in df.columns:
            ideal = config['ideal']
            std = config['std']
            weight = config['weight']
            
            # Q = 100 * exp(-((value - ideal)^2) / (2 * std^2))
            deviation = (df[param_name] - ideal) ** 2
            q_col = f'{param_name}_Q'
            df[q_col] = 100 * np.exp(-deviation / (2 * std ** 2))
            
            # Contribution = weight * Q
            contrib_col = f'{param_name}_Contribution'
            df[contrib_col] = weight * df[q_col]
            
            wqi += df[contrib_col]
    
    # Final WQI (0-100 scale)
    df['WQI'] = wqi.fillna(0).clip(0, 100)
    
    # WQI Classification
    def classify_wqi(wqi_value):
        if pd.isna(wqi_value):
            return 'Unknown'
        elif wqi_value >= 80:
            return 'Excellent'
        elif wqi_value >= 60:
            return 'Good'
        elif wqi_value >= 40:
            return 'Poor'
        else:
            return 'Very Poor'
    
    df['WQI_Classification'] = df['WQI'].apply(classify_wqi)
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame, date_column: str = 'Timestamp') -> pd.DataFrame:
    """
    Create derived temporal and statistical features.
    
    Features created:
        - Season: Winter, Spring, Summer, Fall
        - Month: 1-12
        - DayOfWeek: Monday-Sunday
        - Hour: 0-23 (if timestamp has time component)
        - IsWeekend: Boolean
        - DayOfYear: 1-365
    
    Args:
        df: Input DataFrame
        date_column: Name of datetime column
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    if date_column not in df.columns:
        return df
    
    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Extract temporal features
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['DayOfWeek'] = df[date_column].dt.day_name()
    df['DayOfWeekNum'] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
    df['IsWeekend'] = df['DayOfWeekNum'].isin([5, 6]).astype(int)
    df['DayOfYear'] = df[date_column].dt.dayofyear
    
    # Season (Northern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    
    # Hour if time component exists
    if df[date_column].dt.hour.nunique() > 1:
        df['Hour'] = df[date_column].dt.hour
    
    return df


# ============================================================================
# COMPLETE PIPELINE FUNCTIONS
# ============================================================================

def generate_synthetic_rainfall(
    real_rainfall_df: pd.DataFrame,
    target_date_range: Tuple[pd.Timestamp, pd.Timestamp],
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic rainfall data for a target date range based on 
    statistical properties of real rainfall data.
    
    Creates daily rainfall by:
    1. Computing seasonal statistics (mean, std) from real data
    2. Sampling from normal distribution for each day
    3. Maintaining seasonal patterns
    
    Args:
        real_rainfall_df: DataFrame with real rainfall (must have 'rainfall' column)
        target_date_range: Tuple of (start_date, end_date) for synthetic data
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic daily rainfall for target date range
    """
    np.random.seed(seed)
    
    # Ensure rainfall column exists
    if 'rainfall' not in real_rainfall_df.columns:
        raise ValueError("real_rainfall_df must have a 'rainfall' column")
    
    # Extract month from date to compute seasonal statistics
    real_rainfall_df = real_rainfall_df.copy()
    if 'date' in real_rainfall_df.columns:
        real_rainfall_df['month'] = pd.to_datetime(real_rainfall_df['date'], errors='coerce').dt.month
    
    # Compute seasonal mean and std for rainfall
    seasonal_stats = real_rainfall_df.groupby('month')['rainfall'].agg(['mean', 'std']).to_dict()
    
    # Create date range for synthetic data
    start_date, end_date = target_date_range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic rainfall
    synthetic_data = []
    for date in date_range:
        month = date.month
        # Get seasonal statistics for this month (fallback to overall mean if not present)
        mean_rainfall = seasonal_stats.get(month, {}).get('mean', real_rainfall_df['rainfall'].mean())
        std_rainfall = seasonal_stats.get(month, {}).get('std', real_rainfall_df['rainfall'].std())
        
        # Ensure std is positive
        std_rainfall = max(std_rainfall, 0.1)
        
        # Sample from normal distribution and clip to non-negative
        synthetic_rainfall = max(0, np.random.normal(mean_rainfall, std_rainfall))
        
        synthetic_data.append({
            'date': date,
            'rainfall': synthetic_rainfall,
            'rain_temperature': np.random.normal(20, 5),  # synthetic temp (°C)
            'rain_humidity': np.clip(np.random.normal(70, 15), 0, 100),  # synthetic humidity (%)
            'rain_wind_speed': max(0, np.random.normal(5, 2)),  # synthetic wind speed (m/s)
            'rain_weather_condition': np.random.choice(['Clear', 'Rainy', 'Cloudy', 'Humid'], p=[0.3, 0.2, 0.35, 0.15])
        })
    
    return pd.DataFrame(synthetic_data)


def process_water_data(
    filepath: str,
    apply_outlier_cap: bool = True,
    imputation_method: str = 'median',
    engineer_feat: bool = True,
    rainfall_path: Optional[str] = None,
    use_synthetic_rainfall: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for water quality data.
    
    Steps:
        1. Load CSV
        2. Coerce numeric columns
        3. Cap outliers (optional)
        4. Impute missing values
        5. Compute WQI
        6. Engineer features (optional)
    
    Args:
        filepath: Path to water_quality.csv
        apply_outlier_cap: Whether to cap outliers at 0.1%-99.9% quantiles
        imputation_method: Method for missing value imputation
        engineer_feat: Whether to create derived features
        rainfall_path: Path to rainfall.csv (optional)
        use_synthetic_rainfall: If rainfall dates don't overlap water dates, generate synthetic rainfall
        
    Returns:
        Fully processed DataFrame
    """
    # Load water
    df = load_water_data(filepath)
    
    # Coerce to numeric early so rainfall merge works with numeric joins
    df = coerce_numeric(df)
    
    # If rainfall provided, load and merge (daily aggregation)
    if rainfall_path and os.path.exists(rainfall_path):
        rain = load_rainfall_data(rainfall_path)
        # Coerce numeric rainfall columns
        rain = coerce_numeric(rain)
        # Ensure date column is floored to day
        if 'date' in rain.columns:
            rain['date'] = pd.to_datetime(rain['date'], errors='coerce').dt.floor('D')
        else:
            rain.columns.values[0] = 'date'
            rain['date'] = pd.to_datetime(rain['date'], errors='coerce').dt.floor('D')

        # Aggregate rainfall daily: sum rainfall (if present), mean of others
        agg_map = {}
        for c in rain.columns:
            if c == 'date':
                continue
            if np.issubdtype(rain[c].dtype, np.number) or c.lower() in ('rainfall', 'temperature', 'humidity', 'wind_speed'):
                agg_map[c] = 'sum' if c.lower() == 'rainfall' else 'mean'
            else:
                agg_map[c] = 'first'

        rain_daily = rain.groupby('date').agg(agg_map).reset_index()

        # Standardize rainfall column names to avoid accidental overwrites and
        # make downstream merging predictable. Prefix non-primary rainfall
        # quantities with 'rain_' while keeping the main rainfall amount as
        # 'rainfall' if present.
        rename_map = {}
        for c in rain_daily.columns:
            if c == 'date':
                continue
            if c.lower() == 'rainfall':
                rename_map[c] = 'rainfall'
            else:
                clean = c.lower().strip().replace(' ', '_')
                rename_map[c] = f'rain_{clean}'

        rain_daily = rain_daily.rename(columns=rename_map)

        # Ensure water df has a date column
        if 'Timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.floor('D')
        else:
            df['date'] = pd.NaT

        # Inspect date ranges to detect potential non-overlap
        try:
            water_min = df['date'].min()
            water_max = df['date'].max()
            rain_min = rain_daily['date'].min()
            rain_max = rain_daily['date'].max()
            print(f"Rainfall data date range: {rain_min} to {rain_max}")
            print(f"Water data date range: {water_min} to {water_max}")
            
            # If no overlap, generate synthetic rainfall for water date range
            if use_synthetic_rainfall and ((rain_max < water_min) or (rain_min > water_max)):
                print("⚠️ Rainfall data does not overlap water data dates.")
                print("📊 Generating synthetic rainfall based on 2022 statistical properties...")
                rain_daily = generate_synthetic_rainfall(rain_daily, (water_min, water_max))
                print(f"✓ Generated synthetic rainfall for {len(rain_daily)} days ({water_min.date()} to {water_max.date()})")
        except Exception as e:
            print(f"Error during date range check: {e}")

        # Merge rainfall into water data (left join keeps all water rows)
        df = pd.merge(df, rain_daily, on='date', how='left')

    # Cap outliers
    if apply_outlier_cap:
        df = cap_outliers(df, quantile_range=(0.001, 0.999))

    # Impute missing (now includes rainfall columns)
    df = impute_missing(df, method=imputation_method)

    # Compute WQI
    df = compute_wqi(df)

    # Engineer features
    if engineer_feat:
        df = engineer_features(df, date_column='Timestamp')

    return df


def process_rainfall_data(
    filepath: str,
    engineer_feat: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for rainfall data.
    
    Steps:
        1. Load CSV
        2. Coerce numeric columns
        3. Engineer features (optional)
    
    Args:
        filepath: Path to rainfall.csv
        engineer_feat: Whether to create derived features
        
    Returns:
        Fully processed DataFrame
    """
    # Load
    df = load_rainfall_data(filepath)
    
    # Coerce numeric (exclude date and weather columns)
    exclude_cols = ['date', 'weather_condition', 'Weather']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    df = coerce_numeric(df, columns=numeric_cols)
    
    # Engineer features
    if engineer_feat:
        df = engineer_features(df, date_column='date')
    
    return df


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_processed_data(
    df: pd.DataFrame,
    output_path: str,
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> None:
    """
    Save processed DataFrame to CSV with proper formatting.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save CSV file
        date_format: Format string for datetime columns
    """
    df_export = df.copy()
    
    # Format datetime columns
    datetime_cols = df_export.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df_export[col] = df_export[col].dt.strftime(date_format)
    
    # Save to CSV
    df_export.to_csv(output_path, index=False)
    print(f"✓ Saved processed data to {output_path}")
    print(f"  Shape: {df_export.shape}")
    print(f"  Date range: {df_export.iloc[0] if len(df_export) > 0 else 'N/A'}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate data quality report for DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Summary DataFrame with null counts and percentages
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'DataType': df.dtypes,
        'NonNullCount': df.count(),
        'NullCount': df.isnull().sum(),
        'NullPercentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    return summary.sort_values('NullPercentage', ascending=False)


def align_water_rainfall_data(
    water_df: pd.DataFrame,
    rainfall_df: pd.DataFrame,
    water_date_col: str = 'Timestamp',
    rainfall_date_col: str = 'date'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align water and rainfall data to common date range.
    
    Args:
        water_df: Processed water quality DataFrame
        rainfall_df: Processed rainfall DataFrame
        water_date_col: Name of date column in water_df
        rainfall_date_col: Name of date column in rainfall_df
        
    Returns:
        Tuple of (aligned_water_df, aligned_rainfall_df)
    """
    # Extract dates
    water_dates = pd.to_datetime(water_df[water_date_col]).dt.date
    rainfall_dates = pd.to_datetime(rainfall_df[rainfall_date_col]).dt.date
    
    # Find common date range
    common_min = max(water_dates.min(), rainfall_dates.min())
    common_max = min(water_dates.max(), rainfall_dates.max())
    
    # Filter to common range
    water_mask = (water_dates >= common_min) & (water_dates <= common_max)
    rainfall_mask = (rainfall_dates >= common_min) & (rainfall_dates <= common_max)
    
    water_aligned = water_df[water_mask].reset_index(drop=True)
    rainfall_aligned = rainfall_df[rainfall_mask].reset_index(drop=True)
    
    print(f"Data Alignment Summary:")
    print(f"  Water records in range: {len(water_aligned)}")
    print(f"  Rainfall records in range: {len(rainfall_aligned)}")
    print(f"  Date range: {common_min} to {common_max}")
    
    return water_aligned, rainfall_aligned


# ============================================================================
# MAIN EXECUTION (FOR STANDALONE TESTING)
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    water_path = os.path.join(data_dir, 'water_quality.csv')
    rainfall_path = os.path.join(data_dir, 'rainfall.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    print("=" * 70)
    print("WATER QUALITY DATA PREPROCESSING")
    print("=" * 70)
    
    # Process water data
    print("\n[1/2] Processing water quality data...")
    if os.path.exists(water_path):
        water_processed = process_water_data(water_path)
        print(f"✓ Loaded and processed: {water_processed.shape[0]} records, {water_processed.shape[1]} columns")
        print(f"  WQI range: {water_processed['WQI'].min():.2f} - {water_processed['WQI'].max():.2f}")
        
        # Save processed water data
        output_water = os.path.join(output_dir, 'water_quality_processed.csv')
        save_processed_data(water_processed, output_water)
        
        # Daily aggregation
        print("\n[1b] Creating daily aggregation...")
        water_sorted = water_processed.sort_values('Timestamp')
        numeric_cols = water_sorted.select_dtypes(include=[np.number]).columns
        daily_water = water_sorted.set_index('Timestamp')[numeric_cols].resample('D').mean()
        
        output_daily = os.path.join(output_dir, 'water_quality_daily.csv')
        daily_water.to_csv(output_daily)
        print(f"✓ Saved daily aggregated data: {daily_water.shape[0]} records")
    else:
        print(f"✗ Water data file not found: {water_path}")
        sys.exit(1)
    
    # Process rainfall data
    print("\n[2/2] Processing rainfall data...")
    if os.path.exists(rainfall_path):
        rainfall_processed = process_rainfall_data(rainfall_path)
        print(f"✓ Loaded and processed: {rainfall_processed.shape[0]} records, {rainfall_processed.shape[1]} columns")
        
        # Save processed rainfall data
        output_rainfall = os.path.join(output_dir, 'rainfall_processed.csv')
        save_processed_data(rainfall_processed, output_rainfall)
    else:
        print(f"✗ Rainfall data file not found: {rainfall_path}")
        sys.exit(1)
    
    # Data quality summary
    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    print("\nWater Quality Data:")
    print(get_data_quality_summary(water_processed).head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 70)
