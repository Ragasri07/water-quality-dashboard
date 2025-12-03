"""
Advanced Machine Learning Models for Water Quality Prediction

This module provides implementations of multiple ML models for forecasting
water quality parameters, including model comparison, hyperparameter tuning,
and explainability using SHAP.

Models:
    - RandomForest (baseline)
    - XGBoost (gradient boosting)
    - LSTM (deep learning for time-series)
    - Ensemble (voting of multiple models)

Functions:
    train_random_forest() - Baseline RandomForest model
    train_xgboost_model() - XGBoost gradient boosting
    train_lstm_model() - LSTM neural network
    train_ensemble_model() - Voting ensemble
    compare_models() - Train and compare all models
    generate_shap_explanation() - SHAP feature importance
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    # Catch both ImportError and XGBoostError (missing libomp)
    HAS_XGBOOST = False
    xgb = None

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_ml_data(
    df: pd.DataFrame,
    target_col: str = 'Turbidity',
    test_size: float = 0.2,
    lags: int = 2,
    min_feature_coverage: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Prepare data for ML modeling with lag features.
    
    Args:
        df: Input DataFrame with time-series data
        target_col: Target column for prediction
        test_size: Test set fraction
        lags: Number of lag features to create
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    df_sorted = df.sort_values('Timestamp').copy()
    
    if target_col not in df_sorted.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    # Remove rows with NaN target
    df_sorted = df_sorted[df_sorted[target_col].notna()].copy()
    
    # Select numeric features (exclude timestamp and quality flags)
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['WQI', 'Record number', 'date', 'month', 'year']

    # Prefer rainfall-related environmental features if present. Look for
    # columns that start with 'rain' (we prefix rainfall features during
    # preprocessing) or exact 'rainfall'. Only consider numeric cols.
    rainfall_related = [c for c in numeric_cols if c.lower().startswith('rain') or c.lower() == 'rainfall']

    # Reduced to top 6 core features for faster training
    feature_cols_core = [col for col in numeric_cols if col not in exclude and '_' not in col and col not in rainfall_related][:6]
    # Put rainfall features first (if available), then core features
    feature_cols = rainfall_related + [c for c in feature_cols_core if c not in rainfall_related]

    # Create lag features for the target and append to feature list
    for lag in range(1, lags + 1):
        lag_col = f'{target_col}_lag_{lag}'
        df_sorted[lag_col] = df_sorted[target_col].shift(lag)
        feature_cols.append(lag_col)

    # Before dropping rows, filter out features with extremely low coverage
    # (default: 1%). This prevents all-NaN feature columns from removing
    # the entire dataset during dropna.
    coverage = df_sorted[feature_cols].notna().mean()
    keep_features = [f for f in feature_cols if coverage.get(f, 0) >= min_feature_coverage]
    dropped = [f for f in feature_cols if f not in keep_features]
    if dropped:
        print(f"Note: Dropping low-coverage features (<{min_feature_coverage*100:.1f}%): {dropped}")

    if not keep_features:
        raise ValueError("No features meet the minimum coverage threshold. Provide more data or reduce 'min_feature_coverage'.")

    # Drop rows with NaNs in the selected features or target
    df_sorted = df_sorted.dropna(subset=keep_features + [target_col])
    
    # Sample data for faster training (use last 2000 rows for recency)
    if len(df_sorted) > 2000:
        print(f"Sampling {len(df_sorted)} rows down to 2000 for faster training")
        df_sorted = df_sorted.iloc[-2000:]

    X = df_sorted[keep_features].values
    y = df_sorted[target_col].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # No shuffle for time-series
    )
    
    return X_train, X_test, y_train, y_test, keep_features


# ============================================================================
# RANDOM FOREST MODEL (BASELINE)
# ============================================================================

def train_random_forest(
    df: pd.DataFrame,
    target_col: str = 'Turbidity',
    n_estimators: int = 150,
    max_depth: int = 15,
    min_samples_split: int = 5,
    random_state: int = 42
) -> Tuple[RandomForestRegressor, float, float, float]:
    """
    Train RandomForest baseline model.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split node
        random_state: Random seed
        
    Returns:
        Tuple of (model, rmse, r2, mae)
    """
    try:
        X_train, X_test, y_train, y_test, _ = prepare_ml_data(df, target_col)
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None, None, None, None
    
    if len(X_train) < 30:
        print("Insufficient training data (need >= 30 samples)")
        return None, None, None, None
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, rmse, r2, mae


# ============================================================================
# XGBOOST MODEL
# ============================================================================

def train_xgboost_model(
    df: pd.DataFrame,
    target_col: str = 'Turbidity',
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[Optional[object], float, float, float]:
    """
    Train XGBoost gradient boosting model.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        random_state: Random seed
        
    Returns:
        Tuple of (model, rmse, r2, mae)
    """
    if not HAS_XGBOOST:
        print("XGBoost not installed. Install with: pip install xgboost")
        return None, None, None, None
    
    try:
        X_train, X_test, y_train, y_test, _ = prepare_ml_data(df, target_col)
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None, None, None, None
    
    if len(X_train) < 30:
        print("Insufficient training data")
        return None, None, None, None
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        tree_method='hist',
        device='cpu'
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, rmse, r2, mae


# ============================================================================
# LSTM MODEL
# ============================================================================

def create_sequences(data: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm_model(
    df: pd.DataFrame,
    target_col: str = 'Turbidity',
    seq_length: int = 10,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[Optional[object], float, float, float]:
    """
    Train LSTM neural network for time-series forecasting.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        seq_length: Sequence length for LSTM
        epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Tuple of (model, rmse, r2, mae)
    """
    if not HAS_LSTM:
        print("TensorFlow/Keras not installed. Install with: pip install tensorflow")
        return None, None, None, None
    
    try:
        df_sorted = df.sort_values('Timestamp').copy()
        data = df_sorted[target_col].dropna().values
        
        if len(data) < seq_length + 30:
            print("Insufficient data for LSTM")
            return None, None, None, None
        
        # Normalize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = create_sequences(data_scaled, seq_length)
        
        # Train-test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_length, 1)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        y_pred_scaled = model.predict(X_test, verbose=0).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        mae = mean_absolute_error(y_test_actual, y_pred)
        
        return model, rmse, r2, mae
        
    except Exception as e:
        print(f"LSTM training failed: {e}")
        return None, None, None, None


# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

def train_ensemble_model(
    df: pd.DataFrame,
    target_col: str = 'Turbidity'
) -> Tuple[Optional[VotingRegressor], Dict]:
    """
    Train voting ensemble of multiple models.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        
    Returns:
        Tuple of (ensemble_model, metrics_dict)
    """
    try:
        X_train, X_test, y_train, y_test, _ = prepare_ml_data(df, target_col)
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return None, {}
    
    if len(X_train) < 30:
        print("Insufficient training data")
        return None, {}
    
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # Try adding XGBoost if available
    if HAS_XGBOOST:
        estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)))
    
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    return ensemble, metrics


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def compare_models(
    df: pd.DataFrame,
    target_col: str = 'Turbidity'
) -> pd.DataFrame:
    """
    Train and compare all available models.
    
    Args:
        df: Input DataFrame
        target_col: Target column
        
    Returns:
        DataFrame with model comparison results
    """
    results = []
    
    # RandomForest
    rf_model, rf_rmse, rf_r2, rf_mae = train_random_forest(df, target_col)
    if rf_model:
        results.append({
            'Model': 'RandomForest',
            'RMSE': rf_rmse,
            'R²': rf_r2,
            'MAE': rf_mae,
            'Status': '✓'
        })
    else:
        results.append({
            'Model': 'RandomForest',
            'RMSE': None,
            'R²': None,
            'MAE': None,
            'Status': '✗ Insufficient data'
        })
    
    # XGBoost
    if HAS_XGBOOST:
        xgb_model, xgb_rmse, xgb_r2, xgb_mae = train_xgboost_model(df, target_col)
        if xgb_model:
            results.append({
                'Model': 'XGBoost',
                'RMSE': xgb_rmse,
                'R²': xgb_r2,
                'MAE': xgb_mae,
                'Status': '✓'
            })
    else:
        results.append({
            'Model': 'XGBoost',
            'RMSE': None,
            'R²': None,
            'MAE': None,
            'Status': '✗ Not installed'
        })
    
    # LSTM
    if HAS_LSTM:
        lstm_model, lstm_rmse, lstm_r2, lstm_mae = train_lstm_model(df, target_col)
        if lstm_model:
            results.append({
                'Model': 'LSTM',
                'RMSE': lstm_rmse,
                'R²': lstm_r2,
                'MAE': lstm_mae,
                'Status': '✓'
            })
    else:
        results.append({
            'Model': 'LSTM',
            'RMSE': None,
            'R²': None,
            'MAE': None,
            'Status': '✗ Not installed'
        })
    
    # Ensemble
    ensemble_model, ensemble_metrics = train_ensemble_model(df, target_col)
    if ensemble_model:
        results.append({
            'Model': 'Ensemble',
            'RMSE': ensemble_metrics.get('rmse'),
            'R²': ensemble_metrics.get('r2'),
            'MAE': ensemble_metrics.get('mae'),
            'Status': '✓'
        })
    else:
        results.append({
            'Model': 'Ensemble',
            'RMSE': None,
            'R²': None,
            'MAE': None,
            'Status': '✗ Training failed'
        })
    
    return pd.DataFrame(results)


# ============================================================================
# SHAP EXPLAINABILITY
# ============================================================================

def generate_shap_explanation(
    model: object,
    X: np.ndarray,
    feature_names: list,
    model_type: str = 'tree'
) -> Optional[Dict]:
    """
    Generate SHAP feature importance explanation.
    
    Args:
        model: Trained model
        X: Input features
        feature_names: List of feature names
        model_type: Type of model ('tree', 'linear', 'deep')
        
    Returns:
        Dictionary with SHAP values and mean absolute SHAP values
    """
    if not HAS_SHAP:
        print("SHAP not installed. Install with: pip install shap")
        return None
    
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X[:100])  # Use sample
        
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        return {
            'shap_values': shap_values,
            'base_value': explainer.expected_value,
            'feature_importance': dict(zip(feature_names, mean_abs_shap)),
            'feature_names': feature_names
        }
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_recommendations(comparison_df: pd.DataFrame) -> str:
    """Generate recommendation based on model comparison."""
    valid_models = comparison_df[comparison_df['Status'] == '✓'].copy()
    
    if valid_models.empty:
        return "No models trained successfully. Check data quality."
    
    best_r2 = valid_models.loc[valid_models['R²'].idxmax()]
    best_rmse = valid_models.loc[valid_models['RMSE'].idxmin()]
    
    recommendation = f"""
**Recommended Models:**
1. Best R² Score: {best_r2['Model']} (R² = {best_r2['R²']:.4f})
2. Best RMSE: {best_rmse['Model']} (RMSE = {best_rmse['RMSE']:.4f})

**Next Steps:**
- Use {best_r2['Model']} for interpretability and explanation
- Consider {best_rmse['Model']} for lower prediction error
- Ensemble combines strengths of multiple models
- Deploy chosen model using FastAPI for real-time predictions
"""
    return recommendation


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.insert(0, '..')
    from src.preprocessing import process_water_data
    
    # Load and process data
    water_df = process_water_data('data/water_quality.csv')
    
    # Compare models
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    comparison = compare_models(water_df, target_col='Turbidity')
    print(comparison.to_string(index=False))
    
    print("\n" + get_model_recommendations(comparison))
