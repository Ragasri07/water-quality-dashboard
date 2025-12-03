"""Water Quality Analysis Package"""

from .preprocessing import (
    load_water_data,
    load_rainfall_data,
    coerce_numeric,
    cap_outliers,
    impute_missing,
    compute_wqi,
    engineer_features,
    process_water_data,
    process_rainfall_data,
    save_processed_data,
    get_data_quality_summary,
    align_water_rainfall_data
)

from .ml_models import (
    prepare_ml_data,
    train_random_forest,
    train_xgboost_model,
    train_lstm_model,
    train_ensemble_model,
    compare_models,
    generate_shap_explanation,
    get_model_recommendations
)

__all__ = [
    'load_water_data',
    'load_rainfall_data',
    'coerce_numeric',
    'cap_outliers',
    'impute_missing',
    'compute_wqi',
    'engineer_features',
    'process_water_data',
    'process_rainfall_data',
    'save_processed_data',
    'get_data_quality_summary',
    'align_water_rainfall_data',
    'prepare_ml_data',
    'train_random_forest',
    'train_xgboost_model',
    'train_lstm_model',
    'train_ensemble_model',
    'compare_models',
    'generate_shap_explanation',
    'get_model_recommendations'
]
