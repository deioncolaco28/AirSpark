"""
Unit tests for Spark ML AQI Forecasting module.
Validates chronological split, temporal feature extraction, model training, persistence, and loading.
"""
import tempfile
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel
from app.ml.prediction import AQIPredictionModel


def test_chronological_train_test_split(spark_session: SparkSession):
    # Create sequential timestamps for 10 records
    data = []
    for i in range(10):
        ts = f"2026-01-01 {i:02d}:00:00"
        data.append(("STN_001", ts, 20.0 + i, 40.0 + i, 10.0, 5.0, 0.5, 20.0, 50.0 + i))
    
    df = spark_session.createDataFrame(
        data, 
        ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]
    )
    
    model = AQIPredictionModel()
    train_df, test_df = model.prepare_time_split(df, train_ratio=0.8)
    
    train_max_ts = train_df.agg({"timestamp": "max"}).collect()[0][0]
    test_min_ts = test_df.agg({"timestamp": "min"}).collect()[0][0]
    
    # Chronological guarantee: train timestamps must be strictly earlier than test timestamps
    assert train_max_ts <= test_min_ts
    assert train_df.count() == 8
    assert test_df.count() == 2


def test_temporal_feature_generation(spark_session: SparkSession):
    data = [
        ("STN_001", "2026-01-01 00:00:00", 25.0, 50.0, 10.0, 5.0, 0.5, 20.0, 50.0),
        ("STN_001", "2026-01-01 01:00:00", 35.0, 70.0, 12.0, 6.0, 0.6, 22.0, 70.0),
        ("STN_001", "2026-01-01 02:00:00", 45.0, 90.0, 15.0, 8.0, 0.8, 25.0, 90.0),
    ]
    df = spark_session.createDataFrame(
        data, 
        ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]
    )
    
    model = AQIPredictionModel()
    feat_df = model.extract_features(df)
    
    assert "hour" in feat_df.columns
    assert "day_of_week" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "prev_aqi" in feat_df.columns
    assert "prev_pm2_5" in feat_df.columns
    assert "rolling_pm2_5_mean" in feat_df.columns


def test_spark_ml_training_evaluation_and_persistence(spark_session: SparkSession):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        
        # Generate sequential dataset for 30 hours
        data = []
        for hour in range(30):
            ts = f"2026-01-01 {hour % 24:02d}:00:00"
            data.append(("STN_001", ts, 20.0 + (hour % 10), 40.0, 10.0, 5.0, 0.5, 20.0, 50.0 + (hour % 10)))
        
        df = spark_session.createDataFrame(
            data, 
            ["station_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"]
        )
        
        model = AQIPredictionModel()
        metrics, preds_df = model.train_and_evaluate(df, models_save_dir=str(models_dir))
        
        # Verify metrics structure
        assert "linear_regression" in metrics["models"]
        assert "random_forest" in metrics["models"]
        assert "rmse" in metrics["models"]["linear_regression"]
        assert "mae" in metrics["models"]["linear_regression"]
        assert "r2" in metrics["models"]["linear_regression"]
        
        # Verify predictions
        assert preds_df.count() > 0
        assert "actual_aqi" in preds_df.columns
        assert "predicted_aqi" in preds_df.columns
        assert "model" in preds_df.columns
        
        # Verify model persistence and reloading
        lr_path = models_dir / "linear_regression"
        rf_path = models_dir / "random_forest"
        assert lr_path.exists()
        assert rf_path.exists()
        
        loaded_lr = AQIPredictionModel.load_model(spark_session, str(lr_path), model_type="linear_regression")
        loaded_rf = AQIPredictionModel.load_model(spark_session, str(rf_path), model_type="random_forest")
        assert loaded_lr is not None
        assert loaded_rf is not None
