"""
Task 2: Latency Forecasting - Network Monitoring MLOps
Time series forecasting for network latency using multiple approaches
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available. Install with: pip install statsmodels")

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    
    LSTM = tf.keras.layers.LSTM
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")

import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt

# MLflow Configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")


class LatencyForecastingExperiment:
    """Complete latency forecasting experiment with multiple time series approaches."""

    def __init__(self, data_path="data/processed/cloud_network_performance.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.load_data()
        self.prepare_time_series()

    def load_data(self):
        """Load and prepare network monitoring data for time series analysis."""
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(self.df):,} records with {len(self.df.columns)} features")

        if "experiment_datetime" in self.df.columns:
            self.df["experiment_datetime"] = pd.to_datetime(self.df["experiment_datetime"])
        else:
            raise ValueError("❌ experiment_datetime column not found!")

        self.df = self.df.sort_values("experiment_datetime").reset_index(drop=True)
        print(f"📅 Time range: {self.df['experiment_datetime'].min()} → {self.df['experiment_datetime'].max()}")

        provider_dist = self.df["provider"].value_counts()
        print(f"🏢 Provider distribution: {dict(provider_dist)}")

    def prepare_time_series(self):
        """Prepare data for time series forecasting."""
        print("🔧 Preparing time series data...")
        os.makedirs("models", exist_ok=True)

        if "duration" not in self.df.columns:
            raise ValueError("❌ Duration column not found for latency forecasting!")

        # Time-based features
        self.df["hour"] = self.df["experiment_datetime"].dt.hour
        self.df["day_of_week"] = self.df["experiment_datetime"].dt.dayofweek
        self.df["month"] = self.df["experiment_datetime"].dt.month

        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            self.df[f"duration_lag_{lag}"] = self.df["duration"].shift(lag)

        # Rolling statistics
        for window in [3, 6, 12, 24]:
            self.df[f"duration_rolling_mean_{window}"] = self.df["duration"].rolling(window=window).mean()
            self.df[f"duration_rolling_std_{window}"] = self.df["duration"].rolling(window=window).std()

        self.df = self.df.dropna().reset_index(drop=True)
        print(f"✅ Time series prepared. Final dataset: {len(self.df):,} records")

        # Encode categorical features
        self.encoders = {}
        for col in ["provider", "vm_size"]:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f"{col}_encoded"] = le.fit_transform(self.df[col])
                self.encoders[col] = le

        with open("models/latency_encoders.pkl", "wb") as f:
            pickle.dump(self.encoders, f)

    def run_arima_experiment(self):
        """ARIMA model for latency forecasting."""
        if not STATSMODELS_AVAILABLE:
            print("❌ Skipping ARIMA - statsmodels not available")
            return

        mlflow.set_experiment("latency_forecasting")
        ts_data = self.df.set_index("experiment_datetime")["duration"].resample("H").mean().ffill()

        split_point = int(len(ts_data) * 0.8)
        train_data, test_data = ts_data[:split_point], ts_data[split_point:]

        with mlflow.start_run(run_name=f"arima_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            order = (2, 1, 2)
            mlflow.log_param("arima_order", str(order))

            try:
                model = ARIMA(train_data, order=order).fit()
                forecast = model.forecast(steps=len(test_data))

                mse = mean_squared_error(test_data, forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test_data, forecast)

                mlflow.log_metric("test_mse", mse)
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mae", mae)

                with open("models/arima_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                mlflow.log_artifact("models/arima_model.pkl", "models")

                self.results["arima"] = {"rmse": rmse, "mae": mae, "mse": mse}
                print(f"✅ ARIMA Results → RMSE: {rmse:.4f}, MAE: {mae:.4f}")

            except Exception as e:
                print(f"❌ ARIMA failed: {e}")
                mlflow.log_param("error", str(e))

    def run_random_forest_time_series(self):
        """Random Forest with time series features for latency forecasting."""
        mlflow.set_experiment("latency_forecasting")
        features = [
            "hour", "day_of_week", "month",
            "duration_lag_1", "duration_lag_2", "duration_lag_3",
            "duration_rolling_mean_3", "duration_rolling_mean_6",
            "provider_encoded", "vm_size_encoded"
        ]
        features = [f for f in features if f in self.df.columns]
        X, y = self.df[features], self.df["duration"]

        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(X))[-1]
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

        with mlflow.start_run(run_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.sklearn.log_model(model, "model")

            self.results["random_forest"] = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"✅ RF Results → RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def run_lstm_experiment(self):
        """LSTM model for latency forecasting."""
        if not TENSORFLOW_AVAILABLE:
            print("❌ Skipping LSTM - TensorFlow not available")
            return

        mlflow.set_experiment("latency_forecasting")
        sequence_length = 24

        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.df[["duration"]])
        X, y = [], []
        for i in range(sequence_length, len(scaled)):
            X.append(scaled[i-sequence_length:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        with mlflow.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=50, batch_size=32, verbose=0, callbacks=[callback])

            y_pred_scaled = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)

            model.save("models/lstm_model")
            mlflow.log_artifacts("models/lstm_model", "lstm_model")

            self.results["lstm"] = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"✅ LSTM Results → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    def run_linear_trend_experiment(self):
        """Linear regression baseline."""
        mlflow.set_experiment("latency_forecasting")
        self.df["time_index"] = range(len(self.df))
        features = ["time_index", "hour", "day_of_week"]
        if "provider_encoded" in self.df.columns:
            features.append("provider_encoded")

        X, y = self.df[features], self.df["duration"]
        split = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

        with mlflow.start_run(run_name=f"linear_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.sklearn.log_model(model, "model")

            self.results["linear"] = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"✅ Linear Trend Results → RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def run_all_experiments(self):
        print("🚀 Running all latency forecasting models\n" + "=" * 50)
        self.run_linear_trend_experiment()
        self.run_random_forest_time_series()
        self.run_arima_experiment()
        self.run_lstm_experiment()

        print("\n📊 Results Summary")
        for name, res in self.results.items():
            print(f"{name:12} → RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}, R²: {res.get('r2','-')}")

        if self.results:
            best = min(self.results, key=lambda k: self.results[k]["rmse"])
            print(f"\n🏆 Best Model: {best} (RMSE={self.results[best]['rmse']:.4f})")

        return self.results


def main():
    print("🎯 Latency Forecasting MLOps Pipeline\n" + "=" * 60)
    try:
        import requests
        requests.get("http://127.0.0.1:5000", timeout=5)
    except Exception:
        print("❌ MLflow server not running!\nStart it with:")
        print("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local --host 127.0.0.1 --port 5000")
        return

    experiment = LatencyForecastingExperiment()
    experiment.run_all_experiments()


if __name__ == "__main__":
    main()
