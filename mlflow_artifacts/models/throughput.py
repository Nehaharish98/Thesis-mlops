"""
FIXED - Task 1: Throughput Prediction - Network Monitoring MLOps
Following ZoomCamp methodology for throughput prediction experiments
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MLflow Configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class ThroughputPredictionExperiment:
    """Complete throughput prediction experiment with multiple models."""
    
    def __init__(self, data_path="data/processed/cloud_network_performance_20250918_174137.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        """Load and explore network monitoring data."""
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        
        # Focus on TCP throughput data (your dominant 73.8%)
        self.df = self.df[self.df['protocol'] == 'TCP_Throughput'].copy()
        print(f"🎯 TCP Throughput records: {len(self.df):,}")
        
        # Show provider distribution
        provider_dist = self.df['provider'].value_counts()
        print(f"🏢 Provider distribution: {dict(provider_dist)}")
        
    def prepare_features(self):
        """Feature engineering for throughput prediction."""
        print("🔧 Preparing features...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Label encoding for categorical features
        self.encoders = {}
        categorical_features = ['provider', 'vm_size', 'source_region', 'dest_region', 'tool']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_encoded'] = le.fit_transform(self.df[feature].fillna('unknown'))
                self.encoders[feature] = le
        
        # Feature engineering
        if 'target_bwd' in self.df.columns and 'pkt_size' in self.df.columns:
            self.df['bwd_per_packet'] = self.df['target_bwd'] / (self.df['pkt_size'] + 1)
        
        if 'duration' in self.df.columns:
            self.df['duration_log'] = np.log1p(self.df['duration'])
        
        # Regional connectivity feature
        if 'source_region' in self.df.columns and 'dest_region' in self.df.columns:
            self.df['cross_region'] = (self.df['source_region'] != self.df['dest_region']).astype(int)
        
        # Save encoders
        with open('models/throughput_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        print(f"✅ Feature engineering completed. Dataset shape: {self.df.shape}")
    
    def get_features_and_target(self, balance_providers=True):
        """Get features and target variable."""
        # Handle class imbalance if requested
        if balance_providers and 'provider' in self.df.columns:
            # Balance AWS (91.7%) vs Azure (8.3%) 
            azure_count = (self.df['provider'] == 'Azure').sum()
            aws_data = self.df[self.df['provider'] == 'AWS'].sample(n=azure_count, random_state=42)
            azure_data = self.df[self.df['provider'] == 'Azure']
            balanced_df = pd.concat([aws_data, azure_data]).reset_index(drop=True)
            print(f"⚖️ Balanced dataset: {len(balanced_df):,} records")
        else:
            balanced_df = self.df.copy()
        
        # Select features for modeling
        feature_columns = [
            'provider_encoded', 'vm_size_encoded', 'source_region_encoded', 
            'dest_region_encoded', 'duration', 'pkt_size', 'port', 
            'duration_log', 'bwd_per_packet', 'cross_region'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in balanced_df.columns]
        print(f"🎯 Available features: {available_features}")
        
        X = balanced_df[available_features].fillna(0)
        y = balanced_df['target_bwd'].fillna(0)
        
        return X, y, available_features
    
    def run_random_forest_experiment(self):
        """Random Forest throughput prediction experiment."""
        mlflow.set_experiment("throughput_prediction")
        
        X, y, features = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(run_name=f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "random_forest")
            mlflow.set_tag("task", "throughput_prediction")
            mlflow.set_tag("developer", "neha")
            mlflow.set_tag("data_balanced", "yes")
            
            # Parameters
            n_estimators = 100
            max_depth = 20
            min_samples_split = 5
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("features_count", len(features))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Model training
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions and metrics
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Training metrics - FIXED: Remove squared parameter
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_r2 = r2_score(y_train, y_pred_train)
            
            # Test metrics - FIXED: Remove squared parameter
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log metrics
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("overfitting_gap", train_r2 - test_r2)
            
            # Feature importance
            feature_importance = dict(zip(features, model.feature_importances_))
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
            
            # Log model and artifacts
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("models/throughput_encoders.pkl", "preprocessors")
            
            # Store results
            self.models['random_forest'] = model
            self.results['random_forest'] = {
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            print(f"✅ Random Forest Results:")
            print(f"   Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
            print(f"   Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def run_xgboost_experiment(self):
        """XGBoost throughput prediction with hyperparameter tuning."""
        mlflow.set_experiment("throughput_prediction")
        
        X, y, features = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "xgboost")
            mlflow.set_tag("task", "throughput_prediction")
            mlflow.set_tag("developer", "neha")
            
            # Parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'min_child_weight': 1,
                'seed': 42
            }
            
            mlflow.log_params(params)
            mlflow.log_param("num_boost_round", 100)
            mlflow.log_param("early_stopping_rounds", 20)
            
            # Model training
            evals = [(dtrain, 'train'), (dtest, 'eval')]
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100,
                evals=evals,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Predictions and metrics
            y_pred_test = model.predict(dtest)
            
            # FIXED: Remove squared parameter
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_artifact("models/throughput_encoders.pkl", "preprocessors")
            
            # Store results
            self.models['xgboost'] = model
            self.results['xgboost'] = {
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            print(f"✅ XGBoost Results:")
            print(f"   Test RMSE: {test_rmse:.4f}")
            print(f"   Test R²: {test_r2:.4f}")
    
    def run_lightgbm_experiment(self):
        """LightGBM throughput prediction experiment."""
        mlflow.set_experiment("throughput_prediction")
        
        X, y, features = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run(run_name=f"lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "lightgbm")
            mlflow.set_tag("task", "throughput_prediction")
            mlflow.set_tag("developer", "neha")
            
            # Parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            mlflow.log_params(params)
            
            # Model training
            model = lgb.LGBMRegressor(**params, n_estimators=100)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            
            # Predictions and metrics
            y_pred_test = model.predict(X_test)
            
            # FIXED: Remove squared parameter
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_r2 = r2_score(y_test, y_pred_test)
            
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store results
            self.models['lightgbm'] = model
            self.results['lightgbm'] = {
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            }
            
            print(f"✅ LightGBM Results:")
            print(f"   Test RMSE: {test_rmse:.4f}")
            print(f"   Test R²: {test_r2:.4f}")
    
    def run_linear_models_experiment(self):
        """Linear models (Ridge, Lasso) for throughput prediction."""
        mlflow.set_experiment("throughput_prediction")
        
        X, y, features = self.get_features_and_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ridge Regression
        with mlflow.start_run(run_name=f"ridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.set_tag("model_type", "ridge")
            mlflow.set_tag("task", "throughput_prediction")
            
            alpha = 1.0
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("scaled_features", True)
            
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred_test = model.predict(X_test_scaled)
            
            # FIXED: Remove squared parameter
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_r2 = r2_score(y_test, y_pred_test)
            
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_r2", test_r2)
            
            mlflow.sklearn.log_model(model, "model")
            
            # Save scaler
            with open('models/throughput_scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact("models/throughput_scaler.pkl", "preprocessors")
            
            print(f"✅ Ridge Regression - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    def run_all_experiments(self):
        """Run all throughput prediction experiments."""
        print("🚀 Starting Throughput Prediction Experiments")
        print("=" * 50)
        
        # Run all models
        self.run_random_forest_experiment()
        self.run_xgboost_experiment()
        self.run_lightgbm_experiment()
        self.run_linear_models_experiment()
        
        # Summary
        print("\n📊 Experiment Summary:")
        print("-" * 30)
        for model_name, results in self.results.items():
            print(f"{model_name:12} | RMSE: {results['test_rmse']:.4f} | R²: {results['test_r2']:.4f}")
        
        # Find best model
        if self.results:
            best_model = min(self.results.keys(), key=lambda k: self.results[k]['test_rmse'])
            print(f"\n🏆 Best Model: {best_model} (RMSE: {self.results[best_model]['test_rmse']:.4f})")
        
        print(f"\n🌐 View results at: http://127.0.0.1:5000")
        
        return self.results

def main():
    """Main execution function."""
    print("🎯 Throughput Prediction MLOps Pipeline")
    print("Following ZoomCamp methodology for network monitoring")
    print("=" * 60)
    
    # Check if MLflow server is running
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        if response.status_code != 200:
            raise Exception("MLflow server not responding")
    except:
        print("❌ MLflow server not running!")
        print("Please start: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts_local --host 127.0.0.1 --port 5000")
        return
    
    # Run experiments
    experiment = ThroughputPredictionExperiment()
    results = experiment.run_all_experiments()
    
    print("\n✅ Throughput prediction experiments completed!")
    return results

if __name__ == "__main__":
    main()
