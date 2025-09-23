"""
Task 4: Performance Classification - Network Monitoring MLOps
Classification of network performance levels and provider comparison
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# MLflow Configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class PerformanceClassificationExperiment:
    """Complete performance classification experiment for network monitoring."""
    
    def __init__(self, data_path="data/processed/cloud_network_performance_20250918_174137.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.encoders = {}
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        """Load and explore network monitoring data."""
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        
        # Show provider distribution (our imbalance challenge)
        provider_dist = self.df['provider'].value_counts()
        print(f"🏢 Provider distribution: {dict(provider_dist)}")
        
        # Show protocol distribution
        protocol_dist = self.df['protocol'].value_counts()
        print(f"🌐 Protocol distribution: {dict(protocol_dist)}")
        
        # Focus on TCP throughput for consistency
        self.df = self.df[self.df['protocol'] == 'TCP_Throughput'].copy()
        print(f"🎯 TCP Throughput records: {len(self.df):,}")
        
    def prepare_features(self):
        """Feature engineering for classification tasks."""
        print("🔧 Preparing features for classification...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Encode categorical features
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
        
        # Performance efficiency metrics
        if 'target_bwd' in self.df.columns and 'duration' in self.df.columns:
            self.df['throughput_efficiency'] = self.df['target_bwd'] / (self.df['duration'] + 1)
        
        # Save encoders
        with open('models/classification_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        print(f"✅ Feature preparation completed. Dataset shape: {self.df.shape}")
    
    def create_performance_labels(self, target_metric='target_bwd', method='quartiles'):
        """Create performance classification labels."""
        
        if method == 'quartiles':
            # Quartile-based classification
            quartiles = self.df[target_metric].quantile([0.25, 0.5, 0.75])
            
            conditions = [
                self.df[target_metric] <= quartiles[0.25],
                (self.df[target_metric] > quartiles[0.25]) & (self.df[target_metric] <= quartiles[0.5]),
                (self.df[target_metric] > quartiles[0.5]) & (self.df[target_metric] <= quartiles[0.75]),
                self.df[target_metric] > quartiles[0.75]
            ]
            
            choices = ['Poor', 'Fair', 'Good', 'Excellent']
            
        elif method == 'binary_provider':
            # Binary classification: AWS vs Azure
            return self.df['provider']
            
        elif method == 'vm_performance':
            # VM size performance classification
            return self.df['vm_size']
            
        elif method == 'adaptive_thresholds':
            # Adaptive thresholds based on provider-specific statistics
            performance_labels = []
            
            for provider in self.df['provider'].unique():
                provider_data = self.df[self.df['provider'] == provider][target_metric]
                provider_median = provider_data.median()
                provider_std = provider_data.std()
                
                provider_mask = self.df['provider'] == provider
                provider_values = self.df.loc[provider_mask, target_metric]
                
                # Provider-specific thresholds
                low_threshold = provider_median - 0.5 * provider_std
                high_threshold = provider_median + 0.5 * provider_std
                
                provider_labels = pd.cut(
                    provider_values,
                    bins=[-np.inf, low_threshold, high_threshold, np.inf],
                    labels=['Below_Average', 'Average', 'Above_Average']
                )
                
                performance_labels.extend(provider_labels)
            
            return pd.Series(performance_labels, index=self.df.index)
        
        self.df['performance_class'] = np.select(conditions, choices)
        return self.df['performance_class']
    
    def get_balanced_dataset(self, X, y, method='undersample'):
        """Handle class imbalance for training."""
        
        if method == 'undersample':
            # Random undersampling
            undersampler = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
            
        elif method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        elif method == 'none':
            X_balanced, y_balanced = X, y
        
        print(f"🔄 Balancing method: {method}")
        print(f"   Original shape: {X.shape}")
        print(f"   Balanced shape: {X_balanced.shape}")
        
        if hasattr(y_balanced, 'value_counts'):
            print(f"   Balanced classes: {dict(y_balanced.value_counts())}")
        
        return X_balanced, y_balanced
    
    def run_provider_classification_experiment(self):
        """Binary classification: AWS vs Azure performance prediction."""
        mlflow.set_experiment("performance_classification")
        
        # Prepare data for provider classification
        target = self.create_performance_labels(method='binary_provider')
        
        feature_cols = [
            'vm_size_encoded', 'source_region_encoded', 'dest_region_encoded',
            'duration', 'pkt_size', 'port', 'duration_log', 'bwd_per_packet',
            'cross_region', 'throughput_efficiency'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = target
        
        # Handle AWS/Azure imbalance
        X_balanced, y_balanced = self.get_balanced_dataset(X, y, method='undersample')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        with mlflow.start_run(run_name=f"provider_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "random_forest")
            mlflow.set_tag("task", "provider_classification")
            mlflow.set_tag("target", "provider")
            mlflow.set_tag("balancing", "undersample")
            mlflow.set_tag("developer", "neha")
            
            # Parameters
            n_estimators = 100
            max_depth = 15
            class_weight = 'balanced'
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("class_weight", class_weight)
            mlflow.log_param("features", available_features)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # ROC AUC for binary classification
            try:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                auc = 0.5
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Feature importance
            feature_importance = dict(zip(available_features, model.feature_importances_))
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = "models/provider_classification_report.json"
            
            with open(report_path, 'w') as f:
                import json
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path, "reports")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("models/classification_encoders.pkl", "preprocessors")
            
            self.results['provider_classification'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            }
            
            print(f"✅ Provider Classification Results:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   ROC-AUC: {auc:.4f}")
            print(f"   Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    def run_performance_level_classification(self):
        """Multi-class classification: Poor/Fair/Good/Excellent performance."""
        mlflow.set_experiment("performance_classification")
        
        # Create performance labels
        target = self.create_performance_labels(target_metric='target_bwd', method='quartiles')
        
        feature_cols = [
            'provider_encoded', 'vm_size_encoded', 'source_region_encoded',
            'duration', 'pkt_size', 'port', 'duration_log', 'bwd_per_packet',
            'cross_region', 'throughput_efficiency'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = target
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name=f"performance_levels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "gradient_boosting")
            mlflow.set_tag("task", "performance_levels")
            mlflow.set_tag("target", "performance_quartiles")
            mlflow.set_tag("developer", "neha")
            
            # Parameters
            n_estimators = 100
            learning_rate = 0.1
            max_depth = 6
            
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("features", available_features)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Per-class metrics
            class_report = classification_report(y_test, y_pred, output_dict=True)
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{class_name}_{metric_name}", value)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            self.results['performance_levels'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"✅ Performance Levels Classification:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
    
    def run_xgboost_classification_experiment(self):
        """XGBoost classification for provider performance prediction."""
        mlflow.set_experiment("performance_classification")
        
        # Provider classification task
        target = self.create_performance_labels(method='binary_provider')
        
        feature_cols = [
            'vm_size_encoded', 'source_region_encoded', 'dest_region_encoded',
            'duration', 'pkt_size', 'port', 'duration_log', 'bwd_per_packet',
            'throughput_efficiency'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = LabelEncoder().fit_transform(target)  # XGBoost needs numeric labels
        
        # Balance dataset
        X_balanced, y_balanced = self.get_balanced_dataset(X, y, method='smote')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name=f"xgboost_provider_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "xgboost")
            mlflow.set_tag("task", "provider_classification")
            mlflow.set_tag("balancing", "smote")
            
            # Parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': 6,
                'learning_rate': 0.1,
                'reg_alpha': 0.01,
                'reg_lambda': 0.01,
                'seed': 42
            }
            
            mlflow.log_params(params)
            mlflow.log_param("num_boost_round", 100)
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            # Train model
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'eval')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Predictions
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            self.results['xgboost_provider'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            }
            
            print(f"✅ XGBoost Provider Classification:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   ROC-AUC: {auc:.4f}")
    
    def run_vm_size_classification_experiment(self):
        """VM size performance classification."""
        mlflow.set_experiment("performance_classification")
        
        # VM size classification
        target = self.create_performance_labels(method='vm_performance')
        
        feature_cols = [
            'provider_encoded', 'source_region_encoded', 'dest_region_encoded',
            'target_bwd', 'duration', 'pkt_size', 'port', 
            'bwd_per_packet', 'throughput_efficiency'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = target
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        with mlflow.start_run(run_name=f"vm_size_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "svm")
            mlflow.set_tag("task", "vm_size_classification")
            mlflow.set_tag("target", "vm_size")
            
            # Parameters
            C = 1.0
            kernel = 'rbf'
            gamma = 'scale'
            
            mlflow.log_param("C", C)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("gamma", gamma)
            mlflow.log_param("scaled_features", True)
            
            # Train model
            model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model and scaler
            mlflow.sklearn.log_model(model, "model")
            
            scaler_path = "models/vm_classification_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessors")
            
            self.results['vm_size_classification'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"✅ VM Size Classification (SVM):")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   F1-Score: {f1:.4f}")
    
    def run_cross_validation_experiment(self):
        """Cross-validation experiment for robust performance estimation."""
        mlflow.set_experiment("performance_classification")
        
        # Use provider classification as main task
        target = self.create_performance_labels(method='binary_provider')
        
        feature_cols = [
            'vm_size_encoded', 'source_region_encoded', 'dest_region_encoded',
            'duration', 'pkt_size', 'port', 'bwd_per_packet'
        ]
        
        available_features = [col for col in feature_cols if col in self.df.columns]
        X = self.df[available_features].fillna(0)
        y = target
        
        # Balance dataset
        X_balanced, y_balanced = self.get_balanced_dataset(X, y, method='undersample')
        
        with mlflow.start_run(run_name=f"cross_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "logistic_regression")
            mlflow.set_tag("task", "cross_validation")
            mlflow.set_tag("method", "stratified_kfold")
            
            # Parameters
            cv_folds = 5
            C = 1.0
            max_iter = 1000
            
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("class_weight", "balanced")
            
            # Model
            model = LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight='balanced',
                random_state=42
            )
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=cv, scoring='accuracy')
            
            # Log CV results
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())
            mlflow.log_metric("cv_min_accuracy", cv_scores.min())
            mlflow.log_metric("cv_max_accuracy", cv_scores.max())
            
            for i, score in enumerate(cv_scores):
                mlflow.log_metric(f"fold_{i+1}_accuracy", score)
            
            # Train final model on full dataset
            model.fit(X_balanced, y_balanced)
            mlflow.sklearn.log_model(model, "model")
            
            self.results['cross_validation'] = {
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            print(f"✅ Cross-Validation Results:")
            print(f"   Mean Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"   Score Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")
    
    def run_all_experiments(self):
        """Run all performance classification experiments."""
        print("🚀 Starting Performance Classification Experiments")
        print("=" * 50)
        
        # Run all classification tasks
        self.run_provider_classification_experiment()
        self.run_performance_level_classification()
        self.run_xgboost_classification_experiment()
        self.run_vm_size_classification_experiment()
        self.run_cross_validation_experiment()
        
        # Summary
        print("\n📊 Experiment Summary:")
        print("-" * 60)
        for model_name, results in self.results.items():
            accuracy = results.get('accuracy', results.get('mean_accuracy', 'N/A'))
            f1 = results.get('f1_score', 'N/A')
            print(f"{model_name:25} | Accuracy: {accuracy:>8} | F1: {f1:>8}")
        
        # Find best model
        if self.results:
            valid_results = {k: v for k, v in self.results.items() if 'accuracy' in v or 'mean_accuracy' in v}
            if valid_results:
                best_model = max(valid_results.keys(), 
                               key=lambda k: valid_results[k].get('accuracy', valid_results[k].get('mean_accuracy', 0)))
                best_score = valid_results[best_model].get('accuracy', valid_results[best_model].get('mean_accuracy'))
                print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_score:.4f})")
        
        print(f"\n🌐 View results at: http://127.0.0.1:5000")
        
        return self.results

def main():
    """Main execution function."""
    print("🎯 Performance Classification MLOps Pipeline")
    print("Following ZoomCamp methodology for network performance classification")
    print("=" * 70)
    
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
    experiment = PerformanceClassificationExperiment()
    results = experiment.run_all_experiments()
    
    print("\n✅ Performance classification experiments completed!")
    return results

if __name__ == "__main__":
    main()
