"""
Task 3: Anomaly Detection - Network Monitoring MLOps
Network performance anomaly detection using multiple unsupervised learning approaches
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# MLflow Configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class AnomalyDetectionExperiment:
    """Complete anomaly detection experiment for network performance monitoring."""
    
    def __init__(self, data_path="data/processed/cloud_network_performance_20250918_174137.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.load_data()
        self.prepare_features()
        
    def load_data(self):
        """Load and explore network monitoring data."""
        self.df = pd.read_csv(self.data_path)
        print(f"📊 Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        
        # Show basic statistics
        print(f"📅 Time range: {self.df['experiment_datetime'].min()} to {self.df['experiment_datetime'].max()}")
        provider_dist = self.df['provider'].value_counts()
        print(f"🏢 Provider distribution: {dict(provider_dist)}")
        
        protocol_dist = self.df['protocol'].value_counts()
        print(f"🌐 Protocol distribution: {dict(protocol_dist)}")
        
    def prepare_features(self):
        """Feature engineering for anomaly detection."""
        print("🔧 Preparing features for anomaly detection...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Encode categorical features
        self.encoders = {}
        categorical_features = ['provider', 'vm_size', 'protocol', 'source_region', 'dest_region', 'tool']
        
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_encoded'] = le.fit_transform(self.df[feature].fillna('unknown'))
                self.encoders[feature] = le
        
        # Feature engineering for anomaly detection
        if 'target_bwd' in self.df.columns and 'pkt_size' in self.df.columns:
            self.df['bwd_per_packet'] = self.df['target_bwd'] / (self.df['pkt_size'] + 1)
            
        if 'duration' in self.df.columns:
            self.df['duration_log'] = np.log1p(self.df['duration'])
            
        # Network performance ratios
        if 'target_bwd' in self.df.columns and 'duration' in self.df.columns:
            self.df['bwd_duration_ratio'] = self.df['target_bwd'] / (self.df['duration'] + 1)
        
        # Statistical features for each provider
        if 'provider' in self.df.columns:
            for metric in ['target_bwd', 'duration', 'pkt_size']:
                if metric in self.df.columns:
                    provider_stats = self.df.groupby('provider')[metric].transform(lambda x: (x - x.mean()) / x.std())
                    self.df[f'{metric}_provider_zscore'] = provider_stats
        
        # Save encoders
        with open('models/anomaly_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        print(f"✅ Feature preparation completed. Dataset shape: {self.df.shape}")
        
    def get_features_for_anomaly_detection(self, feature_set="basic"):
        """Get feature matrix for anomaly detection."""
        
        if feature_set == "basic":
            # Basic network performance features
            feature_cols = ['target_bwd', 'duration', 'pkt_size', 'port']
            
        elif feature_set == "extended":
            # Extended features with engineering
            feature_cols = [
                'target_bwd', 'duration', 'pkt_size', 'port',
                'bwd_per_packet', 'duration_log', 'bwd_duration_ratio',
                'provider_encoded', 'vm_size_encoded', 'protocol_encoded'
            ]
            
        elif feature_set == "provider_specific":
            # Provider-specific statistical features
            feature_cols = [
                'target_bwd', 'duration', 'pkt_size',
                'target_bwd_provider_zscore', 'duration_provider_zscore', 'pkt_size_provider_zscore'
            ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in self.df.columns]
        print(f"🎯 Using features for {feature_set} anomaly detection: {available_features}")
        
        X = self.df[available_features].fillna(0)
        return X, available_features
    
    def run_isolation_forest_experiment(self):
        """Isolation Forest for network anomaly detection."""
        mlflow.set_experiment("anomaly_detection")
        
        X, features = self.get_features_for_anomaly_detection("extended")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with mlflow.start_run(run_name=f"isolation_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "isolation_forest")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("developer", "neha")
            mlflow.set_tag("feature_set", "extended")
            
            # Parameters
            contamination = 0.05  # Expect 5% anomalies
            n_estimators = 100
            max_samples = 'auto'
            
            mlflow.log_param("contamination", contamination)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_samples", max_samples)
            mlflow.log_param("features", features)
            mlflow.log_param("n_samples", len(X))
            
            # Train model
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42,
                n_jobs=-1
            )
            
            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            
            # Calculate metrics
            n_anomalies = (anomaly_labels == -1).sum()
            anomaly_rate = n_anomalies / len(anomaly_labels) * 100
            
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
            mlflow.log_metric("mean_anomaly_score", np.mean(anomaly_scores))
            mlflow.log_metric("std_anomaly_score", np.std(anomaly_scores))
            
            # Provider-wise anomaly analysis
            self.df['anomaly_if'] = anomaly_labels
            provider_anomalies = self.df.groupby('provider')['anomaly_if'].apply(lambda x: (x == -1).sum()).to_dict()
            for provider, count in provider_anomalies.items():
                mlflow.log_metric(f"anomalies_{provider.lower()}", count)
            
            # Log model and scaler
            mlflow.sklearn.log_model(model, "model")
            
            scaler_path = "models/isolation_forest_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessors")
            mlflow.log_artifact("models/anomaly_encoders.pkl", "preprocessors")
            
            # Store results
            self.models['isolation_forest'] = model
            self.scalers['isolation_forest'] = scaler
            self.results['isolation_forest'] = {
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies,
                'provider_anomalies': provider_anomalies
            }
            
            print(f"✅ Isolation Forest Results:")
            print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
            print(f"   Provider anomalies: {provider_anomalies}")
            
    def run_one_class_svm_experiment(self):
        """One-Class SVM for network anomaly detection."""
        mlflow.set_experiment("anomaly_detection")
        
        X, features = self.get_features_for_anomaly_detection("basic")
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X)
        
        with mlflow.start_run(run_name=f"oneclass_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "oneclass_svm")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("developer", "neha")
            mlflow.set_tag("feature_set", "basic")
            
            # Parameters
            nu = 0.05  # Expected fraction of anomalies
            kernel = 'rbf'
            gamma = 'scale'
            
            mlflow.log_param("nu", nu)
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("gamma", gamma)
            mlflow.log_param("features", features)
            mlflow.log_param("scaler", "robust")
            
            # Train model
            model = OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma=gamma
            )
            
            anomaly_labels = model.fit_predict(X_scaled)
            
            # Calculate metrics
            n_anomalies = (anomaly_labels == -1).sum()
            anomaly_rate = n_anomalies / len(anomaly_labels) * 100
            
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
            
            # Provider-wise analysis
            self.df['anomaly_svm'] = anomaly_labels
            provider_anomalies = self.df.groupby('provider')['anomaly_svm'].apply(lambda x: (x == -1).sum()).to_dict()
            for provider, count in provider_anomalies.items():
                mlflow.log_metric(f"anomalies_{provider.lower()}", count)
            
            # Log model and scaler
            mlflow.sklearn.log_model(model, "model")
            
            scaler_path = "models/oneclass_svm_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessors")
            
            self.results['oneclass_svm'] = {
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies,
                'provider_anomalies': provider_anomalies
            }
            
            print(f"✅ One-Class SVM Results:")
            print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
            print(f"   Provider anomalies: {provider_anomalies}")
    
    def run_local_outlier_factor_experiment(self):
        """Local Outlier Factor for network anomaly detection."""
        mlflow.set_experiment("anomaly_detection")
        
        X, features = self.get_features_for_anomaly_detection("extended")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with mlflow.start_run(run_name=f"lof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "local_outlier_factor")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("developer", "neha")
            
            # Parameters
            n_neighbors = 20
            contamination = 0.05
            
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("contamination", contamination)
            mlflow.log_param("features", features)
            
            # Train model
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                n_jobs=-1
            )
            
            anomaly_labels = model.fit_predict(X_scaled)
            outlier_scores = model.negative_outlier_factor_
            
            # Calculate metrics
            n_anomalies = (anomaly_labels == -1).sum()
            anomaly_rate = n_anomalies / len(anomaly_labels) * 100
            
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
            mlflow.log_metric("mean_outlier_score", np.mean(outlier_scores))
            
            # Provider-wise analysis
            self.df['anomaly_lof'] = anomaly_labels
            provider_anomalies = self.df.groupby('provider')['anomaly_lof'].apply(lambda x: (x == -1).sum()).to_dict()
            for provider, count in provider_anomalies.items():
                mlflow.log_metric(f"anomalies_{provider.lower()}", count)
            
            # Note: LOF doesn't have a predict method, so we can't save it the same way
            # We'll save the outlier scores instead
            scores_path = "models/lof_outlier_scores.pkl"
            with open(scores_path, 'wb') as f:
                pickle.dump(outlier_scores, f)
            mlflow.log_artifact(scores_path, "models")
            
            self.results['local_outlier_factor'] = {
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies,
                'provider_anomalies': provider_anomalies
            }
            
            print(f"✅ Local Outlier Factor Results:")
            print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
            print(f"   Provider anomalies: {provider_anomalies}")
    
    def run_dbscan_experiment(self):
        """DBSCAN clustering for network anomaly detection."""
        mlflow.set_experiment("anomaly_detection")
        
        X, features = self.get_features_for_anomaly_detection("provider_specific")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with mlflow.start_run(run_name=f"dbscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "dbscan")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("developer", "neha")
            mlflow.set_tag("approach", "clustering")
            
            # Parameters
            eps = 0.5
            min_samples = 10
            
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.log_param("features", features)
            
            # Train model
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            cluster_labels = model.fit_predict(X_scaled)
            
            # Identify anomalies (noise points with label -1)
            n_anomalies = (cluster_labels == -1).sum()
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            anomaly_rate = n_anomalies / len(cluster_labels) * 100
            
            mlflow.log_metric("n_anomalies", n_anomalies)
            mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
            mlflow.log_metric("n_clusters", n_clusters)
            
            # Provider-wise analysis
            self.df['anomaly_dbscan'] = (cluster_labels == -1).astype(int) * -1
            provider_anomalies = self.df.groupby('provider')['anomaly_dbscan'].apply(lambda x: (x == -1).sum()).to_dict()
            for provider, count in provider_anomalies.items():
                mlflow.log_metric(f"anomalies_{provider.lower()}", count)
            
            # Cluster size analysis
            cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
            for cluster_id, size in cluster_sizes.items():
                if cluster_id != -1:  # Don't log noise as cluster
                    mlflow.log_metric(f"cluster_{cluster_id}_size", size)
            
            # Save model and scaler
            model_path = "models/dbscan_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path, "models")
            
            scaler_path = "models/dbscan_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, "preprocessors")
            
            self.results['dbscan'] = {
                'anomaly_rate': anomaly_rate,
                'n_anomalies': n_anomalies,
                'n_clusters': n_clusters,
                'provider_anomalies': provider_anomalies
            }
            
            print(f"✅ DBSCAN Results:")
            print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
            print(f"   Number of clusters: {n_clusters}")
            print(f"   Provider anomalies: {provider_anomalies}")
    
    def run_statistical_anomaly_experiment(self):
        """Statistical approach for network anomaly detection."""
        mlflow.set_experiment("anomaly_detection")
        
        with mlflow.start_run(run_name=f"statistical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "statistical")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("method", "zscore_iqr")
            
            # Parameters
            z_threshold = 3.0
            iqr_multiplier = 1.5
            
            mlflow.log_param("z_threshold", z_threshold)
            mlflow.log_param("iqr_multiplier", iqr_multiplier)
            
            # Statistical anomaly detection on key metrics
            anomaly_flags = []
            metrics = ['target_bwd', 'duration', 'pkt_size']
            
            for metric in metrics:
                if metric in self.df.columns:
                    # Z-score method
                    z_scores = np.abs((self.df[metric] - self.df[metric].mean()) / self.df[metric].std())
                    z_anomalies = z_scores > z_threshold
                    
                    # IQR method
                    Q1 = self.df[metric].quantile(0.25)
                    Q3 = self.df[metric].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    iqr_anomalies = (self.df[metric] < lower_bound) | (self.df[metric] > upper_bound)
                    
                    # Combine methods (anomaly if detected by either method)
                    combined_anomalies = z_anomalies | iqr_anomalies
                    anomaly_flags.append(combined_anomalies)
                    
                    # Log metric-specific results
                    mlflow.log_metric(f"{metric}_z_anomalies", z_anomalies.sum())
                    mlflow.log_metric(f"{metric}_iqr_anomalies", iqr_anomalies.sum())
                    mlflow.log_metric(f"{metric}_combined_anomalies", combined_anomalies.sum())
            
            # Overall anomalies (any metric flagged)
            if anomaly_flags:
                overall_anomalies = pd.concat(anomaly_flags, axis=1).any(axis=1)
                n_anomalies = overall_anomalies.sum()
                anomaly_rate = n_anomalies / len(overall_anomalies) * 100
                
                mlflow.log_metric("n_anomalies", n_anomalies)
                mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
                
                # Provider-wise analysis
                self.df['anomaly_statistical'] = overall_anomalies.astype(int) * -1
                provider_anomalies = self.df.groupby('provider')['anomaly_statistical'].apply(lambda x: (x == -1).sum()).to_dict()
                for provider, count in provider_anomalies.items():
                    mlflow.log_metric(f"anomalies_{provider.lower()}", count)
                
                self.results['statistical'] = {
                    'anomaly_rate': anomaly_rate,
                    'n_anomalies': n_anomalies,
                    'provider_anomalies': provider_anomalies
                }
                
                print(f"✅ Statistical Anomaly Detection Results:")
                print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
                print(f"   Provider anomalies: {provider_anomalies}")
    
    def run_ensemble_anomaly_experiment(self):
        """Ensemble approach combining multiple anomaly detection methods."""
        if len(self.models) < 2:
            print("❌ Need at least 2 trained models for ensemble approach")
            return
        
        mlflow.set_experiment("anomaly_detection")
        
        with mlflow.start_run(run_name=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Tags
            mlflow.set_tag("model_type", "ensemble")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("method", "voting")
            
            # Collect anomaly predictions from different models
            anomaly_columns = [col for col in self.df.columns if col.startswith('anomaly_')]
            mlflow.log_param("ensemble_methods", anomaly_columns)
            
            if len(anomaly_columns) >= 2:
                # Voting-based ensemble (majority vote)
                voting_threshold = len(anomaly_columns) // 2 + 1
                
                anomaly_votes = self.df[anomaly_columns].apply(lambda x: (x == -1).sum(), axis=1)
                ensemble_anomalies = (anomaly_votes >= voting_threshold).astype(int) * -1
                
                n_anomalies = (ensemble_anomalies == -1).sum()
                anomaly_rate = n_anomalies / len(ensemble_anomalies) * 100
                
                mlflow.log_param("voting_threshold", voting_threshold)
                mlflow.log_metric("n_anomalies", n_anomalies)
                mlflow.log_metric("anomaly_rate_percent", anomaly_rate)
                
                # Provider-wise analysis
                self.df['anomaly_ensemble'] = ensemble_anomalies
                provider_anomalies = self.df.groupby('provider')['anomaly_ensemble'].apply(lambda x: (x == -1).sum()).to_dict()
                for provider, count in provider_anomalies.items():
                    mlflow.log_metric(f"anomalies_{provider.lower()}", count)
                
                # Agreement analysis
                for i, method in enumerate(anomaly_columns):
                    agreement = (self.df[method] == ensemble_anomalies).mean() * 100
                    mlflow.log_metric(f"agreement_{method}", agreement)
                
                self.results['ensemble'] = {
                    'anomaly_rate': anomaly_rate,
                    'n_anomalies': n_anomalies,
                    'provider_anomalies': provider_anomalies,
                    'voting_threshold': voting_threshold
                }
                
                print(f"✅ Ensemble Anomaly Detection Results:")
                print(f"   Total anomalies: {n_anomalies:,} ({anomaly_rate:.2f}%)")
                print(f"   Voting threshold: {voting_threshold}/{len(anomaly_columns)}")
                print(f"   Provider anomalies: {provider_anomalies}")
    
    def run_all_experiments(self):
        """Run all anomaly detection experiments."""
        print("🚀 Starting Anomaly Detection Experiments")
        print("=" * 50)
        
        # Run all models
        self.run_isolation_forest_experiment()
        self.run_one_class_svm_experiment()
        self.run_local_outlier_factor_experiment()
        self.run_dbscan_experiment()
        self.run_statistical_anomaly_experiment()
        self.run_ensemble_anomaly_experiment()
        
        # Summary
        print("\n📊 Experiment Summary:")
        print("-" * 50)
        for model_name, results in self.results.items():
            anomaly_rate = results.get('anomaly_rate', 'N/A')
            n_anomalies = results.get('n_anomalies', 'N/A')
            print(f"{model_name:18} | Rate: {anomaly_rate:>6}% | Count: {n_anomalies:>6}")
        
        # Cross-method comparison
        if len(self.results) > 1:
            print("\n🔍 Cross-Method Analysis:")
            rates = [r['anomaly_rate'] for r in self.results.values() if 'anomaly_rate' in r]
            if rates:
                print(f"   Average anomaly rate: {np.mean(rates):.2f}%")
                print(f"   Std dev anomaly rate: {np.std(rates):.2f}%")
                print(f"   Rate range: {min(rates):.2f}% - {max(rates):.2f}%")
        
        print(f"\n🌐 View results at: http://127.0.0.1:5000")
        
        return self.results

def main():
    """Main execution function."""
    print("🎯 Anomaly Detection MLOps Pipeline")
    print("Following ZoomCamp methodology for network anomaly detection")
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
    experiment = AnomalyDetectionExperiment()
    results = experiment.run_all_experiments()
    
    print("\n✅ Anomaly detection experiments completed!")
    return results

if __name__ == "__main__":
    main()
