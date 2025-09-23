"""
Fixed EDA for Network Monitoring Data - Custom for Your Exact Column Names
Perfect for your 51 columns including provider, vm_size, protocol, regions, timestamps, and network tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
from datetime import datetime
import logging

class NetworkEDA:
    """Comprehensive EDA class for network monitoring data - Custom for your exact columns."""
    
    def __init__(self, data_path=None):
        # Setup paths
        self.data_path = data_path or "data/processed/"
        self.output_path = Path("reports/eda/")
        self.viz_path = Path("visualizations/")
        
        # Create directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load data
        self.df = None
        self.load_data()
        
        # Set up your specific columns
        self.setup_columns()
        
    def load_data(self):
        """Load the processed network data."""
        data_files = list(Path(self.data_path).glob("cloud_network_performance_*.csv"))
        if not data_files:
            data_files = list(Path(self.data_path).glob("*.csv"))
        
        if data_files:
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            self.df = pd.read_csv(latest_file)
            self.logger.info(f"Loaded data: {latest_file} - {len(self.df)} records")
        else:
            raise FileNotFoundError("No CSV files found in data directory")
    
    def setup_columns(self):
        """Set up column mappings for your specific dataset."""
        # Your key network analysis columns
        self.key_columns = {
            'provider': 'provider',
            'vm_size': 'vm_size', 
            'protocol': 'protocol',
            'source_region': 'source_region',
            'dest_region': 'dest_region',
            'timestamp': 'timestamp',
            'experiment_datetime': 'experiment_datetime',
            'duration': 'duration',
            'target_bandwidth': 'target_bwd',
            'packet_size': 'pkt_size',
            'port': 'port',
            'tool': 'tool',
            'granularity': 'granularity'
        }
        
        # Get numeric and categorical columns
        self.numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.df.select_dtypes(include=['object']).columns)
        
        self.logger.info(f"Key network columns identified: {list(self.key_columns.keys())}")
        self.logger.info(f"Numeric columns ({len(self.numeric_columns)}): {self.numeric_columns[:10]}...")
        self.logger.info(f"Categorical columns ({len(self.categorical_columns)}): {self.categorical_columns[:10]}...")
    
    def generate_data_profile(self):
        """Generate comprehensive data profiling report."""
        profile = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_records': len(self.df),
                'total_features': len(self.df.columns),
                'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
                'numeric_features': len(self.numeric_columns),
                'categorical_features': len(self.categorical_columns)
            },
            'data_quality': {
                'missing_values': {col: int(self.df[col].isnull().sum()) for col in self.df.columns if self.df[col].isnull().sum() > 0},
                'duplicate_records': int(self.df.duplicated().sum()),
            },
            'network_analysis': {}
        }
        
        # Provider analysis
        if 'provider' in self.df.columns:
            profile['network_analysis']['provider_distribution'] = self.df['provider'].value_counts().to_dict()
            profile['network_analysis']['unique_providers'] = int(self.df['provider'].nunique())
        
        # VM size analysis
        if 'vm_size' in self.df.columns:
            profile['network_analysis']['vm_size_distribution'] = self.df['vm_size'].value_counts().to_dict()
        
        # Protocol analysis
        if 'protocol' in self.df.columns:
            profile['network_analysis']['protocol_distribution'] = self.df['protocol'].value_counts().to_dict()
        
        # Region analysis
        if 'source_region' in self.df.columns:
            profile['network_analysis']['source_region_distribution'] = self.df['source_region'].value_counts().to_dict()
        
        if 'dest_region' in self.df.columns:
            profile['network_analysis']['dest_region_distribution'] = self.df['dest_region'].value_counts().to_dict()
        
        # Tool analysis
        if 'tool' in self.df.columns:
            profile['network_analysis']['tool_distribution'] = self.df['tool'].value_counts().to_dict()
        
        # Numeric summary for key metrics
        numeric_metrics = ['duration', 'target_bwd', 'pkt_size', 'port', 'granularity', 'timestamp']
        profile['network_analysis']['numeric_summary'] = {}
        
        for col in numeric_metrics:
            if col in self.df.columns and self.df[col].dtype in [np.int64, np.float64]:
                profile['network_analysis']['numeric_summary'][col] = {
                    'mean': float(self.df[col].mean()) if not self.df[col].isnull().all() else None,
                    'median': float(self.df[col].median()) if not self.df[col].isnull().all() else None,
                    'std': float(self.df[col].std()) if not self.df[col].isnull().all() else None,
                    'min': float(self.df[col].min()) if not self.df[col].isnull().all() else None,
                    'max': float(self.df[col].max()) if not self.df[col].isnull().all() else None,
                    'missing_pct': float(self.df[col].isnull().mean() * 100)
                }
        
        # Save profile
        profile_path = self.output_path / f"data_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        self.logger.info(f"Data profile saved: {profile_path}")
        return profile
    
    def provider_analysis(self):
        """Analyze performance differences between cloud providers."""
        if 'provider' not in self.df.columns:
            self.logger.warning("No provider column found")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cloud Provider Analysis - AWS vs Azure', fontsize=16, fontweight='bold')
        
        # Provider distribution
        provider_counts = self.df['provider'].value_counts()
        axes[0,0].pie(provider_counts.values, labels=provider_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title(f'Provider Distribution\n(Total: {len(self.df):,} records)')
        
        # VM size by provider
        if 'vm_size' in self.df.columns:
            provider_vm = pd.crosstab(self.df['provider'], self.df['vm_size'])
            provider_vm.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            axes[0,1].set_title('VM Size Distribution by Provider')
            axes[0,1].set_xlabel('Provider')
            axes[0,1].set_ylabel('Count')
            axes[0,1].legend(title='VM Size')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Protocol by provider
        if 'protocol' in self.df.columns:
            provider_protocol = pd.crosstab(self.df['provider'], self.df['protocol'])
            provider_protocol.plot(kind='bar', stacked=True, ax=axes[0,2])
            axes[0,2].set_title('Protocol Distribution by Provider')
            axes[0,2].set_xlabel('Provider')
            axes[0,2].legend(title='Protocol')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # Source region by provider
        if 'source_region' in self.df.columns:
            top_regions = self.df['source_region'].value_counts().head(10)
            region_provider = self.df[self.df['source_region'].isin(top_regions.index)]
            sns.countplot(data=region_provider, x='source_region', hue='provider', ax=axes[1,0])
            axes[1,0].set_title('Top 10 Source Regions by Provider')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Tool usage by provider
        if 'tool' in self.df.columns:
            tool_provider = pd.crosstab(self.df['provider'], self.df['tool'])
            tool_provider.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Network Tools by Provider')
            axes[1,1].set_xlabel('Provider')
            axes[1,1].legend(title='Tool')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # Duration analysis by provider
        if 'duration' in self.df.columns and self.df['duration'].dtype in [np.int64, np.float64]:
            sns.boxplot(data=self.df, x='provider', y='duration', ax=axes[1,2])
            axes[1,2].set_title('Experiment Duration by Provider')
            axes[1,2].set_ylabel('Duration')
        
        plt.tight_layout()
        save_path = self.viz_path / 'provider_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Provider analysis saved: {save_path}")
        return save_path
    
    def network_infrastructure_analysis(self):
        """Analyze network infrastructure patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Network Infrastructure Analysis', fontsize=16, fontweight='bold')
        
        # VM Size distribution
        if 'vm_size' in self.df.columns:
            vm_counts = self.df['vm_size'].value_counts()
            axes[0,0].pie(vm_counts.values, labels=vm_counts.index, autopct='%1.1f%%')
            axes[0,0].set_title('VM Size Distribution')
        
        # Protocol distribution
        if 'protocol' in self.df.columns:
            protocol_counts = self.df['protocol'].value_counts()
            protocol_counts.plot(kind='bar', ax=axes[0,1], color='lightblue')
            axes[0,1].set_title('Protocol Distribution')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Regional connectivity patterns
        if 'source_region' in self.df.columns and 'dest_region' in self.df.columns:
            # Create region pair analysis
            self.df['region_pair'] = self.df['source_region'] + ' → ' + self.df['dest_region']
            top_pairs = self.df['region_pair'].value_counts().head(15)
            top_pairs.plot(kind='barh', ax=axes[1,0])
            axes[1,0].set_title('Top 15 Region Connectivity Patterns')
            axes[1,0].set_xlabel('Count')
        
        # Tools used
        if 'tool' in self.df.columns:
            tool_counts = self.df['tool'].value_counts()
            tool_counts.plot(kind='bar', ax=axes[1,1], color='lightgreen')
            axes[1,1].set_title('Network Testing Tools Used')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.viz_path / 'infrastructure_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Infrastructure analysis saved: {save_path}")
        return save_path
    
    def time_series_analysis(self):
        """Analyze temporal patterns in network experiments."""
        if 'experiment_datetime' not in self.df.columns:
            self.logger.warning("No experiment_datetime column found")
            return None
        
        # Convert to datetime
        self.df['experiment_datetime'] = pd.to_datetime(self.df['experiment_datetime'], errors='coerce')
        
        # Remove rows with invalid dates
        valid_dates = self.df['experiment_datetime'].notna()
        if not valid_dates.any():
            self.logger.warning("No valid dates found in experiment_datetime")
            return None
        
        df_time = self.df[valid_dates].copy()
        
        # Create time-based features
        df_time['date'] = df_time['experiment_datetime'].dt.date
        df_time['hour'] = df_time['experiment_datetime'].dt.hour
        df_time['day_of_week'] = df_time['experiment_datetime'].dt.day_name()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Analysis of Network Experiments', fontsize=16, fontweight='bold')
        
        # Experiments over time
        daily_counts = df_time.groupby('date').size()
        axes[0,0].plot(daily_counts.index, daily_counts.values, marker='o')
        axes[0,0].set_title('Daily Experiment Count')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Number of Experiments')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Hourly patterns
        hourly_counts = df_time['hour'].value_counts().sort_index()
        axes[0,1].bar(hourly_counts.index, hourly_counts.values, color='lightcoral')
        axes[0,1].set_title('Experiments by Hour of Day')
        axes[0,1].set_xlabel('Hour')
        axes[0,1].set_ylabel('Count')
        
        # Day of week patterns
        dow_counts = df_time['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_ordered = dow_counts.reindex([day for day in dow_order if day in dow_counts.index])
        axes[1,0].bar(range(len(dow_ordered)), dow_ordered.values, color='lightgreen')
        axes[1,0].set_xticks(range(len(dow_ordered)))
        axes[1,0].set_xticklabels(dow_ordered.index, rotation=45)
        axes[1,0].set_title('Experiments by Day of Week')
        axes[1,0].set_ylabel('Count')
        
        # Provider activity over time
        if 'provider' in self.df.columns:
            provider_time = df_time.groupby(['date', 'provider']).size().unstack(fill_value=0)
            for provider in provider_time.columns:
                axes[1,1].plot(provider_time.index, provider_time[provider], marker='o', label=provider)
            axes[1,1].set_title('Provider Activity Over Time')
            axes[1,1].set_xlabel('Date')
            axes[1,1].set_ylabel('Number of Experiments')
            axes[1,1].legend()
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.viz_path / 'time_series_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Time series analysis saved: {save_path}")
        return save_path
    
    def correlation_analysis(self):
        """Analyze correlations between numeric metrics."""
        if len(self.numeric_columns) < 2:
            self.logger.warning("Not enough numeric columns for correlation analysis")
            return None
        
        # Select relevant numeric columns (remove IDs and indices)
        exclude_patterns = ['id', 'index', 'filepath']
        relevant_numeric = [col for col in self.numeric_columns 
                          if not any(pattern in col.lower() for pattern in exclude_patterns)]
        
        if len(relevant_numeric) < 2:
            self.logger.warning("Not enough relevant numeric columns for correlation")
            return None
        
        # Take top 15 columns to avoid overcrowding
        top_numeric = relevant_numeric[:15]
        corr_matrix = self.df[top_numeric].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix - Network Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        corr_path = self.viz_path / 'correlation_matrix.png'
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Correlation analysis saved: {corr_path}")
        return corr_path
    
    def target_bandwidth_analysis(self):
        """Analyze target bandwidth patterns."""
        if 'target_bwd' not in self.df.columns:
            self.logger.warning("No target_bwd column found")
            return None
        
        # Convert to numeric if needed
        self.df['target_bwd'] = pd.to_numeric(self.df['target_bwd'], errors='coerce')
        
        if self.df['target_bwd'].isnull().all():
            self.logger.warning("target_bwd column contains no valid numeric data")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Target Bandwidth Analysis', fontsize=16, fontweight='bold')
        
        # Distribution of target bandwidth
        axes[0,0].hist(self.df['target_bwd'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Target Bandwidth Distribution')
        axes[0,0].set_xlabel('Target Bandwidth')
        axes[0,0].set_ylabel('Frequency')
        
        # Target bandwidth by provider
        if 'provider' in self.df.columns:
            sns.boxplot(data=self.df, x='provider', y='target_bwd', ax=axes[0,1])
            axes[0,1].set_title('Target Bandwidth by Provider')
        
        # Target bandwidth by VM size
        if 'vm_size' in self.df.columns:
            sns.boxplot(data=self.df, x='vm_size', y='target_bwd', ax=axes[1,0])
            axes[1,0].set_title('Target Bandwidth by VM Size')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Target bandwidth by protocol
        if 'protocol' in self.df.columns:
            sns.boxplot(data=self.df, x='protocol', y='target_bwd', ax=axes[1,1])
            axes[1,1].set_title('Target Bandwidth by Protocol')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = self.viz_path / 'bandwidth_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Bandwidth analysis saved: {save_path}")
        return save_path
    
    def generate_executive_summary(self):
        """Generate executive summary with key insights."""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'dataset_overview': {
                'total_records': len(self.df),
                'total_features': len(self.df.columns),
                'numeric_features': len(self.numeric_columns),
                'categorical_features': len(self.categorical_columns),
                'data_quality_score': round((1 - (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)))) * 100, 2)
            },
            'key_insights': [],
            'recommendations': []
        }
        
        # Provider insights
        if 'provider' in self.df.columns:
            provider_counts = self.df['provider'].value_counts()
            aws_pct = (provider_counts.get('AWS', 0) / len(self.df)) * 100
            azure_pct = (provider_counts.get('Azure', 0) / len(self.df)) * 100
            summary['key_insights'].append(
                f"Dataset is AWS-heavy: {aws_pct:.1f}% AWS ({provider_counts.get('AWS', 0):,} records) vs "
                f"{azure_pct:.1f}% Azure ({provider_counts.get('Azure', 0):,} records)"
            )
        
        # Protocol insights
        if 'protocol' in self.df.columns:
            protocol_counts = self.df['protocol'].value_counts()
            top_protocol = protocol_counts.index[0] if len(protocol_counts) > 0 else 'Unknown'
            summary['key_insights'].append(
                f"Primary protocol: {top_protocol} ({protocol_counts.iloc[0]:,} records, "
                f"{(protocol_counts.iloc[0]/len(self.df)*100):.1f}%)"
            )
        
        # VM size insights
        if 'vm_size' in self.df.columns:
            vm_counts = self.df['vm_size'].value_counts()
            summary['key_insights'].append(
                f"VM distribution: {dict(vm_counts)}"
            )
        
        # Regional insights
        if 'source_region' in self.df.columns:
            region_counts = self.df['source_region'].value_counts()
            top_region = region_counts.index[0] if len(region_counts) > 0 else 'Unknown'
            summary['key_insights'].append(
                f"Most active source region: {top_region} ({region_counts.iloc[0]:,} experiments)"
            )
        
        # Tool insights
        if 'tool' in self.df.columns:
            tool_counts = self.df['tool'].value_counts()
            summary['key_insights'].append(f"Network tools used: {list(tool_counts.index)}")
        
        # Data quality insights
        missing_cols = [col for col in self.df.columns if self.df[col].isnull().sum() > 0]
        if missing_cols:
            summary['key_insights'].append(f"Columns with missing data: {len(missing_cols)} out of {len(self.df.columns)}")
        
        # Recommendations
        summary['recommendations'] = [
            "Consider balancing AWS/Azure data for comparative analysis",
            "Focus on the dominant protocol for initial modeling efforts",
            "Validate regional connectivity patterns for geographical insights",
            "Assess data quality and handle missing values appropriately",
            "Use temporal patterns to understand experiment scheduling",
            "Leverage target bandwidth variations for performance modeling"
        ]
        
        # Save summary
        summary_path = self.output_path / 'executive_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Executive summary saved: {summary_path}")
        return summary
    
    def run_full_eda(self):
        """Execute complete EDA pipeline."""
        self.logger.info("Starting comprehensive EDA pipeline...")
        
        results = {}
        
        try:
            # Generate data profile
            results['data_profile'] = self.generate_data_profile()
            
            # Run all analyses
            results['provider_analysis'] = self.provider_analysis()
            results['infrastructure_analysis'] = self.network_infrastructure_analysis()
            results['time_series_analysis'] = self.time_series_analysis()
            results['correlation_analysis'] = self.correlation_analysis()
            results['bandwidth_analysis'] = self.target_bandwidth_analysis()
            results['executive_summary'] = self.generate_executive_summary()
            
            self.logger.info("✅ EDA pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Error in EDA pipeline: {e}")
            results['error'] = str(e)
        
        return results

def main():
    """Main execution function for EDA."""
    try:
        eda = NetworkEDA()
        results = eda.run_full_eda()
        print("\n🎉 EDA COMPLETED SUCCESSFULLY!")
        print("📊 Generated comprehensive network monitoring analysis")
        print("📁 Check these directories for outputs:")
        print("   • reports/eda/ - Data profiles and summaries")
        print("   • visualizations/ - Charts and plots")
        print(f"\n📈 Analysis covered {len(eda.df)} records across {len(eda.df.columns)} features")
        return 0
    except Exception as e:
        print(f"❌ EDA failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())