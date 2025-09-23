#!/usr/bin/env python3
"""
MLflow Session Starter
Run this to start MLflow tracking server and setup experiments
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml.mlflow_manager import start_mlflow_session
import webbrowser
import time

def main():
    """Start MLflow session and open dashboard."""
    print("🚀 Starting MLflow Session for Network Monitoring MLOps...")
    
    if start_mlflow_session():
        # Wait a moment for server to fully start
        time.sleep(2)
        
        # Open dashboard in browser
        try:
            webbrowser.open("http://127.0.0.1:5000")
            print("🌐 Opening MLflow dashboard in browser...")
        except:
            print("💡 Manual: Open http://127.0.0.1:5000 in your browser")
        
        print("\n✅ MLflow session started successfully!")
        print("📊 Dashboard: http://127.0.0.1:5000")
        print("🧪 Ready for ML experiments!")
        
        return True
    else:
        print("❌ Failed to start MLflow session")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*50)
        print("MLflow is now running in the background")
        print("Use 'python scripts/stop_mlflow.py' to stop")
        print("="*50)
    else:
        sys.exit(1)
