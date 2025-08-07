#!/usr/bin/env python3
"""
Launcher script for the ECDA Preschool Demand Forecasting Dashboard
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'pandas',
        'numpy',
        'matplotlib',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n�� Installing missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main function to launch the dashboard"""
    print("=" * 60)
    print("🏫 ECDA Preschool Demand Forecasting Dashboard")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ Please run this script from the childcare/code directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("🚀 Launching dashboard...")
    print("📊 Dashboard will be available at: http://localhost:7860")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and launch dashboard
        from dashboard import launch_dashboard
        launch_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()