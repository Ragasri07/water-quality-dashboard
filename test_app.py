#!/usr/bin/env python
"""Quick test script to verify the app can run."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

print("✅ Starting diagnostic test...")
print()

# Test 1: Check Python version
print("1. Python version:")
print(f"   {sys.version}")
print()

# Test 2: Check key imports
print("2. Checking key imports...")
try:
    import pandas as pd
    print("   ✅ pandas")
except:
    print("   ❌ pandas - INSTALL: pip install pandas")

try:
    import numpy as np
    print("   ✅ numpy")
except:
    print("   ❌ numpy - INSTALL: pip install numpy")

try:
    import streamlit as st
    print(f"   ✅ streamlit (v{st.__version__})")
except:
    print("   ❌ streamlit - INSTALL: pip install streamlit")

try:
    import plotly
    print("   ✅ plotly")
except:
    print("   ❌ plotly - INSTALL: pip install plotly")

try:
    from sklearn.ensemble import RandomForestRegressor
    print("   ✅ scikit-learn")
except:
    print("   ❌ scikit-learn - INSTALL: pip install scikit-learn")

print()

# Test 3: Check data files
print("3. Checking data files...")
water_file = os.path.join(os.path.dirname(__file__), 'data', 'water_quality.csv')
rain_file = os.path.join(os.path.dirname(__file__), 'data', 'rainfall.csv')

if os.path.exists(water_file):
    print(f"   ✅ water_quality.csv ({os.path.getsize(water_file) / 1024:.1f} KB)")
else:
    print(f"   ❌ water_quality.csv NOT FOUND")

if os.path.exists(rain_file):
    print(f"   ✅ rainfall.csv ({os.path.getsize(rain_file) / 1024:.1f} KB)")
else:
    print(f"   ❌ rainfall.csv NOT FOUND")

print()

# Test 4: Load data
print("4. Testing data loading...")
try:
    df = pd.read_csv(water_file)
    print(f"   ✅ Loaded water quality data: {len(df)} rows × {len(df.columns)} columns")
except Exception as e:
    print(f"   ❌ Failed to load water data: {e}")

print()
print("=" * 60)
print("✅ If all checks pass, the app should work!")
print("=" * 60)
print()
print("To run the app, paste this command in your terminal:")
print()
print("   cd /Users/raga/Desktop/water_quality_project && streamlit run app.py")
print()
