cd /Users/raga/Desktop/water_quality_project && python test_app.py#!/usr/bin/env python
"""Minimal Streamlit app to test if basic setup works."""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="WaterWatch Test", page_icon="💧", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
    <h1 style="color: white; margin: 0; font-size: 2.5em;">🌊 WaterWatch — Test Version</h1>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin: 10px 0 0 0;">Quick Test to Verify Setup</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### ✅ If you see this message, Streamlit is working!")

st.info("""
**Great news!** Your Streamlit installation is working correctly.

If you're seeing this, it means:
- ✅ Python environment is set up
- ✅ Streamlit is installed
- ✅ The app can launch

Now let's test data loading...
""")

# Test data loading
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
WATER_FILE = os.path.join(DATA_DIR, "water_quality.csv")

if os.path.exists(WATER_FILE):
    try:
        df = pd.read_csv(WATER_FILE)
        st.success(f"✅ **Data Loaded Successfully!**")
        st.write(f"- **Records**: {len(df):,}")
        st.write(f"- **Columns**: {len(df.columns)}")
        st.write(f"- **Date Range**: {df.columns[0]}")
        
        st.markdown("---")
        st.markdown("### 📊 First 5 Rows of Data")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
else:
    st.error(f"❌ Data file not found: {WATER_FILE}")

st.markdown("---")
st.markdown("""
### 🎯 Next Steps

1. **Close this app** (press Ctrl+C in terminal)
2. **Run the full app**:
   ```bash
   cd /Users/raga/Desktop/water_quality_project
   streamlit run app.py
   ```
3. **Check terminal output** for any error messages

If you still see a blank screen:
- Check the terminal for error messages
- Post the exact error message here
""")
