import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add utils to path
sys.path.append('utils')

# Page configuration
st.set_page_config(
    page_title="Battery Failure Prediction Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .sidebar-info {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔋 Battery Failure Prediction Dashboard</h1>
    <p>Advanced ML-powered battery health monitoring and failure prediction system</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("""
<div class="sidebar-info">
    <h3>📊 Dashboard Navigation</h3>
    <p>Explore different aspects of battery failure prediction using ensemble ML models.</p>
</div>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "📊 Data Exploration", "🎯 Model Performance", "🔮 Prediction", "📈 Battery Monitoring"]
)

# Main content area
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔬 Project Overview")
        st.markdown("""
        This dashboard showcases an advanced battery failure prediction system using NASA battery dataset 
        and ensemble machine learning models. The system combines three powerful approaches:
        
        **🤖 Machine Learning Models:**
        - **XGBoost Classifier**: Gradient boosting for robust classification
        - **LSTM Neural Network**: Sequential pattern recognition for time-series data
        - **One-Class SVM**: Anomaly detection for failure outliers
        
        **📈 Key Features:**
        - Real-time battery health monitoring (SOC/SOH)
        - Interactive data exploration and visualization
        - Model performance comparison and analysis
        - Predictive analytics for failure forecasting
        - Feature importance analysis
        """)
        
        st.header("🔋 Battery Dataset Information")
        st.markdown("""
        **NASA Battery Dataset Features:**
        - **Cycle**: Charge/discharge cycle number
        - **Voltage**: Battery terminal voltage (V)
        - **Current**: Charge/discharge current (A)
        - **Temperature**: Operating temperature (°C)
        - **Capacity**: Available battery capacity (Ah)
        - **Time**: Cycle duration (seconds)
        - **Internal Resistance**: Calculated resistance (Ω)
        - **SOC**: State of Charge (%)
        - **SOH**: State of Health (%)
        """)
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Display some quick statistics
        try:
            from utils.data_loader import load_battery_data
            data = load_battery_data()
            if data is not None:
                st.metric("Total Records", f"{len(data):,}")
                st.metric("Battery Units", len(data['battery_id'].unique()) if 'battery_id' in data.columns else "N/A")
                st.metric("Features", len(data.columns) - 1 if 'failure' in data.columns else len(data.columns))
                
                failure_rate = (data['failure'].sum() / len(data) * 100) if 'failure' in data.columns else 0
                st.metric("Failure Rate", f"{failure_rate:.1f}%")
        except Exception as e:
            st.error("Unable to load dataset statistics")
            st.info("Please upload the NASA battery dataset using the data upload section below.")
        
        st.header("🚀 Getting Started")
        st.markdown("""
        1. **📊 Data Exploration**: Analyze battery degradation patterns
        2. **🎯 Model Performance**: Compare ensemble model results
        3. **🔮 Prediction**: Make predictions on new battery data
        4. **📈 Battery Monitoring**: Monitor real-time battery health
        """)
        
        st.header("ℹ️ About the Models")
        st.info("""
        **Ensemble Approach**: Combines predictions from multiple models:
        - 50% LSTM (temporal patterns)
        - 30% XGBoost (feature relationships)
        - 20% One-Class SVM (anomaly detection)
        """)

    # Data Upload Section
    st.header("📁 NASA Battery Dataset Upload")
    st.markdown("Upload the actual NASA battery dataset for full functionality.")

    with st.expander("📤 Upload Dataset", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose the NASA battery dataset CSV file",
            type="csv",
            help="Upload the nasa_battery_data_combined.csv file generated from the NASA battery .mat files"
        )
        
        if uploaded_file is not None:
            try:
                # Save the uploaded file
                with open("nasa_battery_data_combined.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success("Dataset uploaded successfully! Please refresh the page to load the new data.")
                
                # Show preview of uploaded data
                uploaded_data = pd.read_csv(uploaded_file)
                st.write("**Data Preview:**")
                st.dataframe(uploaded_data.head())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(uploaded_data):,}")
                with col2:
                    st.metric("Features", len(uploaded_data.columns))
                with col3:
                    if 'failure' in uploaded_data.columns:
                        failure_rate = (uploaded_data['failure'].sum() / len(uploaded_data) * 100)
                        st.metric("Failure Rate", f"{failure_rate:.1f}%")
                    
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
        
        st.markdown("""
        **Expected CSV format should include these columns:**
        - battery_id, cycle, voltage, current, temperature, capacity, time, internal_resistance, SOC, SOH, failure
        """)

    # Model Upload Section  
    st.header("🤖 Pre-trained Models Upload")
    st.markdown("Upload pre-trained models for enhanced prediction capabilities.")

    with st.expander("📤 Upload Models", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**XGBoost Model**")
            xgb_file = st.file_uploader("XGBoost model (.json)", type="json", key="xgb")
            if xgb_file:
                with open("xgboost_model_tuned.json", "wb") as f:
                    f.write(xgb_file.getbuffer())
                st.success("XGBoost model uploaded!")
        
        with col2:
            st.write("**LSTM Model**")
            lstm_file = st.file_uploader("LSTM model (.h5)", type="h5", key="lstm")
            if lstm_file:
                with open("lstm_model_tuned.h5", "wb") as f:
                    f.write(lstm_file.getbuffer())
                st.success("LSTM model uploaded!")
        
        with col3:
            st.write("**SVM Model**")
            svm_file = st.file_uploader("SVM model (.joblib)", type="joblib", key="svm")
            if svm_file:
                with open("one_class_svm_model_tuned.joblib", "wb") as f:
                    f.write(svm_file.getbuffer())
                st.success("SVM model uploaded!")

elif page == "📊 Data Exploration":
    from pages.data_exploration import show_data_exploration
    show_data_exploration()

elif page == "🎯 Model Performance":
    from pages.model_performance import show_model_performance
    show_model_performance()

elif page == "🔮 Prediction":
    from pages.prediction import show_prediction
    show_prediction()

elif page == "📈 Battery Monitoring":
    from pages.battery_monitoring import show_battery_monitoring
    show_battery_monitoring()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Battery Failure Prediction System</strong></p>
    <p>Powered by Ensemble ML Models</p>
</div>
""", unsafe_allow_html=True)

