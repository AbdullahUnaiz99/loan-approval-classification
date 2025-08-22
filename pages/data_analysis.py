# ===================================
# FILE: pages/data_analysis.py
# ===================================

import streamlit as st
from src.data_processing import load_sample_data
from src.visualization import (
    plot_target_distribution, plot_numerical_distribution,
    plot_categorical_distribution, plot_correlation_matrix
)
from utils.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def show():
    """Display data analysis page"""
    st.header("📊 Exploratory Data Analysis")
    
    # Load data
    df = load_sample_data()
    
    # Target distribution
    st.subheader("🎯 Target Variable Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_target_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        target_counts = df['loan_status'].value_counts()
        st.write("**Class Distribution:**")
        st.write(f"- Approved: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
        st.write(f"- Rejected: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
        st.write(f"- Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1")
    
    # Numerical features
    st.subheader("📈 Numerical Features Distribution")
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    selected_num_col = st.selectbox("Select a numerical feature:", numerical_cols)
    
    fig = plot_numerical_distribution(df, selected_num_col)
    st.plotly_chart(fig, use_container_width=True)
    
    # Categorical features
    st.subheader("📊 Categorical Features Analysis")
    categorical_cols = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    selected_cat_col = st.selectbox("Select a categorical feature:", categorical_cols)
    
    fig = plot_categorical_distribution(df, selected_cat_col)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔗 Correlation Analysis")
    fig = plot_correlation_matrix(df)
    st.plotly_chart(fig, use_container_width=True)