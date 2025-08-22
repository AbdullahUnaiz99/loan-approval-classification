# ===================================
# FILE: utils/helpers.py
# ===================================

import streamlit as st
import os

def load_css():
    """Load custom CSS styles"""
    css = """
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 5px solid #1f77b4;
        }
        .prediction-card {
            padding: 2rem;
            border-radius: 1rem;
            text-align: center;
            margin: 1rem 0;
        }
        .approved {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }
        .rejected {
            background: linear-gradient(135deg, #f44336, #da190b);
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def format_currency(amount):
    """Format currency with commas"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format percentage"""
    return f"{value:.1%}"