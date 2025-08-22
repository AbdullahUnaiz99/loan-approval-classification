# ===================================
# FILE: app.py (Main Entry Point)
# ===================================

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'pages'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.constants import PAGE_CONFIG
from utils.helpers import load_css
import home
import data_analysis
import model_training
import prediction

def main():
    # Page configuration
    st.set_page_config(**PAGE_CONFIG)
    
    # Load custom CSS
    load_css()
    
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Loan Prediction"],
        key="main_navigation"
    )
    
    # Route to appropriate page
    if page == "🏠 Home":
        home.show()
    elif page == "📊 Data Analysis":
        data_analysis.show()
    elif page == "🤖 Model Training":
        model_training.show()
    elif page == "🔮 Loan Prediction":
        prediction.show()

if __name__ == "__main__":
    main()
