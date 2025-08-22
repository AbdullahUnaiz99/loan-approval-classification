# ===================================
# FILE: src/data_processing.py
# ===================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from utils.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, CATEGORY_MAPPINGS

@st.cache_data
def load_sample_data():
    """Load and generate sample data"""
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic data
    data = {
        'person_age': np.random.normal(27, 4, n_samples).clip(18, 65),
        'person_gender': np.random.choice(['male', 'female'], n_samples),
        'person_education': np.random.choice(['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate'], n_samples),
        'person_income': np.random.lognormal(11, 0.5, n_samples).clip(8000, 200000),
        'person_home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n_samples, p=[0.5, 0.35, 0.13, 0.02]),
        'loan_amnt': np.random.lognormal(8.5, 0.8, n_samples).clip(500, 50000),
        'loan_intent': np.random.choice(['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], n_samples),
        'loan_int_rate': np.random.normal(11, 3, n_samples).clip(5, 20),
        'loan_percent_income': np.random.beta(2, 5, n_samples) * 0.5,
        'cb_person_cred_hist_length': np.random.poisson(5, n_samples).clip(1, 20),
        'credit_score': np.random.normal(630, 50, n_samples).clip(300, 850),
        'previous_loan_defaults_on_file': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable
    loan_status_prob = (
        0.1 +
        (df['credit_score'] - 300) / (850 - 300) * 0.4 +
        (1 - df['loan_percent_income']) * 0.3 +
        (df['previous_loan_defaults_on_file'] == 'No').astype(int) * 0.3 -
        (df['loan_int_rate'] - 5) / (20 - 5) * 0.2
    ).clip(0.05, 0.95)
    
    df['loan_status'] = np.random.binomial(1, loan_status_prob)
    
    return df

def cap_outliers_iqr(df, column):
    """Cap outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[column].clip(lower_bound, upper_bound)

def preprocess_data(df):
    """Preprocess the loan data"""
    df = df.copy()
    
    # Handle outliers
    numerical_cols = [col for col in NUMERICAL_FEATURES if col in df.columns]
    for col in numerical_cols:
        df[col] = cap_outliers_iqr(df, col)
    
    # Encode categorical variables
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
    return df, encoders

def prepare_input_data(input_dict, encoders):
    """Prepare input data for prediction"""
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables
    for col in CATEGORICAL_FEATURES:
        if col in encoders and col in input_df.columns:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding {col}: {e}")
                return None
    
    return input_df