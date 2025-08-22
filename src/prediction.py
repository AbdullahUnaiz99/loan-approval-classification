# ===================================
# FILE: src/prediction.py
# ===================================

import streamlit as st
from utils.constants import CATEGORY_MAPPINGS

def get_user_input():
    """Get user input for loan prediction"""
    st.subheader("📝 Enter Applicant Information")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
            person_gender = st.selectbox("Gender", CATEGORY_MAPPINGS['person_gender'])
            person_education = st.selectbox("Education", CATEGORY_MAPPINGS['person_education'])
            person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        
        with col2:
            person_home_ownership = st.selectbox("Home Ownership", CATEGORY_MAPPINGS['person_home_ownership'])
            loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000)
            loan_intent = st.selectbox("Loan Purpose", CATEGORY_MAPPINGS['loan_intent'])
            loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0)
        
        with col3:
            loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
            st.metric("Loan-to-Income Ratio", f"{loan_percent_income:.3f}")
            
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            previous_loan_defaults = st.selectbox("Previous Loan Defaults", CATEGORY_MAPPINGS['previous_loan_defaults_on_file'])
        
        submitted = st.form_submit_button("🔮 Predict Loan Approval", type="primary")
    
    if submitted:
        return {
            'person_age': person_age,
            'person_gender': person_gender,
            'person_education': person_education,
            'person_income': person_income,
            'person_home_ownership': person_home_ownership,
            'loan_amnt': loan_amnt,
            'loan_intent': loan_intent,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': previous_loan_defaults
        }
    
    return None

def display_prediction_result(prediction, prediction_proba):
    """Display prediction results"""
    st.subheader("📋 Prediction Results")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="prediction-card approved">
            <h2>✅ LOAN APPROVED</h2>
            <h3>Confidence: {prediction_proba[1]:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card rejected">
            <h2>❌ LOAN REJECTED</h2>
            <h3>Confidence: {prediction_proba[0]:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

def display_risk_assessment(input_data):
    """Display risk assessment"""
    st.subheader("📊 Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Positive Factors:**")
        factors = []
        if input_data['credit_score'] >= 700:
            factors.append("• Excellent credit score")
        if input_data['loan_percent_income'] <= 0.2:
            factors.append("• Low debt-to-income ratio")
        if input_data['previous_loan_defaults_on_file'] == 'No':
            factors.append("• No previous defaults")
        if input_data['person_income'] >= 60000:
            factors.append("• High income")
        
        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.write("• None identified")
    
    with col2:
        st.write("**Risk Factors:**")
        risk_factors = []
        if input_data['credit_score'] < 600:
            risk_factors.append("• Low credit score")
        if input_data['loan_percent_income'] > 0.4:
            risk_factors.append("• High debt-to-income ratio")
        if input_data['previous_loan_defaults_on_file'] == 'Yes':
            risk_factors.append("• Previous loan defaults")
        if input_data['loan_int_rate'] > 15:
            risk_factors.append("• High interest rate")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("• None identified")

# ===================================
# FILE: pages/home.py
# ===================================

import streamlit as st
from src.data_processing import load_sample_data

def show():
    """Display home page"""
    st.markdown('<h1 class="main-header">💰 Loan Approval Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("## Welcome to the Loan Approval Prediction System")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Data Analysis</h3>
            <p>Explore comprehensive insights from loan data including distributions, correlations, and patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model Training</h3>
            <p>Train and evaluate machine learning models with advanced techniques like SMOTE and hyperparameter tuning.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Prediction</h3>
            <p>Make real-time loan approval predictions with confidence scores and detailed explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset overview
    df = load_sample_data()
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        approval_rate = df['loan_status'].mean()
        st.metric("Approval Rate", f"{approval_rate:.1%}")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Sample data
    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)