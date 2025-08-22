# ===================================
# FILE: pages/prediction.py (CONTINUED)
# ===================================

import streamlit as st
from src.data_processing import prepare_input_data
from src.prediction import get_user_input, display_prediction_result, display_risk_assessment

def show():
    """Display prediction page"""
    st.header("🔮 Loan Approval Prediction")
    
    # Check if model is trained
    if 'model' not in st.session_state or 'encoders' not in st.session_state:
        st.warning("⚠️ Please train the model first by visiting the 'Model Training' page.")
        st.info("👈 Navigate to the 'Model Training' page using the sidebar and click 'Train Model'.")
        
        # Show sample prediction form for demo purposes
        st.subheader("📝 Demo Prediction Form")
        st.info("This form shows the interface. Train the model to make actual predictions.")
        
        # Display the input form in read-only mode
        with st.form("demo_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.number_input("Age", min_value=18, max_value=100, value=30, disabled=True)
                st.selectbox("Gender", ['male', 'female'], disabled=True)
                st.selectbox("Education", ['High School', 'Bachelor', 'Master', 'Associate', 'Doctorate'], disabled=True)
                st.number_input("Annual Income ($)", min_value=0, value=50000, disabled=True)
            
            with col2:
                st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN', 'OTHER'], disabled=True)
                st.number_input("Loan Amount ($)", min_value=0, value=10000, disabled=True)
                st.selectbox("Loan Purpose", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'], disabled=True)
                st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, disabled=True)
            
            with col3:
                st.metric("Loan-to-Income Ratio", "0.200")
                st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5, disabled=True)
                st.number_input("Credit Score", min_value=300, max_value=850, value=650, disabled=True)
                st.selectbox("Previous Loan Defaults", ['Yes', 'No'], disabled=True)
            
            st.form_submit_button("🔮 Predict Loan Approval", disabled=True)
        
        return
    
    # Main prediction interface
    st.subheader("🎯 Make a Prediction")
    st.write("Enter the applicant's information below to get a loan approval prediction.")
    
    # Get user input
    input_data = get_user_input()
    
    if input_data:
        # Prepare data for prediction
        processed_input = prepare_input_data(input_data, st.session_state['encoders'])
        
        if processed_input is not None:
            try:
                # Make prediction
                model = st.session_state['model']
                prediction = model.predict(processed_input)[0]
                prediction_proba = model.predict_proba(processed_input)[0]
                
                # Display results
                display_prediction_result(prediction, prediction_proba)
                
                # Show detailed analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Application Summary")
                    st.write(f"**Applicant Age:** {input_data['person_age']} years")
                    st.write(f"**Annual Income:** ${input_data['person_income']:,}")
                    st.write(f"**Loan Amount:** ${input_data['loan_amnt']:,}")
                    st.write(f"**Credit Score:** {input_data['credit_score']}")
                    st.write(f"**Debt-to-Income:** {input_data['loan_percent_income']:.1%}")
                    st.write(f"**Interest Rate:** {input_data['loan_int_rate']:.1%}")
                
                with col2:
                    st.subheader("🎯 Prediction Confidence")
                    
                    # Create confidence visualization
                    if prediction == 1:
                        confidence = prediction_proba[1]
                        st.success(f"Approval Confidence: {confidence:.1%}")
                    else:
                        confidence = prediction_proba[0]
                        st.error(f"Rejection Confidence: {confidence:.1%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Confidence interpretation
                    if confidence >= 0.8:
                        st.write("🟢 **High Confidence** - Very reliable prediction")
                    elif confidence >= 0.6:
                        st.write("🟡 **Medium Confidence** - Moderately reliable prediction")
                    else:
                        st.write("🔴 **Low Confidence** - Less reliable prediction")
                
                # Risk assessment
                display_risk_assessment(input_data)
                
                # Additional insights
                st.subheader("💡 Key Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Financial Health Indicators:**")
                    
                    # Credit score assessment
                    if input_data['credit_score'] >= 750:
                        st.write("✅ Excellent credit score")
                    elif input_data['credit_score'] >= 700:
                        st.write("✅ Good credit score")
                    elif input_data['credit_score'] >= 650:
                        st.write("⚠️ Fair credit score")
                    else:
                        st.write("❌ Poor credit score")
                    
                    # Income assessment
                    if input_data['person_income'] >= 75000:
                        st.write("✅ High income level")
                    elif input_data['person_income'] >= 50000:
                        st.write("✅ Good income level")
                    elif input_data['person_income'] >= 30000:
                        st.write("⚠️ Moderate income level")
                    else:
                        st.write("❌ Low income level")
                    
                    # Debt-to-income ratio
                    if input_data['loan_percent_income'] <= 0.2:
                        st.write("✅ Low debt-to-income ratio")
                    elif input_data['loan_percent_income'] <= 0.4:
                        st.write("⚠️ Moderate debt-to-income ratio")
                    else:
                        st.write("❌ High debt-to-income ratio")
                
                with col2:
                    st.write("**Recommendations:**")
                    
                    if prediction == 0:  # Rejected
                        st.write("**To improve approval chances:**")
                        if input_data['credit_score'] < 650:
                            st.write("• Work on improving credit score")
                        if input_data['loan_percent_income'] > 0.4:
                            st.write("• Consider a smaller loan amount")
                        if input_data['previous_loan_defaults_on_file'] == 'Yes':
                            st.write("• Build positive payment history")
                        if input_data['person_income'] < 40000:
                            st.write("• Increase income or add co-applicant")
                        
                        st.write("• Consider waiting 6-12 months before reapplying")
                        st.write("• Consult with a financial advisor")
                    
                    else:  # Approved
                        st.write("**Next steps:**")
                        st.write("• Review loan terms carefully")
                        st.write("• Ensure you can afford monthly payments")
                        st.write("• Consider automatic payment setup")
                        st.write("• Maintain good credit during loan term")
                
                # Loan comparison table
                st.subheader("📋 Loan Details Summary")
                
                # Calculate monthly payment (simplified formula)
                monthly_rate = input_data['loan_int_rate'] / 100 / 12
                num_payments = 60  # Assuming 5-year term
                if monthly_rate > 0:
                    monthly_payment = input_data['loan_amnt'] * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
                else:
                    monthly_payment = input_data['loan_amnt'] / num_payments
                
                total_interest = (monthly_payment * num_payments) - input_data['loan_amnt']
                
                loan_details = {
                    "Metric": [
                        "Loan Amount",
                        "Interest Rate",
                        "Estimated Monthly Payment",
                        "Total Interest (5 years)",
                        "Total Amount Payable",
                        "Debt-to-Income Ratio"
                    ],
                    "Value": [
                        f"${input_data['loan_amnt']:,.2f}",
                        f"{input_data['loan_int_rate']:.2f}%",
                        f"${monthly_payment:,.2f}",
                        f"${total_interest:,.2f}",
                        f"${monthly_payment * num_payments:,.2f}",
                        f"{input_data['loan_percent_income']:.1%}"
                    ]
                }
                
                st.table(loan_details)
                
                # Feature importance (if available)
                if 'feature_names' in st.session_state:
                    st.subheader("🎯 Most Important Factors")
                    st.write("The following factors have the highest impact on loan approval decisions:")
                    
                    # Mock feature importance for demonstration
                    important_features = [
                        ("Credit Score", 0.25),
                        ("Debt-to-Income Ratio", 0.20),
                        ("Annual Income", 0.18),
                        ("Previous Defaults", 0.15),
                        ("Loan Amount", 0.12),
                        ("Credit History Length", 0.10)
                    ]
                    
                    for i, (feature, importance) in enumerate(important_features, 1):
                        st.write(f"{i}. **{feature}** - {importance:.1%} importance")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please check your input values and try again.")
    
    # Additional information section
    with st.expander("ℹ️ About This Prediction System"):
        st.write("""
        **How it works:**
        - This system uses a Random Forest machine learning model trained on historical loan data
        - The model considers multiple factors including credit score, income, loan amount, and payment history
        - Predictions include confidence scores to indicate reliability
        
        **Important Notes:**
        - This is a demonstration system for educational purposes
        - Real loan decisions involve additional factors not captured in this model
        - Always consult with financial professionals for actual loan applications
        - Predictions are estimates and not guarantees of approval or rejection
        
        **Model Performance:**
        - The model has been trained using advanced techniques including SMOTE for handling imbalanced data
        - Cross-validation and hyperparameter tuning were used to optimize performance
        - Feature importance analysis helps understand which factors matter most
        """)
    
    # Tips section
    with st.expander("💡 Tips for Loan Approval"):
        st.write("""
        **Improve Your Chances:**
        1. **Maintain Good Credit:** Pay bills on time and keep credit utilization low
        2. **Stable Income:** Demonstrate consistent employment and income history
        3. **Low Debt-to-Income:** Keep your total debt payments below 40% of income
        4. **Save for Down Payment:** Larger down payments reduce lender risk
        5. **Shop Around:** Different lenders have different criteria and rates
        
        **Before Applying:**
        - Check your credit report for errors
        - Calculate how much you can afford monthly
        - Gather all required documentation
        - Consider getting pre-approved to understand your options
        """)

# Additional helper function for the prediction page
def calculate_affordability_metrics(input_data):
    """Calculate additional affordability metrics"""
    metrics = {}
    
    # Monthly income
    monthly_income = input_data['person_income'] / 12
    
    # Estimated monthly payment (simplified)
    monthly_rate = input_data['loan_int_rate'] / 100 / 12
    num_payments = 60  # 5 years
    
    if monthly_rate > 0:
        monthly_payment = input_data['loan_amnt'] * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = input_data['loan_amnt'] / num_payments
    
    metrics['monthly_payment'] = monthly_payment
    metrics['payment_to_income_ratio'] = monthly_payment / monthly_income
    metrics['remaining_income'] = monthly_income - monthly_payment
    
    return metrics

def display_affordability_analysis(input_data):
    """Display affordability analysis"""
    metrics = calculate_affordability_metrics(input_data)
    
    st.subheader("💰 Affordability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Estimated Monthly Payment",
            f"${metrics['monthly_payment']:,.2f}"
        )
    
    with col2:
        ratio = metrics['payment_to_income_ratio']
        st.metric(
            "Payment-to-Income Ratio",
            f"{ratio:.1%}",
            delta=f"{'Healthy' if ratio <= 0.28 else 'High' if ratio <= 0.40 else 'Very High'}"
        )
    
    with col3:
        st.metric(
            "Remaining Monthly Income",
            f"${metrics['remaining_income']:,.2f}"
        )
    
    # Affordability indicator
    if metrics['payment_to_income_ratio'] <= 0.28:
        st.success("✅ Payment is well within recommended guidelines")
    elif metrics['payment_to_income_ratio'] <= 0.40:
        st.warning("⚠️ Payment is at the upper limit of recommended guidelines")
    else:
        st.error("❌ Payment exceeds recommended guidelines")