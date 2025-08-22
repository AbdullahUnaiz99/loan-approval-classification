# ===================================
# FILE: utils/constants.py
# ===================================

# Configuration constants
PAGE_CONFIG = {
    "page_title": "Loan Approval Prediction System",
    "page_icon": "💰",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model parameters
MODEL_PARAMS = {
    'random_state': 42,
    'test_size': 0.2,
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}

# Feature lists
CATEGORICAL_FEATURES = [
    'person_gender', 'person_education', 'person_home_ownership',
    'loan_intent', 'previous_loan_defaults_on_file'
]

NUMERICAL_FEATURES = [
    'person_age', 'person_income', 'loan_amnt', 'loan_int_rate',
    'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score'
]

FEATURES_TO_SCALE = [
    'person_age', 'person_income', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'credit_score'
]

# Category mappings
CATEGORY_MAPPINGS = {
    'person_gender': ['female', 'male'],
    'person_education': ['Associate', 'Bachelor', 'Doctorate', 'High School', 'Master'],
    'person_home_ownership': ['MORTGAGE', 'OTHER', 'OWN', 'RENT'],
    'loan_intent': ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 
                   'MEDICAL', 'PERSONAL', 'VENTURE'],
    'previous_loan_defaults_on_file': ['No', 'Yes']
}