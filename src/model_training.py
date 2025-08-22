# ===================================
# FILE: src/model_training.py
# ===================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import streamlit as st
from utils.constants import MODEL_PARAMS, FEATURES_TO_SCALE

@st.cache_resource
def train_loan_model(df):
    """Train the loan approval model"""
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_PARAMS['test_size'],
        stratify=y,
        random_state=MODEL_PARAMS['random_state']
    )
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), FEATURES_TO_SCALE)
    ], remainder='passthrough')
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('smote', SMOTE(random_state=MODEL_PARAMS['random_state'])),
        ('clf', RandomForestClassifier(
            n_estimators=MODEL_PARAMS['n_estimators'],
            max_depth=MODEL_PARAMS['max_depth'],
            min_samples_split=MODEL_PARAMS['min_samples_split'],
            min_samples_leaf=MODEL_PARAMS['min_samples_leaf'],
            class_weight='balanced',
            random_state=MODEL_PARAMS['random_state'],
            n_jobs=-1
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return pipeline, metrics, X_test, y_test, y_pred, y_proba

def get_feature_importance(model, feature_names):
    """Get feature importance from trained model"""
    try:
        importances = model.named_steps['clf'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        return feature_importance
    except Exception as e:
        st.error(f"Error getting feature importance: {e}")
        return None