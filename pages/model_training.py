# ===================================
# FILE: pages/model_training.py
# ===================================

import streamlit as st
from src.data_processing import load_sample_data, preprocess_data
from src.model_training import train_loan_model, get_feature_importance
from src.visualization import plot_confusion_matrix, plot_feature_importance

def show():
    """Display model training page"""
    st.header("🤖 Model Training & Evaluation")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model... This may take a few minutes."):
            # Load and preprocess data
            df_raw = load_sample_data()
            df_processed, encoders = preprocess_data(df_raw)
            
            # Train model
            model, metrics, X_test, y_test, y_pred, y_proba = train_loan_model(df_processed)
            
            # Store in session state
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['feature_names'] = X_test.columns.tolist()
        
        # Display metrics
        st.subheader("📊 Model Performance")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cm = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            feature_importance = get_feature_importance(model, X_test.columns.tolist())
            if feature_importance is not None:
                fig_fi = plot_feature_importance(feature_importance)
                st.plotly_chart(fig_fi, use_container_width=True)
        
        st.success("✅ Model trained successfully!")
    
    else:
        st.info("Click the 'Train Model' button to start training.")
