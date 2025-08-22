# ===================================
# FILE: src/visualization.py
# ===================================

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_target_distribution(df):
    """Plot target variable distribution"""
    target_counts = df['loan_status'].value_counts()
    
    fig = px.pie(
        values=target_counts.values,
        names=['Rejected', 'Approved'],
        title="Loan Status Distribution",
        color_discrete_map={'Rejected': '#ff7f7f', 'Approved': '#7fbf7f'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_numerical_distribution(df, column):
    """Plot numerical feature distribution"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f'Distribution of {column}', f'{column} by Loan Status'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=30, name='Distribution'),
        row=1, col=1
    )
    
    # Box plot by loan status
    for status, label in [(0, 'Rejected'), (1, 'Approved')]:
        fig.add_trace(
            go.Box(
                y=df[df['loan_status'] == status][column],
                name=label,
                boxpoints='outliers'
            ),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=True)
    return fig

def plot_categorical_distribution(df, column):
    """Plot categorical feature distribution by loan status"""
    crosstab = pd.crosstab(df[column], df['loan_status'], normalize='index') * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Rejected',
        x=crosstab.index,
        y=crosstab[0],
        marker_color='#ff7f7f'
    ))
    
    fig.add_trace(go.Bar(
        name='Approved',
        x=crosstab.index,
        y=crosstab[1],
        marker_color='#7fbf7f'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f'{column} Distribution by Loan Status (%)',
        xaxis_title=column,
        yaxis_title='Percentage'
    )
    
    return fig

def plot_correlation_matrix(df):
    """Plot correlation matrix"""
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    
    fig.update_layout(width=700, height=600)
    return fig

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=['Rejected', 'Approved'],
        y=['Rejected', 'Approved'],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(title="Confusion Matrix", width=500, height=400)
    return fig

def plot_feature_importance(importance_df):
    """Plot feature importance"""
    top_features = importance_df.head(10)
    
    fig = px.bar(
        top_features.sort_values('importance'),
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Feature Importances",
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    
    fig.update_layout(height=500)
    return fig