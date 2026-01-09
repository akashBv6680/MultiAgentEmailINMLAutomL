"""
Advanced ML Module (2026 Update)
Optimized for Gemini 2.5 Flash Agentic Workflows.
Includes: Automated Classification/Regression, Visualization, and Data Ingestion.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION CONSTANTS ---
# Used by the Orchestrator Agent to decide on deployment status
TARGET_ACCURACY_MIN = 0.75 

class AdvancedMLPipeline:
    """Core ML Engine that dynamically selects between Classification and Regression."""
    
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col or df.columns[-1]
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = [None]*4
        self.results = {}
        self.is_classification = False
        self.problem_type = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Dynamic Encoding and Problem Type detection."""
        df_p = self.df.copy()
        
        # Handle Categorical Features
        cat_cols = df_p.select_dtypes(include=['object']).columns.tolist()
        if self.target_col in cat_cols: cat_cols.remove(self.target_col)
        
        for col in cat_cols:
            if df_p[col].nunique() < 10:
                df_p = pd.get_dummies(df_p, columns=[col], drop_first=True)
            else:
                df_p[col] = LabelEncoder().fit_transform(df_p[col].astype(str))
        
        # Detect Target Type
        target = df_p[self.target_col]
        if target.dtype == 'object' or (len(target.unique()) <= 15 and all(target == target.astype(int))):
            self.is_classification = True
            self.problem_type = 'Classification'
            if target.dtype == 'object':
                df_p[self.target_col] = LabelEncoder().fit_transform(target)
        else:
            self.is_classification = False
            self.problem_type = 'Regression'
        
        self.X = df_p.drop(columns=[self.target_col])
        self.y = df_p[self.target_col]
        
        stratify_val = self.y if self.is_classification else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=stratify_val
        )
    
    def train_all_models(self):
        """Train standard and ensemble models."""
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(self.X_train)
        X_te = scaler.transform(self.X_test)
        
        if self.is_classification:
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, verbosity=0),
                'LightGBM': lgb.LGBMClassifier(n_estimators=100, verbose=-1),
                'SVM': SVC(probability=True)
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, n_jobs=-1),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, verbosity=0),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, verbose=-1),
                'KNN': KNeighborsRegressor()
            }

        for name, model in models.items():
            try:
                model.fit(X_tr, self.y_train)
                y_pred = model.predict(X_te)
                
                if self.is_classification:
                    self.results[name] = {
                        'Accuracy': accuracy_score(self.y_test, y_pred),
                        'F1-Score': f1_score(self.y_test, y_pred, average='weighted'),
                        'model': model
                    }
                else:
                    self.results[name] = {
                        'R2': r2_score(self.y_test, y_pred),
                        'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                        'model': model
                    }
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")
        return self.results

    def get_best_model(self):
        metric = 'Accuracy' if self.is_classification else 'R2'
        best_name = max(self.results, key=lambda x: self.results[x][metric])
        return best_name, self.results[best_name]

# --- REQUIRED TOOL FUNCTIONS FOR LANGGRAPH ---

def download_dataset_from_email() -> Optional[pd.DataFrame]:
    """
    Mock Data Ingestion Tool. 
    In a production 2026 setup, this would connect to the Gmail/Outlook API.
    """
    print("Tool: Ingesting data from simulated source...")
    # Generating a synthetic dataset for demonstration
    np.random.seed(42)
    data = {
        'Feature_A': np.random.rand(100),
        'Feature_B': np.random.rand(100) * 100,
        'Category': np.random.choice(['High', 'Low', 'Medium'], 100),
        'Target': np.random.rand(100) * 50 # Regression Target
    }
    return pd.DataFrame(data)

def generate_visualizations(df: pd.DataFrame):
    """Generates and saves visual report for the human-in-the-loop."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.savefig("correlation_report.png")
    plt.close()
    print("Tool: Visualization saved as correlation_report.png")

def run_manual_ml(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Main entry point called by the ML Training Agent.
    Returns (Detailed Report, Best Metric Score).
    """
    pipeline = AdvancedMLPipeline(df)
    results = pipeline.train_all_models()
    
    if not results:
        return "Critical Error: No models converged.", 0.0
    
    best_name, best_metrics = pipeline.get_best_model()
    metric_key = 'Accuracy' if pipeline.is_classification else 'R2'
    score = best_metrics[metric_key]
    
    # Format a dense report for Gemini 2.5 Flash to analyze
    report = f"ML Analysis Report ({pipeline.problem_type})\n"
    report += f"Rows: {df.shape[0]} | Columns: {df.shape[1]}\n"
    report += f"Best Performing Model: {best_name}\n"
    report += f"Top Metric ({metric_key}): {score:.4f}\n"
    
    # Feature Importance logic
    try:
        if hasattr(best_metrics['model'], 'feature_importances_'):
            importances = pd.Series(best_metrics['model'].feature_importances_, index=pipeline.X.columns)
            top_feats = importances.nlargest(3).to_dict()
            report += f"Top Features: {top_feats}"
    except: pass
    
    return report, score
