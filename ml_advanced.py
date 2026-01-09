"""
Advanced ML Module with Multiple Algorithms
Supports Classification and Regression with comprehensive metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
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

class AdvancedMLPipeline:
    """Advanced ML Pipeline supporting both Classification and Regression"""
    
    def __init__(self, df, target_col=None):
        self.df = df.copy()
        self.target_col = target_col or df.columns[-1]
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.is_classification = False
        self.problem_type = None
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Handle data preprocessing and encoding"""
        df_processed = self.df.copy()
        
        # Identify categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        
        # Dynamic encoding
        for col in categorical_cols:
            if df_processed[col].nunique() == 2:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        
        # One-hot encoding for multi-class categorical
        df_processed = pd.get_dummies(df_processed, drop_first=True)
        
        # Check if target is classification or regression
        target = df_processed[self.target_col]
        if target.dtype == 'object':
            le = LabelEncoder()
            df_processed[self.target_col] = le.fit_transform(target)
            self.is_classification = True
            self.problem_type = 'Classification'
        elif len(target.unique()) <= 10 and all(target == target.astype(int)):
            self.is_classification = True
            self.problem_type = 'Classification'
        else:
            self.is_classification = False
            self.problem_type = 'Regression'
        
        self.X = df_processed.drop(columns=[self.target_col])
        self.y = df_processed[self.target_col]
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y if self.is_classification else None
        )
    
    def train_all_models(self):
        """Train all available models"""
        print(f"\n=== Training {self.problem_type} Models ===")
        
        if self.is_classification:
            self._train_classification_models()
        else:
            self._train_regression_models()
        
        return self.results
    
    def _train_regression_models(self):
        """Train all regression models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                
                self.results[name] = {
                    'R2': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'model': model,
                    'scaler': scaler
                }
                print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            except Exception as e:
                print(f"{name} failed: {str(e)}")
    
    def _train_classification_models(self):
        """Train all classification models"""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'SVM': SVC(kernel='rbf', random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                self.results[name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'model': model,
                    'scaler': scaler
                }
                print(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            except Exception as e:
                print(f"{name} failed: {str(e)}")
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.results:
            return None
        
        metric = 'R2' if not self.is_classification else 'Accuracy'
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        return best_model_name, self.results[best_model_name]
    
    def get_detailed_report(self):
        """Generate detailed performance report"""
        report = f"\n{'='*70}\n"
        report += f"Problem Type: {self.problem_type}\n"
        report += f"Dataset Shape: {self.df.shape}\n"
        report += f"Training Set: {self.X_train.shape[0]}, Test Set: {self.X_test.shape[0]}\n"
        report += f"{'='*70}\n\n"
        
        if self.is_classification:
            report += "CLASSIFICATION MODELS PERFORMANCE:\n"
            report += "-"*70 + "\n"
            for model_name, metrics in sorted(self.results.items(), key=lambda x: x[1]['Accuracy'], reverse=True):
                report += f"\n{model_name}:\n"
                report += f"  Accuracy: {metrics['Accuracy']:.4f}\n"
                report += f"  Precision: {metrics['Precision']:.4f}\n"
                report += f"  Recall: {metrics['Recall']:.4f}\n"
                report += f"  F1-Score: {metrics['F1-Score']:.4f}\n"
        else:
            report += "REGRESSION MODELS PERFORMANCE:\n"
            report += "-"*70 + "\n"
            for model_name, metrics in sorted(self.results.items(), key=lambda x: x[1]['R2'], reverse=True):
                report += f"\n{model_name}:\n"
                report += f"  R-squared: {metrics['R2']:.4f}\n"
                report += f"  RMSE: {metrics['RMSE']:.4f}\n"
                report += f"  MAE: {metrics['MAE']:.4f}\n"
        
        best_name, best_metrics = self.get_best_model()
        metric_key = list(best_metrics.keys())[0]
        report += f"\n{'='*70}\n"
        report += f"BEST MODEL: {best_name}\n"
        report += f"Best {metric_key}: {best_metrics[metric_key]:.4f}\n"
        report += f"{'='*70}\n"
        
        return report

def run_advanced_ml(df: pd.DataFrame) -> tuple[str, float]:
    """Main function to run advanced ML pipeline"""
    print("\nStarting Advanced ML Pipeline...")
    
    pipeline = AdvancedMLPipeline(df)
    results = pipeline.train_all_models()
    
    if not results:
        return "ML Pipeline Error: No models trained successfully", None
    
    best_model_name, best_metrics = pipeline.get_best_model()
    metric_key = list(best_metrics.keys())[0]
    best_score = best_metrics[metric_key]
    
    report = pipeline.get_detailed_report()
    report += f"\nTop 5 Features (if available):\n"
    
    # Get feature importance if model supports it
    try:
        if hasattr(best_metrics['model'], 'feature_importances_'):
            importance = pd.Series(
                best_metrics['model'].feature_importances_,
                index=pipeline.X.columns
            ).nlargest(5)
            for feat, imp in importance.items():
                report += f"  {feat}: {imp:.4f}\n"
    except:
        pass
    
    return report, best_score
