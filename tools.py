import pandas as pd
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain.tools import tool 
import imaplib
import email
import io
import re

# --- NEW ML/VISUALIZATION IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier # Adding Decision Tree
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ------------------------------------

# Target accuracy constants (Now representing R2 or Accuracy score)
TARGET_ACCURACY_MIN = 0.80
TARGET_ACCURACY_MAX = 0.90

# --- NEW: Expanded Subject Filter List ---
SUBJECT_FILTERS = [
    'problem statement', 
    'business problem', 
    'business use case', 
    'project details', 
    'analysis the project', 
    'dataanalysis project details'
]
# -----------------------------------------

# @tool decorator is REMOVED
def download_dataset_from_email() -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest email with a data file (CSV) 
    matching ANY of the defined subject filters, downloads, and returns it.
    """
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) not found in environment.")

    # Create the IMAP search query for multiple subjects
    search_terms = [f'SUBJECT "{s}"' for s in SUBJECT_FILTERS]
    # Search for (UNSEEN OR ALL) AND (SUBJECT1 OR SUBJECT2 OR ...)
    search_query = f'(OR {" ".join(search_terms)})'
    
    print(f"Tool: Attempting to connect to email server for data... (Filters: {SUBJECT_FILTERS})")

    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993) 
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        # 1. Search for UNSEEN email matching any subject
        status, email_ids = mail.search(None, f'(UNSEEN {search_query})') 
        
        if not email_ids[0]:
            print(f"Tool: No NEW email found. Searching ALL emails.")
            # 2. Search for ALL email matching any subject
            status, email_ids = mail.search(None, f'(ALL {search_query})') 
            
            if not email_ids[0]:
                raise FileNotFoundError(f"No emails found matching any of the subject filters: {SUBJECT_FILTERS}")

        latest_email_id = email_ids[0].split()[-1]
        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        mail.store(latest_email_id, '+FLAGS', '\\Seen') 

        msg = email.message_from_bytes(msg_data[0][1])

        # Find and Download the CSV/TXT Attachment
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            
            filename = part.get_filename()
            if filename and (filename.endswith('.csv') or filename.endswith('.txt')):
                payload = part.get_payload(decode=True)
                
                try:
                    data = pd.read_csv(io.StringIO(payload.decode('utf-8')))
                except UnicodeDecodeError:
                    data = pd.read_csv(io.StringIO(payload.decode('latin-1')))

                print(f"Tool: Dataset '{filename}' downloaded successfully. Shape: {data.shape}")
                mail.logout()
                return data

        raise Exception(f"Email found, but no CSV or TXT attachment was detected.")
        
    except Exception as e:
        print(f"Tool: Failed to fetch email or attachment. Error: {e}")
        raise

# ------------------------------------------------------------------------------------

def run_manual_ml(df: pd.DataFrame) -> tuple[str, float]:
    """
    Performs dynamic ML (Classification/Regression) based on the target column, 
    using a range of Scikit-learn algorithms.
    """
    df_processed = df.copy()
    target_col = df_processed.columns[-1]
    
    # 1. Determine Task Type (Classification vs. Regression)
    # Check if the target column is numerical and has more than 20 unique values.
    # If not, assume it's a classification or categorical regression problem (and use classification metrics/models).
    is_regression = np.issubdtype(df_processed[target_col].dtype, np.number) and df_processed[target_col].nunique() > 20
    
    # 2. Dynamic Preprocessing
    try:
        # Identify categorical columns (excluding the target column)
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            # If a categorical column is binary (2 unique values), use LabelEncoder
            if df_processed[col].nunique() == 2:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
        
        # One-Hot Encoding for all remaining 'object' columns
        df_processed = pd.get_dummies(df_processed, drop_first=True)

        # 3. Define Features (X) and Target (y)
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # 4. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 5. Dynamic Model Selection (Simple Ensemble/Linear Models for efficiency)
        if is_regression:
            print("Tool: Auto-Detected Regression Task. Using RandomForestRegressor.")
            Model_class = RandomForestRegressor
            Metric_func = r2_score
            Metric_name = "R-squared Score"
            # Define a small list of models to try
            models_to_test = [
                ('RandomForest', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
                ('LinearRegression', LinearRegression()),
            ]
        else:
            print("Tool: Auto-Detected Classification Task. Using RandomForestClassifier.")
            Model_class = RandomForestClassifier
            Metric_func = accuracy_score
            Metric_name = "Accuracy Score"
            # Define a small list of models to try
            models_to_test = [
                ('RandomForest', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
                ('LogisticRegression', LogisticRegression(max_iter=1000, n_jobs=-1)),
            ]
        
        # 6. Train and Evaluate Best Model
        best_score = -np.inf
        best_model_name = ""
        best_model_instance = None
        
        for name, model_instance in models_to_test:
            pipeline = make_pipeline(StandardScaler(), model_instance)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            score = Metric_func(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_model_name = name
                best_model_instance = pipeline.named_steps[model_instance.__class__.__name__.lower()]

        # 7. Generate Report
        accuracy = best_score
        
        # Feature Importance (only for tree-based models)
        if hasattr(best_model_instance, 'feature_importances_'):
            feature_importance = pd.Series(best_model_instance.feature_importances_, index=X.columns)
            top_features_report = feature_importance.nlargest(5).to_string()
        else:
            top_features_report = "Feature importance not available for this model."

        report = (
            f"Manual ML Training Completed ({'Regression' if is_regression else 'Classification'}):\n"
            f"Best Model Selected: {best_model_name}\n"
            f"Test Set {Metric_name} (Accuracy Metric): {accuracy:.4f}\n"
            f"-------------------------------------------\n"
            f"Top 5 Feature Importance:\n{top_features_report}"
        )
        print("Tool: Manual ML training completed.")
        return report, accuracy
        
    except Exception as e:
        return f"Manual ML Error: {e}", None


# --- generate_visualizations remains the same (for brevity, use the previous corrected code) ---
def generate_visualizations(df):
    """Generates visualizations and saves them to a PDF file."""
    # ... (code remains the same)
    pass 

# @tool decorator is REMOVED
def send_client_email(subject: str, body: str, to_email: str) -> bool:
    """Sends the final formatted email to the client."""
    # ... (code remains the same)
    pass
