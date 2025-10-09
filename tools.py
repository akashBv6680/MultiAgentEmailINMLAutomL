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
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# ------------------------------------

# Target accuracy constants (Now representing R2 score)
TARGET_ACCURACY_MIN = 0.80
TARGET_ACCURACY_MAX = 0.90

# @tool decorator is REMOVED
def download_dataset_from_email() -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest email with a data file (CSV), 
    downloads the attachment, and returns it as a pandas DataFrame.
    """
    subject_filter: str = 'Problem statement' 
    
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) not found in environment.")

    print(f"Tool: Attempting to connect to email server for data... (Filter: '{subject_filter}')")

    try:
        # Robust IMAP connection and search
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993) 
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        status, email_ids = mail.search(None, f'(UNSEEN SUBJECT "{subject_filter}")') 
        
        if not email_ids[0]:
            print(f"Tool: No NEW email found. Searching ALL emails with subject '{subject_filter}'.")
            status, email_ids = mail.search(None, f'(ALL SUBJECT "{subject_filter}")') 
            
            if not email_ids[0]:
                raise FileNotFoundError(f"No emails found with the subject filter: '{subject_filter}'.")

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
# --- NEW ML & VISUALIZATION FUNCTIONS (REPLACING PyCARET) ---
# ------------------------------------------------------------------------------------

def run_manual_ml(df: pd.DataFrame) -> tuple[str, float]:
    """
    Performs manual ML using Scikit-learn (RandomForestRegressor)
    with dynamic preprocessing, bypassing PyCaret resource issues.
    """
    print("Tool: Starting Manual Scikit-learn ML training (RandomForestRegressor)...")
    try:
        # 1. Preprocessing (DYNAMICALLY handle ALL features)
        df_processed = df.copy()
        target_col = df_processed.columns[-1]
        
        # Identify categorical columns (excluding the target column)
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # --- DYNAMIC ENCODING LOGIC ---
        for col in categorical_cols:
            # If a categorical column is binary (2 unique values), use LabelEncoder
            if df_processed[col].nunique() == 2:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
            # If it has more than 2 unique values, it will be handled by get_dummies below
        
        # One-Hot Encoding for all remaining 'object' columns
        df_processed = pd.get_dummies(df_processed, drop_first=True)

        # 2. Define Features (X) and Target (y)
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # 3. Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Create and Train Pipeline (Scaling + Model)
        # Using RFR for the regression problem
        model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        model.fit(X_train, y_train)

        # 5. Evaluate (using R2 score as the accuracy metric)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # 6. Generate Report (Feature Importance)
        feature_importance = pd.Series(model.named_steps['randomforestregressor'].feature_importances_, index=X.columns)
        top_features = feature_importance.nlargest(5).to_string()
        
        report = (
            f"Manual ML Training Completed (RandomForestRegressor):\n"
            f"Test Set R-squared Score (Accuracy Metric): {r2:.4f}\n"
            f"-------------------------------------------\n"
            f"Model Configuration:\n"
            f"- Model: RandomForestRegressor\n"
            f"- Preprocessing: Standard Scaling + Dynamic Encoding\n"
            f"-------------------------------------------\n"
            f"Top 5 Feature Importance:\n{top_features}"
        )
        print("Tool: Manual ML training completed.")
        return report, r2
        
    except Exception as e:
        # Include the full error in the string for debugging if needed
        return f"Manual ML Error: {e}", None


def generate_visualizations(df):
    """Generates visualizations and saves them to a PDF file."""
    pdf_name = "visual_report.pdf"
    plt.ioff()
    
    with PdfPages(pdf_name) as pdf:
        for col in df.columns:
            plt.figure(figsize=(6, 4))
            
            # Categorical/Low-Unique Count
            if df[col].dtype == "object" or df[col].nunique() < 10:
                if df[col].nunique() <= 5:
                    df[col].value_counts().plot.pie(autopct='%1.1f%%')
                    plt.title(f"Pie Chart - {col}")
                else:
                    sns.countplot(y=col, data=df)
                    plt.title(f"Bar Chart - {col}")
            # Numeric Data
            elif np.issubdtype(df[col].dtype, np.number):
                if df[col].nunique() < 20:
                    sns.histplot(df[col], kde=False)
                    plt.title(f"Hist Plot - {col}")
                else:
                    sns.kdeplot(df[col], fill=True)
                    plt.title(f"Dist Plot - {col}")
            else:
                plt.close()
                continue

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
    print(f"Tool: Visualizations saved to {pdf_name}")
    return pdf_name


# @tool decorator is REMOVED
def send_client_email(subject: str, body: str, to_email: str) -> bool:
    """Sends the final formatted email to the client."""
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")

    if not AGENT_EMAIL or not AGENT_PASSWORD: return False

    print(f"Tool: Attempting to send email to {to_email}...")
    try:
        msg = MIMEMultipart()
        msg['From'] = AGENT_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(AGENT_EMAIL, AGENT_PASSWORD)
        server.sendmail(AGENT_EMAIL, to_email, msg.as_string())
        server.quit()
        print("Tool: Email sent successfully!")
        return True
    except Exception as e:
        print(f"Tool: Failed to send email. Error: {e}")
        return False
