import pandas as pd
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pycaret.classification import setup, compare_models, pull
from langchain.tools import tool 
import imaplib
import email
import io
import re

# Target accuracy constants
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
        # Connect with explicit port 993 (fixes connection errors)
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993) 
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        # Robust IMAP search syntax (fixes BAD command error)
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

# @tool decorator is REMOVED
def run_pycaret_auto_ml(df: pd.DataFrame) -> str:
    """Performs PyCaret AutoML..."""
    print("Tool: Starting PyCaret AutoML process...")
    try:
        target_col = df.columns[-1] 
        setup(df, target=target_col, silent=True, verbose=False, session_id=42) 
        
        # CRITICAL CHANGE: Limit models to prevent memory/CPU crash on GitHub Actions
        best_model = compare_models(
            n_select=1, 
            exclude=['lightgbm', 'xgboost', 'catboost', 'svm', 'rbfsvm', 'ridge'] 
        )
        
        metrics = pull()
        primary_metric = metrics.columns[1] 
        best_metric_value = metrics.loc[metrics['Model'] == best_model.__class__.__name__, primary_metric].iloc[0]
        
        report = (
            f"PyCaret Auto-Detected Task: {setup(df, target=target_col, silent=True, verbose=False).pipeline.steps[0][0]}\n"
            f"Best Model Found: {best_model.__class__.__name__}\n"
            f"Primary Metric ({primary_metric}): {best_metric_value:.4f}\n"
            f"Full Metrics:\n{metrics.to_string()}"
        )
        print("Tool: PyCaret completed.")
        return report
    except Exception as e:
        # Returning a clear error report if PyCaret fails
        return f"PyCaret Error: Failed to run AutoML. Error: {e}"

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
