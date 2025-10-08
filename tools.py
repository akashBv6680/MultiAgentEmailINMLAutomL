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

@tool
# --- FIXED DEFAULT SUBJECT FILTER ---
def download_dataset_from_email(subject_filter: str = 'Problem statement') -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest email with a data file (CSV), 
    downloads the attachment, and returns it as a pandas DataFrame.
    """
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) not found in environment.")

    # Using the local Ollama service URL defined in the YAML
    print(f"Tool: Attempting to connect to email server for data... (Filter: '{subject_filter}')")

    try:
        # 1. Connect to the IMAP server (Assuming Gmail)
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        # 2. Search for the latest email (searching for unseen or all with the subject)
        status, email_ids = mail.search(None, 'UNSEEN', 'HEADER', 'Subject', subject_filter) 
        
        if not email_ids[0]:
            print(f"Tool: No NEW email found. Searching ALL emails with subject '{subject_filter}'.")
            status, email_ids = mail.search(None, 'ALL', 'HEADER', 'Subject', subject_filter)
            if not email_ids[0]:
                raise FileNotFoundError(f"No emails found with the subject filter: '{subject_filter}'.")

        latest_email_id = email_ids[0].split()[-1]
        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        mail.store(latest_email_id, '+FLAGS', '\\Seen') # Mark as read

        msg = email.message_from_bytes(msg_data[0][1])

        # 3. Find and Download the CSV/TXT Attachment
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

@tool
def run_pycaret_auto_ml(df: pd.DataFrame) -> str:
    """Performs PyCaret AutoML..."""
    print("Tool: Starting PyCaret AutoML process...")
    try:
        target_col = df.columns[-1] 
        setup(df, target=target_col, silent=True, verbose=False, session_id=42)
        best_model = compare_models()
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
        return f"PyCaret Error: Failed to run AutoML. Error: {e}"

@tool
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
