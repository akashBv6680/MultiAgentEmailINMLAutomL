import pandas as pd
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import email
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import google.generativeai as genai
from ml_advanced import run_advanced_ml

# Target accuracy constants
TARGET_ACCURACY_MIN = 0.60
TARGET_ACCURACY_MAX = 1.0

# Configure Gemini API
def get_gemini_client():
    """Initialize Gemini API client using GitHub secret"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    return genai

def download_dataset_from_email() -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest email with a data file (CSV),
    downloads the attachment, and returns it as a pandas DataFrame.
    """
    subject_filter = 'Problem statement'
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) not found in environment.")
    
    print(f"Tool: Attempting to connect to email server for data... (Filter: '{subject_filter}')")
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        status, email_ids = mail.search(None, f'(UNSEEN SUBJECT \"{subject_filter}\")')
        
        if not email_ids[0]:
            print(f"Tool: No NEW email found. Searching ALL emails with subject '{subject_filter}'.")
            status, email_ids = mail.search(None, f'(ALL SUBJECT \"{subject_filter}\")')
        
        if not email_ids[0]:
            raise FileNotFoundError(f"No emails found with the subject filter: '{subject_filter}'.")
        
        latest_email_id = email_ids[0].split()[-1]
        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
        mail.store(latest_email_id, '+FLAGS', '\\\\Seen')
        msg = email.message_from_bytes(msg_data[0][1])
        
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

def run_manual_ml(df: pd.DataFrame) -> tuple[str, float]:
    """
    Run advanced ML training with multiple algorithms using the advanced ML module.
    Handles both classification and regression with comprehensive metrics.
    """
    print("Tool: Starting Advanced ML Training with Multiple Algorithms...")
    try:
        report, best_score = run_advanced_ml(df)
        if best_score is None:
            return report, None
        print("Tool: Advanced ML training completed successfully.")
        return report, best_score
    except Exception as e:
        return f"Advanced ML Error: {str(e)}", None

def generate_visualizations(df: pd.DataFrame) -> str:
    """Generates visualizations and saves them to a PDF file."""
    pdf_name = "visual_report.pdf"
    plt.ioff()
    
    try:
        with PdfPages(pdf_name) as pdf:
            for col in df.columns:
                plt.figure(figsize=(6, 4))
                
                if df[col].dtype == "object" or df[col].nunique() < 10:
                    if df[col].nunique() <= 5:
                        df[col].value_counts().plot.pie(autopct='%1.1f%%')
                        plt.title(f"Pie Chart - {col}")
                    else:
                        sns.countplot(y=col, data=df)
                        plt.title(f"Bar Chart - {col}")
                elif np.issubdtype(df[col].dtype, np.number):
                    if df[col].nunique() < 20:
                        sns.histplot(df[col], kde=False)
                        plt.title(f"Histogram - {col}")
                    else:
                        sns.kdeplot(df[col], fill=True)
                        plt.title(f"Distribution - {col}")
                else:
                    plt.close()
                    continue
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        print(f"Tool: Visualizations saved to {pdf_name}")
        return pdf_name
    except Exception as e:
        print(f"Tool: Failed to generate visualizations. Error: {e}")
        return None

def send_client_email(subject: str, body: str, to_email: str) -> bool:
    """Sends the final formatted email to the client."""
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        return False
    
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

def call_gemini_api(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Call Google Gemini 2.5 Flash API for text generation.
    """
    try:
        get_gemini_client()
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Failed to get response from Gemini: {str(e)}"
