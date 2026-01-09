import pandas as pd
import os
import imaplib
import email
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import google.generativeai as genai
from ml_advanced import run_manual_ml  # Updated to match the function name in ml_advanced.py

# --- 1. CONFIGURATION & CONSTANTS ---
# Target accuracy thresholds for the Orchestrator Agent
TARGET_ACCURACY_MIN = 0.60
TARGET_ACCURACY_MAX = 1.0

def get_gemini_client():
    """
    Initialize Gemini API client using the GOOGLE_API_KEY GitHub secret.
    """
    api_key = os.getenv("GOOGLE_API_KEY") # Matches the YAML and main script
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Check GitHub Secrets.")
    genai.configure(api_key=api_key)
    return genai

# --- 2. DATA INGESTION ---

def download_dataset_from_email() -> pd.DataFrame:
    """
    Connects to the email client, searches for the latest 'Problem statement' email,
    downloads the CSV/TXT attachment, and returns it as a DataFrame.
    """
    subject_filter = 'Problem statement'
    AGENT_EMAIL = os.environ.get("AGENT_EMAIL")
    AGENT_PASSWORD = os.environ.get("AGENT_PASSWORD")
    
    if not AGENT_EMAIL or not AGENT_PASSWORD:
        raise ValueError("Email credentials (AGENT_EMAIL, AGENT_PASSWORD) missing.")
    
    print(f"Tool: Searching for data in inbox (Filter: '{subject_filter}')")
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        mail.login(AGENT_EMAIL, AGENT_PASSWORD)
        mail.select('inbox')
        
        # Search for unseen first, then fallback to all
        status, email_ids = mail.search(None, f'(UNSEEN SUBJECT \"{subject_filter}\")')
        if not email_ids[0]:
            status, email_ids = mail.search(None, f'(ALL SUBJECT \"{subject_filter}\")')
        
        if not email_ids[0]:
            raise FileNotFoundError(f"No emails found with subject: '{subject_filter}'")
        
        latest_email_id = email_ids[0].split()[-1]
        status, msg_data = mail.fetch(latest_email_id, '(RFC822)')
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
                
                print(f"Tool: Dataset '{filename}' ingested. Shape: {data.shape}")
                mail.logout()
                return data
        
        raise Exception("No valid CSV/TXT attachment found in the email.")
    except Exception as e:
        print(f"Tool Error: {e}")
        raise

# --- 3. MACHINE LEARNING & VISUALIZATION ---

def run_advanced_ml_tool(df: pd.DataFrame) -> tuple[str, float]:
    """
    Wrapper for the Advanced ML Module.
    """
    print("Tool: Executing Advanced ML Pipeline...")
    try:
        # Calls run_manual_ml from your ml_advanced.py
        report, best_score = run_manual_ml(df)
        return report, best_score
    except Exception as e:
        return f"Tool Error (ML): {str(e)}", 0.0

def generate_visualizations(df: pd.DataFrame) -> str:
    """
    Generates exploratory data analysis plots and saves them to a PDF.
    """
    pdf_name = "visual_report.pdf"
    plt.ioff() # Turn off interactive plotting
    
    try:
        with PdfPages(pdf_name) as pdf:
            # Generate plots for up to 10 columns to avoid massive files
            for col in df.columns[:10]: 
                plt.figure(figsize=(6, 4))
                
                if df[col].dtype == "object" or df[col].nunique() < 10:
                    df[col].value_counts().head(5).plot.bar()
                    plt.title(f"Frequency: {col}")
                elif np.issubdtype(df[col].dtype, np.number):
                    sns.histplot(df[col], kde=True)
                    plt.title(f"Distribution: {col}")
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        
        print(f"Tool: Visual report saved as {pdf_name}")
        return pdf_name
    except Exception as e:
        print(f"Tool Error (Viz): {e}")
        return ""

# --- 4. GEMINI API UTILITY ---

def call_gemini_api(prompt: str, model: str = "gemini-2.5-flash") -> str:
    """
    Standalone utility to call Gemini 2.5 Flash for agent reasoning.
    """
    try:
        get_gemini_client()
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {str(e)}"
