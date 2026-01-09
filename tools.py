import pandas as pd
import os, smtplib, imaplib, email, io, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from google import genai
from google.genai import errors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set the stable model ID
MODEL_ID = "gemini-2.0-flash" 

def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def retry_on_quota(func):
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try: 
                return func(*args, **kwargs)
            except Exception as e:
                # Handle both Quota (429) and unexpected Model errors
                if "429" in str(e):
                    print(f"Quota exceeded, waiting 10s (Attempt {attempt+1})...")
                    time.sleep(10)
                elif "404" in str(e):
                    print("Model ID error. Ensure the model name is correct.")
                    raise e
                else: 
                    raise e
        return func(*args, **kwargs)
    return wrapper

@retry_on_quota
def find_and_validate_email():
    """Validates conditions: Problem Statement, ML Project context, and Dataset."""
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(os.getenv("AGENT_EMAIL"), os.getenv("AGENT_PASSWORD"))
    mail.select('inbox')
    
    status, messages = mail.search(None, 'ALL')
    email_ids = messages[0].split()[-5:] # Check last 5
    
    client = get_gemini_client()
    
    for e_id in reversed(email_ids):
        res, msg_data = mail.fetch(e_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        subject = str(msg['Subject'])
        sender = str(msg['From'])
        
        # Check for dataset attachment
        has_dataset = any(p.get_filename() and p.get_filename().lower().endswith(('.csv', '.txt')) for p in msg.walk())
        
        # Validation prompt
        prompt = (f"Analyze this email. Does it have a clear Problem Statement and an ML project request? "
                  f"Dataset attached: {has_dataset}. Subject: {subject}. Reply ONLY with the word 'VALID' or 'INVALID'.")
        
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            if "VALID" in response.text.upper() and has_dataset:
                mail.logout()
                return e_id.decode(), sender, subject
        except Exception as e:
            print(f"API Call failed for UID {e_id.decode()}: {e}")
            continue
            
    mail.logout()
    return None, None, None

def download_dataset_by_uid(uid):
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(os.getenv("AGENT_EMAIL"), os.getenv("AGENT_PASSWORD"))
    mail.select('inbox')
    res, data = mail.fetch(uid, '(RFC822)')
    msg = email.message_from_bytes(data[0][1])
    for part in msg.walk():
        if part.get_filename() and part.get_filename().lower().endswith(('.csv', '.txt')):
            return pd.read_csv(io.BytesIO(part.get_payload(decode=True)))
    return None

def send_email_report(to_email, subject, body, attachment=None):
    msg = MIMEMultipart()
    msg['From'] = os.getenv("AGENT_EMAIL")
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    if attachment and os.path.exists(attachment):
        with open(attachment, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(attachment)}")
            msg.attach(part)
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls()
        s.login(os.getenv("AGENT_EMAIL"), os.getenv("AGENT_PASSWORD"))
        s.send_message(msg)

def generate_visualizations(df):
    pdf_name = "analysis_report.pdf"
    with PdfPages(pdf_name) as pdf:
        plt.figure(figsize=(10,6))
        df.iloc[:, :min(5, len(df.columns))].hist()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    return pdf_name
