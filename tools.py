import pandas as pd
import os
import smtplib
import imaplib
import email
import io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from matplotlib.backends.backend_pdf import PdfPages
from google import genai

# Configuration
TARGET_ACCURACY_MIN = 0.60

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing.")
    return genai.Client(api_key=api_key)

def find_data_email():
    """Fetches recent email headers and uses Gemini to pick the right one."""
    AGENT_EMAIL = os.getenv("AGENT_EMAIL")
    AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")
    
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(AGENT_EMAIL, AGENT_PASSWORD)
    mail.select('inbox')
    
    # Fetch last 10 emails to analyze context
    status, messages = mail.search(None, 'ALL')
    email_ids = messages[0].split()[-10:]
    
    email_context = []
    for e_id in email_ids:
        res, msg_data = mail.fetch(e_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        email_context.append({
            "uid": e_id.decode(),
            "subject": str(msg['Subject']),
            "from": str(msg['From']),
            "snippet": msg.get_payload(0).get_payload()[:100] if msg.is_multipart() else msg.get_payload()[:100]
        })

    client = get_gemini_client()
    prompt = f"From this list of emails, identify the UID of the one most likely to contain a machine learning dataset or problem statement. Return ONLY the UID number:\n{email_context}"
    
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    target_uid = response.text.strip()
    
    # Also extract the sender's email to reply to them
    target_email = ""
    for item in email_context:
        if item["uid"] == target_uid:
            target_email = item["from"]
            break
            
    mail.logout()
    return target_uid, target_email

def download_dataset_by_uid(uid: str) -> pd.DataFrame:
    AGENT_EMAIL = os.getenv("AGENT_EMAIL")
    AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(AGENT_EMAIL, AGENT_PASSWORD)
    mail.select('inbox')
    
    res, data = mail.fetch(uid, '(RFC822)')
    msg = email.message_from_bytes(data[0][1])
    
    for part in msg.walk():
        if part.get_filename() and (part.get_filename().endswith('.csv') or part.get_filename().endswith('.txt')):
            payload = part.get_payload(decode=True)
            df = pd.read_csv(io.BytesIO(payload))
            mail.logout()
            return df
    mail.logout()
    raise FileNotFoundError("No dataset found in the selected email.")

def send_email_report(to_email: str, subject: str, body: str, attachment_path: str = None):
    AGENT_EMAIL = os.getenv("AGENT_EMAIL")
    AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")
    
    msg = MIMEMultipart()
    msg['From'] = AGENT_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
            
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(AGENT_EMAIL, AGENT_PASSWORD)
        server.send_message(msg)

def generate_visualizations(df: pd.DataFrame):
    pdf_name = "analysis_report.pdf"
    plt.ioff()
    with PdfPages(pdf_name) as pdf:
        for col in df.columns[:5]:
            plt.figure(figsize=(8,4))
            df[col].value_counts().head(10).plot(kind='bar')
            plt.title(f"Column Analysis: {col}")
            pdf.savefig()
            plt.close()
    return pdf_name
