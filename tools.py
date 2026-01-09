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

def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def retry_on_quota(func):
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try: return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e): time.sleep(5)
                else: raise e
        return func(*args, **kwargs)
    return wrapper

@retry_on_quota
def find_and_validate_email():
    """Validates: 1. Problem Statement 2. ML Context 3. Dataset attached."""
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(os.getenv("AGENT_EMAIL"), os.getenv("AGENT_PASSWORD"))
    mail.select('inbox')
    
    status, messages = mail.search(None, 'ALL')
    email_ids = messages[0].split()[-5:] # Check last 5 emails
    
    client = get_gemini_client()
    
    for e_id in reversed(email_ids):
        res, msg_data = mail.fetch(e_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        subject = str(msg['Subject'])
        sender = str(msg['From'])
        
        # Check for dataset
        has_dataset = any(p.get_filename() and p.get_filename().lower().endswith(('.csv', '.txt')) for p in msg.walk())
        
        prompt = f"Analyze this email. Does it have a clear Problem Statement and an ML project request? Dataset attached: {has_dataset}. Subject: {subject}. Reply 'VALID' or 'INVALID'."
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        
        if "VALID" in response.text.upper() and has_dataset:
            mail.logout()
            return e_id.decode(), sender, subject
            
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
        s.sendmail(os.getenv("AGENT_EMAIL"), to_email, msg.as_string())

def generate_visualizations(df):
    pdf_name = "analysis_report.pdf"
    with PdfPages(pdf_name) as pdf:
        df.iloc[:, :5].hist(figsize=(10,6))
        pdf.savefig()
        plt.close()
    return pdf_name
