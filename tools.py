import pandas as pd
import os
import smtplib
import imaplib
import email
import io
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from google import genai
from google.genai import errors

def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)

def retry_on_quota(func):
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try: return func(*args, **kwargs)
            except errors.ClientError:
                time.sleep(5)
        return func(*args, **kwargs)
    return wrapper

@retry_on_quota
def find_and_validate_email():
    """Searches inbox and validates if an email meets the ML Project criteria."""
    AGENT_EMAIL = os.getenv("AGENT_EMAIL")
    AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")
    
    mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
    mail.login(AGENT_EMAIL, AGENT_PASSWORD)
    mail.select('inbox')
    
    # Fetch recent emails
    status, messages = mail.search(None, 'ALL')
    email_ids = messages[0].split()[-5:]
    
    client = get_gemini_client()
    
    for e_id in reversed(email_ids):
        res, msg_data = mail.fetch(e_id, '(RFC822)')
        msg = email.message_from_bytes(msg_data[0][1])
        subject = str(msg['Subject'])
        sender = str(msg['From'])
        
        # Get body content
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = msg.get_payload(decode=True).decode()

        # Check for attachments
        has_dataset = any(p.get_filename() and p.get_filename().endswith(('.csv', '.txt')) 
                         for p in msg.walk())

        # Logic Gate: Use Gemini to check for "Problem Statement" and "ML Project" context
        prompt = f"""
        Analyze this email. Does it contain:
        1. A clear Problem Statement (what needs to be solved)?
        2. Mention of an ML Project or Model building?
        3. A dataset is attached: {has_dataset}

        Email Subject: {subject}
        Email Body: {body[:500]}
        
        Respond with 'VALID' if all 3 conditions are met, otherwise respond 'INVALID'.
        """
        
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        if "VALID" in response.text.upper():
            mail.logout()
            return e_id.decode(), sender, subject
            
    mail.logout()
    return None, None, None

def send_email_report(to_email, subject, body, attachment=None):
    AGENT_EMAIL = os.getenv("AGENT_EMAIL")
    AGENT_PASSWORD = os.getenv("AGENT_PASSWORD")
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
        s.login(AGENT_EMAIL, AGENT_PASSWORD)
        s.sendmail(AGENT_EMAIL, to_email, msg.as_string())
