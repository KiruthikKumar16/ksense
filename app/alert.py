import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from utils import get_env

ALERT_EMOTIONS = ["sad", "angry"]
ALERT_DURATION = 15  # seconds

_last_alert_time = None

def check_and_alert(emotion_log):
    global _last_alert_time
    if not emotion_log:
        return
    now = datetime.now()
    # Filter for recent ALERT_DURATION seconds
    recent = [e for t, e in emotion_log if (now - t).total_seconds() <= ALERT_DURATION]
    if recent and all(e in ALERT_EMOTIONS for e in recent):
        if not _last_alert_time or (now - _last_alert_time).total_seconds() > ALERT_DURATION:
            send_email_alert(recent[-1])
            _last_alert_time = now

def send_email_alert(emotion):
    user = get_env('EMAIL_USER')
    pwd = get_env('EMAIL_PASS')
    to = get_env('ALERT_EMAIL')
    if not (user and pwd and to):
        return
    msg = MIMEText(f"Alert: Detected sustained negative emotion ({emotion}) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    msg['Subject'] = 'ksense Alert: Negative Emotion Detected'
    msg['From'] = user
    msg['To'] = to
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(user, pwd)
            server.sendmail(user, to, msg.as_string())
    except Exception as e:
        pass 