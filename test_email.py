import os
import logging
from pathlib import Path
from email.mime.text import MIMEText
import smtplib

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def load_env(path: Path = Path('.env')):
    if not path.exists():
        # Try python-dotenv if available
        try:
            from dotenv import load_dotenv as _ld
            loaded = _ld(dotenv_path=str(path), override=False)
            logging.info(f"python-dotenv attempted load, success={loaded}")
            return
        except Exception:
            logging.warning(".env not found; relying on process env")
            return
    raw = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    pairs = {}
    for line in raw:
        line_s = line.strip()
        if not line_s or line_s.startswith('#') or '=' not in line_s:
            continue
        k, v = line_s.split('=', 1)
        k = k.strip().lstrip('\ufeff')
        v = v.strip().strip('"').strip("'")
        pairs[k] = v
        if k and k not in os.environ:
            os.environ[k] = v
    # Diagnostics: show what keys we detected
    logging.info("Detected keys in .env: " + ", ".join(sorted(pairs.keys())))


def send_test_email():
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '0') or 0)
    user = os.getenv('SMTP_USER')
    pw = os.getenv('SMTP_PASS')
    to_addr = os.getenv('ALERT_TO') or os.getenv('SMTP_USER')

    if not (host and user and pw and to_addr):
        raise SystemExit("SMTP not fully configured. Ensure SMTP_HOST, SMTP_USER, SMTP_PASS, and ALERT_TO (or SMTP_USER) are set.")

    msg = MIMEText("This is a test email from AQI app SMTP check.")
    msg['Subject'] = 'AQI App SMTP Test'
    msg['From'] = user
    msg['To'] = to_addr

    # Try STARTTLS (587), then SSL (465)
    last_err = None
    try:
        use_port = port or 587
        logging.info(f"Trying STARTTLS on {host}:{use_port} -> {to_addr}")
        with smtplib.SMTP(host, use_port, timeout=10) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        print("OK: Sent via STARTTLS")
        return
    except Exception as e:
        logging.warning(f"STARTTLS failed: {e}")
        last_err = e

    try:
        logging.info(f"Trying SSL on {host}:465 -> {to_addr}")
        with smtplib.SMTP_SSL(host, 465, timeout=10) as s:
            s.login(user, pw)
            s.send_message(msg)
        print("OK: Sent via SSL")
        return
    except Exception as e2:
        logging.error(f"SSL failed: {e2}")
        raise SystemExit(f"Failed to send test email. Last error: {e2} | First error: {last_err}")


if __name__ == '__main__':
    load_env()
    # Masked echo of important vars
    host = os.getenv('SMTP_HOST')
    port = os.getenv('SMTP_PORT')
    user = os.getenv('SMTP_USER')
    to = os.getenv('ALERT_TO') or os.getenv('SMTP_USER')
    logging.info(f"CWD={Path('.').resolve()} | .env exists={Path('.env').exists()}")
    logging.info(f"SMTP_HOST={host}, SMTP_PORT={port}, SMTP_USER={'***' if not user else user[:2]+'***'}, TO={'***' if not to else to[:2]+'***'}")
    send_test_email()
