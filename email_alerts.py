#!/usr/bin/env python3
"""
Professional email alerting module for Pullus Competitor Analysis
Sends HTML emails via Gmail SMTP with market intelligence insights
"""

import base64
import binascii
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from jinja2 import Template
from premailer import transform
from google.oauth2.service_account import Credentials
import logging

logger = logging.getLogger(__name__)

class EmailAlerts:
    def __init__(self, config=None):
        """Initialize email alerts with configuration"""
        self.config = config or {}

        # Load from environment (supports both CI and local .env)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT')
        self.google_credentials = self._load_google_credentials()

        # Validate required credentials
        if not all([self.smtp_username, self.smtp_password, self.email_sender, self.email_recipient]):
            raise ValueError("Missing required email credentials in environment")

    def _load_google_credentials(self):
        """Load Google service account credentials from JSON content or file path"""
        credentials_value = os.getenv('GOOGLE_CREDENTIALS_PATH')
        if not credentials_value:
            return None

        credentials_value = credentials_value.strip()
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]

        try:
            json_sources = []

            if credentials_value.startswith('{'):
                json_sources.append(credentials_value)
            else:
                try:
                    decoded_bytes = base64.b64decode(credentials_value, validate=True)
                    decoded_str = decoded_bytes.decode('utf-8').strip()
                    if decoded_str.startswith('{'):
                        json_sources.append(decoded_str)
                except (binascii.Error, UnicodeDecodeError):
                    pass

            for source in json_sources:
                try:
                    credentials_info = json.loads(source)
                    return Credentials.from_service_account_info(credentials_info, scopes=scopes)
                except (json.JSONDecodeError, ValueError):
                    continue

            return Credentials.from_service_account_file(credentials_value, scopes=scopes)
        except FileNotFoundError:
            logger.warning("⚠️ Google credentials file not found; continuing without Sheets access")
        except json.JSONDecodeError:
            logger.warning("⚠️ Invalid Google credentials JSON; continuing without Sheets access")
        except Exception as exc:
            logger.warning(f"⚠️ Unable to load Google credentials: {exc.__class__.__name__}")

        return None

    def load_template(self):
        """Load HTML email template"""
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'email_template.html')
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return Template(f.read())
        except FileNotFoundError:
            logger.error(f"Email template not found at {template_path}")
            raise

    def format_date(self, date_str):
        """Format date string to readable format"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%B %d, %Y')
        except:
            return date_str

    def prepare_email_data(self, daily_analysis, new_dates):
        """Prepare data for email template"""
        # Find latest date
        all_dates = []
        for location_dates in new_dates.values():
            all_dates.extend(location_dates)
        all_dates = sorted(set(all_dates), reverse=True)
        latest_date = all_dates[0] if all_dates else None

        # Collect urgent items (above market)
        urgent_items = []
        winning_products = []
        location_summaries = {}

        for location, location_data in daily_analysis.items():
            location_wins = 0
            location_total = 0

            for date, date_data in location_data.items():
                if date not in all_dates:
                    continue

                for product, product_data in date_data.items():
                    status_code = product_data.get('status_code')

                    if status_code == 'losing' and product_data.get('price_difference', 0) > 300:
                        urgent_items.append({
                            'location': location,
                            'product': product,
                            'difference': product_data['price_difference'],
                            'date': date
                        })

                    if status_code == 'winning':
                        winning_products.append({
                            'location': location,
                            'product': product,
                            'price': product_data['pullus_price'],
                            'date': date
                        })
                        location_wins += 1

                    if status_code != 'no_data':
                        location_total += 1

            if location_total > 0:
                win_rate = (location_wins / location_total * 100)
                location_summaries[location] = {
                    'wins': location_wins,
                    'total': location_total,
                    'win_rate': win_rate,
                    'status_emoji': '🟢' if win_rate >= 50 else '🟡' if win_rate >= 30 else '🔴'
                }

        # Sort urgent items by difference (highest first)
        urgent_items.sort(key=lambda x: x['difference'], reverse=True)

        return {
            'date': self.format_date(latest_date) if latest_date else 'Recent',
            'latest_date': latest_date,
            'all_dates': all_dates,
            'date_count': len(all_dates),
            'is_multiple_dates': len(all_dates) > 1,
            'urgent_items': urgent_items[:5],  # Top 5
            'winning_products': winning_products[:5],  # Top 5
            'location_summaries': location_summaries,
            'sheet_url': f"https://docs.google.com/spreadsheets/d/{os.getenv('GOOGLE_SHEET_ID')}/edit",
            'current_year': datetime.now().year
        }

    def generate_subject(self, email_data):
        """Generate email subject based on data"""
        if email_data['is_multiple_dates']:
            return f"📊 Pullus Market Intel Update - {email_data['date_count']} dates"
        else:
            return f"📊 Pullus Market Intel - {email_data['date']}"

    def send_email(self, daily_analysis, new_dates):
        """Send professional HTML email"""
        try:
            # Prepare data
            email_data = self.prepare_email_data(daily_analysis, new_dates)

            # Generate subject
            subject = self.generate_subject(email_data)

            # Load and render template
            template = self.load_template()
            html_content = template.render(**email_data)

            # Inline CSS for email clients (Gmail, Outlook, etc.)
            html_content = transform(html_content)

            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['From'] = self.email_sender
            message['To'] = self.email_recipient

            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(message)

            logger.info(f"✅ Email sent successfully to {self.email_recipient}")
            logger.info(f"   Subject: {subject}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False
