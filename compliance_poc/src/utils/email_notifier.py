import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any

from compliance_poc.src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

class EmailNotifier:
    """Sends email notifications about regulation changes and compliance issues."""
    
    def __init__(self):
        config = load_config()
        self.smtp_config = config.get('notifications', {})
        
    def send_notification(
        self, 
        subject: str, 
        body: str, 
        recipients: List[str],
        attachments: Optional[List[Dict[str, Any]]] = None,
        html: bool = True
    ) -> bool:
        """Send an email notification"""
        if not self._can_send_email(recipients):
            return False
            
        try:
            msg = self._create_message(subject, body, recipients, attachments, html)
            return self._send_message(msg)
        except Exception as e:
            logger.error("Failed to send email notification: %s", str(e))
            return False
            
    def _can_send_email(self, recipients: List[str]) -> bool:
        """Check if email can be sent"""
        if not self.smtp_config.get("enabled"):
            logger.info("Email notifications are disabled. Would have sent email to: %s", 
                       ", ".join(recipients))
            return False
        if not self.smtp_config.get("smtp_server") or not recipients:
            logger.warning("Missing SMTP server configuration or recipients")
            return False
        return True
        
    def _create_message(self, subject: str, body: str, recipients: List[str],
                       attachments: Optional[List[Dict[str, Any]]], html: bool) -> MIMEMultipart:
        """Create email message with attachments"""
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.smtp_config["sender_email"]
        msg['To'] = ", ".join(recipients)
        
        msg.attach(MIMEText(body, 'html' if html else 'plain'))
        
        if attachments:
            for attachment in attachments:
                part = MIMEText(attachment['content'])
                part.add_header('Content-Disposition', 
                               f'attachment; filename="{attachment["filename"]}"')
                msg.attach(part)
        
        return msg
        
    def _send_message(self, msg: MIMEMultipart) -> bool:
        """Send email message via SMTP"""
        with smtplib.SMTP(self.smtp_config["smtp_server"], 
                         self.smtp_config["smtp_port"]) as server:
            if self.smtp_config.get("smtp_username") and self.smtp_config.get("smtp_password"):
                server.starttls()
                server.login(self.smtp_config["smtp_username"], 
                           self.smtp_config["smtp_password"])
            server.send_message(msg)
            logger.info("Email notification sent to %s", msg['To'])
            return True
