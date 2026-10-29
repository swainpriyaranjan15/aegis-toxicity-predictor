import re
from typing import Dict, Any, Tuple, List

class PrivacySanitizer:
    """
    Privacy-Aware Preprocessing Engine.
    Detects and redacts sensitive personally identifiable information (PII)
    including phone numbers, emails, personal names, and physical/IP addresses
    before comments are passed to neural toxicity classifiers.
    """

    def __init__(self):
        # Email Regex (Standard RFC compliant pattern)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            re.IGNORECASE
        )

        # Phone Number Regex (Supports international codes, formats with parens, dashes, dots, spaces, 10-12 digits)
        self.phone_pattern = re.compile(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|'
            r'(\+?91[-.\s]?)?[6-9]\d{9}\b|'
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
            re.IGNORECASE
        )

        # IP Address Regex (IPv4)
        self.ip_pattern = re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        )

        # Street & Postal Address Pattern (US / UK / Commonwealth Street identifiers and PO Boxes)
        self.address_pattern = re.compile(
            r'\b(?:\d{1,5}\s+(?:[A-Za-z0-9.]+\s+){1,4}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Circle|Cir|Highway|Hwy|Apartment|Apt|Suite|Ste|Terrace|Ter|Place|Pl|Parkway|Pkwy|Square|Sq|Trail|Trl)\b'
            r'|P\.?O\.?\s+Box\s+\d{1,6})\b',
            re.IGNORECASE
        )

        # Social Media Handles (@username)
        self.handle_pattern = re.compile(
            r'@\w{2,30}\b'
        )

        # Person Honorifics / Titles followed by Capitalized Names
        self.honorific_name_pattern = re.compile(
            r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Officer|Detective|Senator|Mayor|President|Gov\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        )

        # Common personal name prefixes / self-identification ("my name is [Name]", "contact [Name]")
        self.named_intro_pattern = re.compile(
            r'\b(?:my name is|call me|i am|this is|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            re.IGNORECASE
        )

    def redact(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Scans and redacts PII entities in the input text with standardized placeholders:
        - [EMAIL]
        - [PHONE]
        - [ADDRESS]
        - [PERSON]

        Returns:
            sanitized_text (str): Comment text with PII safely replaced.
            metadata (Dict[str, Any]): Telemetry of detected redactions without exposing original PII.
        """
        if not text:
            return text, {
                "pii_detected": False,
                "counts": {"email": 0, "phone": 0, "address": 0, "person": 0},
                "total_redactions": 0,
                "types_detected": []
            }

        sanitized = text
        counts = {
            "email": 0,
            "phone": 0,
            "address": 0,
            "person": 0
        }

        # 1. Redact Emails
        def email_repl(match):
            counts["email"] += 1
            return "[EMAIL]"
        sanitized = self.email_pattern.sub(email_repl, sanitized)

        # 2. Redact Phone Numbers
        def phone_repl(match):
            counts["phone"] += 1
            return "[PHONE]"
        sanitized = self.phone_pattern.sub(phone_repl, sanitized)

        # 3. Redact IP Addresses
        def ip_repl(match):
            counts["address"] += 1
            return "[ADDRESS]"
        sanitized = self.ip_pattern.sub(ip_repl, sanitized)

        # 4. Redact Physical / Postal Addresses
        def address_repl(match):
            counts["address"] += 1
            return "[ADDRESS]"
        sanitized = self.address_pattern.sub(address_repl, sanitized)

        # 5. Redact Social Usernames (@user) -> [PERSON]
        def handle_repl(match):
            counts["person"] += 1
            return "[PERSON]"
        sanitized = self.handle_pattern.sub(handle_repl, sanitized)

        # 6. Redact Honorific Names ("Mr. Smith", "Dr. Jane Doe")
        def honorific_repl(match):
            counts["person"] += 1
            return "[PERSON]"
        sanitized = self.honorific_name_pattern.sub(honorific_repl, sanitized)

        # 7. Redact Direct Introductions ("my name is John", "contact Alice")
        def intro_repl(match):
            counts["person"] += 1
            # Preserve prefix, replace name
            prefix = match.group(0).split()[0]
            return f"{prefix} [PERSON]"
        sanitized = self.named_intro_pattern.sub(intro_repl, sanitized)

        total_redactions = sum(counts.values())
        pii_detected = total_redactions > 0
        types_detected = [k for k, v in counts.items() if v > 0]

        metadata = {
            "pii_detected": pii_detected,
            "counts": counts,
            "total_redactions": total_redactions,
            "types_detected": types_detected
        }

        return sanitized, metadata

# Singleton instance helper
_sanitizer_instance = None

def get_sanitizer() -> PrivacySanitizer:
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = PrivacySanitizer()
    return _sanitizer_instance
