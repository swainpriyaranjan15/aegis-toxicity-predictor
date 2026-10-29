import pytest
from src.privacy import get_sanitizer, PrivacySanitizer
from src.predict import get_predictor

@pytest.fixture(scope="module")
def sanitizer():
    return get_sanitizer()

@pytest.fixture(scope="module")
def predictor():
    return get_predictor()

def test_email_redaction(sanitizer):
    text = "Send your hateful message to victim@example.org right now!"
    sanitized, meta = sanitizer.redact(text)
    assert "[EMAIL]" in sanitized
    assert "victim@example.org" not in sanitized
    assert meta["pii_detected"] is True
    assert meta["counts"]["email"] == 1

def test_phone_redaction(sanitizer):
    text = "Call and harass this number: +1-555-839-2019 or +91 9876543210!"
    sanitized, meta = sanitizer.redact(text)
    assert "[PHONE]" in sanitized
    assert "555-839-2019" not in sanitized
    assert "9876543210" not in sanitized
    assert meta["pii_detected"] is True
    assert meta["counts"]["phone"] >= 1

def test_address_and_ip_redaction(sanitizer):
    text = "Doxxing target: lives at 742 Evergreen Terrace, Springfield and IP is 192.168.1.45."
    sanitized, meta = sanitizer.redact(text)
    assert "[ADDRESS]" in sanitized
    assert "192.168.1.45" not in sanitized
    assert "742 Evergreen Terrace" not in sanitized
    assert meta["pii_detected"] is True
    assert meta["counts"]["address"] >= 1

def test_person_redaction(sanitizer):
    text = "Go attack Mr. John Doe and spam @target_user today!"
    sanitized, meta = sanitizer.redact(text)
    assert "[PERSON]" in sanitized
    assert "@target_user" not in sanitized
    assert "John Doe" not in sanitized
    assert meta["pii_detected"] is True
    assert meta["counts"]["person"] >= 1

def test_benign_text_no_false_positive(sanitizer):
    text = "This is a clean, helpful comment about machine learning algorithms."
    sanitized, meta = sanitizer.redact(text)
    assert sanitized == text
    assert meta["pii_detected"] is False
    assert meta["total_redactions"] == 0

def test_model_preserves_accuracy_with_pii(predictor):
    # Toxic doxxing comment containing multiple PII fields
    toxic_dox = "I hate Dr. Smith, his phone is 555-432-1099 and email is smith@lab.org, you deserve to die!"
    res = predictor.predict(toxic_dox)

    # Prediction must still detect toxicity
    assert res["prediction"] == "Toxic"
    assert res["score_percent"] > 50.0

    # Privacy telemetry must be present
    assert "privacy" in res
    assert res["privacy"]["pii_detected"] is True
    assert res["privacy"]["total_redactions"] >= 2
    assert "[PHONE]" in res["text"]
    assert "[EMAIL]" in res["text"]
    assert "555-432-1099" not in res["text"]
    assert "smith@lab.org" not in res["text"]

def test_benign_with_pii_remains_benign(predictor):
    # Benign constructive comment with contact details
    benign_contact = "Thanks for the great lecture Dr. Watson, reach me at contact@research.edu."
    res = predictor.predict(benign_contact)

    assert res["prediction"] == "Non-toxic"
    assert res["privacy"]["pii_detected"] is True
    assert "[EMAIL]" in res["text"]
