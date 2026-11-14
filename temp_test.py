from unittest.mock import patch

from api.ai_assistant import build_investigation_summary


def test_summary_fallback_when_gemini_fails():
    with patch(
        "api.ai_assistant.generate_investigation_summary",
        side_effect=TimeoutError("Gemini timed out"),
    ):
        result = build_investigation_summary(
            fraud_probability=0.92,
            decision="Block",
            explanation=[
                {"feature": "V14", "contribution": 0.75}
            ],
            transaction={"Amount": 2500},
        )

    assert isinstance(result, str)
    assert "Transaction Amount: $2500" in result
    assert "temporarily unavailable" in result
