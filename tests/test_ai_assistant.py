from api.ai_assistant import build_investigation_summary

def test_build_investigation_summary():
    summary = build_investigation_summary(
        fraud_probability=0.91,
        decision="Block",
        explanation=[
            {"feature": "V14", "contribution": 0.75},
            {"feature": "V10", "contribution": 0.42},
            {"feature": "V4", "contribution": 0.28},
        ],
        transaction={"amount": 250.0}
    )
    assert "0.91" in summary
    assert "Block" in summary
    assert "250.0" in summary
    assert "Human analyst review" in summary
    