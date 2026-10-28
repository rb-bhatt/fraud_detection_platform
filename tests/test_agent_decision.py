import pytest
from api.agent_workflow import recommend_action

def test_agent_decision_mapping():
    rec_approved = recommend_action(fraud_probability=0.1, decision="Approve", investigation_summary="Low risk transaction.")
    assert rec_approved["recommended_action"] == "monitor"
    assert rec_approved["requires_human_approval"] is False

    rec_review = recommend_action(fraud_probability=0.65, decision="Review", investigation_summary="Moderate risk features detected.")
    assert rec_review["recommended_action"] == "human_review"
    assert rec_review["requires_human_approval"] is True

    rec_block = recommend_action(fraud_probability=0.9, decision="Block", investigation_summary="High risk anomaly patterns.")
    assert rec_block["recommended_action"] == "escalate_for_manual_approval"
    assert rec_block["requires_human_approval"] is True