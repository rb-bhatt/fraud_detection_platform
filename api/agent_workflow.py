def recommend_action(
    fraud_probability: float,
    decision: str,
    investigation_summary: str,
) -> dict:
    if decision in {"Approve", "Approved"}:    return{
            "recommended_action":"Monitor",
            "requires_human_approval":False,
            "reason": "Low-risk transaction can continue with monitoring."
        }
    elif decision == "Review":
        return{
            "recommended_action":"human_review",
            "requires_human_approval":True,
            "reason": "Human approval is required before any high-impact action."
        }
    else:
        return{
            "recommended_action":"escalate_for_manual_approval",
            "requires_human_approval":True,
            "reason": "Human approval is required before any high-impact action."
        }