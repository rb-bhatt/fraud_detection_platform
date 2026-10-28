def recommend_action(
    fraud_probability: float,
    decision: str,
    investigation_summary: str,
) -> dict:
    if decision == "Approved":
        return{
            "action":"Monitor",
            "requires_human_approval":False
        }
    elif decision == "Review":
        return{
            "action":"Review",
            "requires_human_approval":True
        }
    else:
        return{
            "action":"escalate_for_manual_approval",
            "requires_human_approval":True
        }