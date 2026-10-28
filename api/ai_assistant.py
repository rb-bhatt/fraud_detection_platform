from api.llm_provider import generate_investigation_summary

def build_investigation_summary(
        fraud_probability:float,
        decision:str,
        explanation:list[dict],
        transaction:dict
) -> str:
    
    amount = transaction.get("amount", 0)
    transaction_id = transaction.get("transaction_id", "Unknown")

    summary = (
        f"Transaction ID: {transaction_id}\n"
        f"Transaction Amount: ${amount}\n"
        f"Fraud Probability: {fraud_probability}\n"
        f"Decision: {decision}\n\n"
        "Key Contributing Features:\n"
    )

    for item in explanation:
        feature = item.get("feature", "Unknown")
        contribution = item.get("contribution", 0)
        summary += f"- {feature}: contribution {contribution}\n"

    if decision == "Block":
        summary += "\nInvestigation: High-risk transaction. Blocking is recommended."
    elif decision == "Review":
        summary += "\nInvestigation: Potentially suspicious transaction. Manual review is recommended."
    else:
        summary += "\nInvestigation: Transaction is considered low risk."
    summary += (
        "\n\nHuman approval required: A human analyst must review "
        "and approve the final action before it is executed."
    )
    prompt = (
        "You are a fraud investigation assistant. "
        "Prepare a concise, professional investigation report using "
        "the following transaction details.\n\n"
        f"{summary}\n\n"
        "Explain the risk indicators clearly. Do not invent facts. "
        "The final action requires human analyst approval."
    )
    return generate_investigation_summary(prompt)
if __name__ == "__main__":
    fraud_probability = 0.92
    decision = "Block"

    explanation = [
        {"feature": "V14", "contribution": 0.75},
        {"feature": "V10", "contribution": 0.42},
        {"feature": "V4", "contribution": 0.28},
    ]

    transaction = {
        "transaction_id": "txn-12345",
        "amount": 2500.00,
        "created_at": "2026-10-27 10:30:00",
    }

    summary = build_investigation_summary(
        fraud_probability=fraud_probability,
        decision=decision,
        explanation=explanation,
        transaction=transaction,
    )

    print("Investigation Summary:")
    print(summary)
