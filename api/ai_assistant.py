import logging
from api.llm_provider import generate_investigation_summary

logger = logging.getLogger(__name__)

def build_investigation_summary(
    fraud_probability: float,
    decision: str,
    explanation: list[dict],
    transaction: dict,
) -> str:

    amount = transaction.get(
        "Amount",
        transaction.get("amount", "Not provided"),
    )
    transaction_id = transaction.get("transaction_id") or "Unknown"

    percentage = fraud_probability * 100

    signals = explanation[:3]

    signal_text = "\n".join(
        f"- {item.get('feature', 'Unknown')}: "
        f"{item.get('contribution', 'Not provided')}"
        for item in signals
    ) or "No feature signals available."

    if decision.lower() == "block":
        next_step = (
            "Human analyst review and approval are required "
            "before final action."
        )
    elif decision.lower() == "review":
        next_step = "Human analyst review is required."
    else:
        next_step = "No manual review requested by the model."

    summary = (
        "AI INVESTIGATION SUMMARY\n\n"
        f"Model Decision: {decision}\n"
        f"Fraud Probability: {fraud_probability:.2f} "
        f"({percentage:.2f}%)\n"
        f"Transaction ID: {transaction_id}\n"
        f"Transaction Amount: ${amount}\n\n"
        "Top Model Signals:\n"
        f"{signal_text}\n\n"
        f"Next Step: {next_step}"
    )

    prompt = (
        "Write a concise assessment of this transaction in no more "
        "than 20 words.\n"
        "Use only the supplied information.\n"
        "Never invent facts or dates.\n"
        "Do not guess the meaning of anonymized features.\n"
        "Do not change the model decision or numerical values.\n"
        "Return plain text only, without Markdown or tables.\n\n"
        f"{summary}"
    )

    try:
        assessment = generate_investigation_summary(prompt)

        if not assessment or not assessment.strip():
            raise ValueError("Empty AI assessment")

        return summary + "\n\nAI Assessment:\n" + assessment.strip()

    except Exception:
        logger.exception(
            "AI summary unavailable. Using deterministic fallback."
        )

        return (
            summary
            + "\n\nAI Assessment:\n"
            "AI-generated investigation summary is temporarily "
            "unavailable. Showing model results only."
        )


if __name__ == "__main__":
    result = build_investigation_summary(
        fraud_probability=0.92,
        decision="Block",
        explanation=[
            {"feature": "V14", "contribution": 0.75},
            {"feature": "V10", "contribution": 0.42},
            {"feature": "V4", "contribution": 0.28},
        ],
        transaction={
            "transaction_id": "txn-12345",
            "amount": 2500,
        },
    )

    print(result)
