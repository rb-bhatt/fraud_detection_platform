from api.database import Transaction,ReviewCase

def get_transaction_history(db, transaction_id: str):
    return db.query(Transaction).filter(Transaction.transaction_id == transaction_id).all()

def check_transaction_risk(transaction)-> dict:
    amount = transaction.amount
    fraud_probability = transaction.fraud_probability
    decision = transaction.decision
    risk_level = None
    if fraud_probability < 0.60:
        risk_level = "Low"
    elif 0.60 <= fraud_probability < 0.80:
        risk_level = "Medium"
    else:
        risk_level = "High"
    return{
        "amount" : transaction.amount,
        "fraud_probability" : transaction.fraud_probability,
        "decision" : transaction.decision,
        "risk_level" : risk_level
    }

def create_review_case(db,transaction_id:str)->ReviewCase:
    status="open"
    review_case = ReviewCase(transaction_id=transaction_id,status=status)
    db.add(review_case)
    db.commit()
    db.refresh(review_case)
    return review_case
