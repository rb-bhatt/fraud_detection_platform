from src.transaction_validator import validate_transaction

def test_valid_transaction(): 
    transactions={"transaction_id": "TX12351", "amount": 45.0, "currency": "CAD"}
    is_valid,validation_errors=validate_transaction(transactions)
    assert is_valid==True
    assert validation_errors==[]
    
def test_negative_amount():
    transactions={"transaction_id": "TX12352", "amount": -150.75, "currency": "AUD"}
    is_valid,validation_errors=validate_transaction(transactions)
    assert is_valid is False
    assert validation_errors !=[]

def test_lowercase_currency():
    transactions={"transaction_id": "TX12353", "amount": 45.00, "currency": "cad"}
    is_valid,validation_errors=validate_transaction(transactions)
    assert is_valid is False
    assert validation_errors !=[]

def test_Empty_TransactionID():
    transactions={"transaction_id": "", "amount": 45.00, "currency": "CAD"}
    is_valid,validation_errors=validate_transaction(transactions)
    assert is_valid is False
    assert validation_errors !=[]
