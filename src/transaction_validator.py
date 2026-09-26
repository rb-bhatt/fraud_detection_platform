def validate_transaction(transaction: dict) -> tuple[bool, list[str]] :
    errors:list[str]=[]

    transaction_id= transaction.get("transaction_id")    
    if not isinstance(transaction_id,str) or not transaction_id.strip():
        errors.append("Invalid transaction_id: must be a non-empty string.")

    amount=transaction.get("amount")
    if not isinstance(amount,(int,float)) or isinstance(amount,bool):
        errors.append("Invalid amount: must be a number.")
    elif amount<=0:
        errors.append("Invalid amount: must be greater than zero.")

    currency=transaction.get("currency")
    if not isinstance(currency,str) or len(currency)!=3:
        errors.append("Invalid currency: must be a 3-letter string.")
    elif not currency.isupper():
        errors.append("Invalid currency: must be uppercase letter.")

    return len(errors) == 0, errors

if __name__ == '__main__':
    sample_transaction=[{
        "transaction_id": "TX12345",
        "amount": 100.50,
        "currency": "USD"},
        {"transaction_id": "",
         "amount": -25,
         "currency": "usd",}]
    for transaction in sample_transaction:
        is_valid, validation_errors = validate_transaction(transaction)
        print(f"Transaction: {transaction}")
        print(f"Is transaction valid: {is_valid}")
        print(f"Validation errors: {validation_errors}")