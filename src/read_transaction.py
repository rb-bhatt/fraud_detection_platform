import json
from transaction_validator import validate_transaction

JSON_FILE=r"data\generated_transactions.json"
try:
    with open(JSON_FILE,'r') as open_file:
        transactions=json.load(open_file)
        for transaction in transactions:
            is_valid,validation_errors=validate_transaction(transaction)
            if is_valid:
                print(f"Transaction: {transaction}")
                print(f"Is transaction valid: {is_valid}")
                print(f"")
            else:
                print(f"Transaction: {transaction}")
                print(f"Is transaction valid: {is_valid}")
                print(f"Validation errors: {validation_errors}")
except FileNotFoundError as e:
    print(f"Error: No file was found!! : {e}")
except json.JSONDecodeError as e:
    print(f" Error: decoding JSON from the file!! : {e}")
except Exception as e:
    print(f"Error: An unexpected error occurred while processing transactions: {e}")