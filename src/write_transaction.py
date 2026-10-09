import json

transactions=[{
    "transaction_id": "TX12351",
    "amount": 45.00,
    "currency": "CAD"
  },
  {
    "transaction_id": "TX12352",
    "amount": -150.75,
    "currency": "AUD"
  }]
def write_transaction(transactions:list[dict]):
    with open(r"data\generated_transactions.json",'w') as open_file:
        json.dump(transactions,open_file,indent=4)
        print(f"Transactions successfully written to generated_transactions.json")

write_transaction(transactions)