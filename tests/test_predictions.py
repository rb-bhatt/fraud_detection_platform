import pytest
from src.predict_transaction import decide_transaction

def test_decision_thresholds():
    assert decide_transaction(0.0) == "Approved"
    assert decide_transaction(0.10) == "Approved"
    assert decide_transaction(0.20) == "Approved"
    assert decide_transaction(0.50) == "Approved"
    assert decide_transaction(0.60) == "Review"
    assert decide_transaction(0.79) == "Review"
    assert decide_transaction(0.80) == "Block"
    assert decide_transaction(0.90) == "Block"
    
    with pytest.raises(ValueError):
        decide_transaction(-0.10)
        
    with pytest.raises(ValueError):
        decide_transaction(1.10)