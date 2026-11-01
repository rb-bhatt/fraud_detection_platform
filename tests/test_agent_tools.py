
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from fastapi.testclient import TestClient
from api.agent_tools import check_transaction_risk, create_review_case
from api.database import Base, Transaction, ReviewCase


# Test database setup
@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )

    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


# Test check_transaction_risk()
@pytest.mark.parametrize(
    "fraud_probability, expected_risk",
    [
        (0.2, "Low"),
        (0.7, "Medium"),
        (0.9, "High"),
    ]
)
def test_check_transaction_risk(fraud_probability, expected_risk):

    class MockTransaction:
        amount = 1000
        decision = "approved"

        def __init__(self, probability):
            self.fraud_probability = probability

    transaction = MockTransaction(fraud_probability)

    result = check_transaction_risk(transaction)

    assert result["fraud_probability"] == fraud_probability
    assert result["risk_level"] == expected_risk
    assert result["amount"] == 1000
    assert result["decision"] == "approved"


# Test create_review_case()
def test_create_review_case(db_session):

    transaction_id = "TXN12345"

    review_case = create_review_case(db_session, transaction_id)

    assert review_case is not None
    assert review_case.transaction_id == transaction_id
    assert review_case.status == "open"

    # Verify that the case was saved in the database
    saved_case = db_session.query(ReviewCase).filter(
        ReviewCase.transaction_id == transaction_id
    ).first()

    assert saved_case is not None
    assert saved_case.status == "open"
