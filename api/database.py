from datetime import datetime, timezone
from sqlalchemy import DateTime, Column, Integer, String, Float,Text, func, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker,Session
DATABASE_URL = "sqlite:///./data/fraud_detection.db"
engine = create_engine(DATABASE_URL,connect_args={"check_same_thread": False},)
SessionLocal = sessionmaker(bind=engine,autoflush=False,autocommit=False,)

class Base(DeclarativeBase):
    pass

class Transaction(Base):
    __tablename__="transaction"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    original_features: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,server_default=func.now(),nullable=False,)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

class ReviewCase(Base):
    __tablename__ = "review_cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="open", nullable=False)
    investigation_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyst_decision: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc),nullable=False,)

def create_tables():
    Base.metadata.create_all(engine)
    columns = {column["name"] for column in inspect(engine).get_columns("transaction")}
    if "anomaly_score" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text('ALTER TABLE "transaction" ADD COLUMN anomaly_score FLOAT')
            )

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
