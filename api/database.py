from datetime import datetime, timezone
from sqlalchemy import DateTime, Column, Integer, String, Float,Text, func
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
    decision: Mapped[str] = mapped_column(String, nullable=False)
    original_features: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,server_default=func.now(),nullable=False,)

class ReviewCase(Base):
    __tablename__ = "review_cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="open", nullable=False)
    analyst_decision: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc),nullable=False,)

def create_tables():
    Base.metadata.create_all(engine)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()
