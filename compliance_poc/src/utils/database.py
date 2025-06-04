"""Database module for storing historical performance metrics."""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Dict, Any, Optional
import json

Base = declarative_base()

class PerformanceMetric(Base):
    """Model for storing individual performance metrics."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    domain = Column(String, index=True)
    threshold_used = Column(Float)
    score = Column(Float)
    is_correct = Column(Integer)  # 0 or 1
    match_details = Column(JSON)
    
class ThresholdHistory(Base):
    """Model for tracking threshold changes over time."""
    __tablename__ = 'threshold_history'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    domain = Column(String, index=True)
    old_value = Column(Float)
    new_value = Column(Float)
    confidence = Column(Float)
    optimization_metrics = Column(JSON)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, connection_string: str = "sqlite:///compliance_poc/data/metrics.db"):
        """Initialize database connection."""
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    def add_performance_metric(self, domain: str, threshold: float, 
                             score: float, is_correct: bool,
                             details: Optional[Dict[str, Any]] = None) -> None:
        """Add a new performance metric to the database."""
        session = self.Session()
        try:
            metric = PerformanceMetric(
                domain=domain,
                threshold_used=threshold,
                score=score,
                is_correct=1 if is_correct else 0,
                match_details=json.dumps(details) if details else None
            )
            session.add(metric)
            session.commit()
        finally:
            session.close()
    
    def add_threshold_change(self, domain: str, old_value: float,
                           new_value: float, confidence: float,
                           metrics: Dict[str, Any]) -> None:
        """Record a threshold change event."""
        session = self.Session()
        try:
            change = ThresholdHistory(
                domain=domain,
                old_value=old_value,
                new_value=new_value,
                confidence=confidence,
                optimization_metrics=json.dumps(metrics)
            )
            session.add(change)
            session.commit()
        finally:
            session.close()
    
    def get_domain_performance(self, domain: str, 
                             limit: int = 100) -> list[PerformanceMetric]:
        """Get recent performance metrics for a domain."""
        session = self.Session()
        try:
            return session.query(PerformanceMetric)\
                .filter_by(domain=domain)\
                .order_by(PerformanceMetric.timestamp.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    def get_threshold_history(self, domain: str, 
                            limit: int = 100) -> list[ThresholdHistory]:
        """Get threshold change history for a domain."""
        session = self.Session()
        try:
            return session.query(ThresholdHistory)\
                .filter_by(domain=domain)\
                .order_by(ThresholdHistory.timestamp.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    def get_performance_summary(self, domain: str) -> Dict[str, Any]:
        """Get summarized performance metrics for a domain."""
        session = self.Session()
        try:
            metrics = session.query(PerformanceMetric)\
                .filter_by(domain=domain)\
                .order_by(PerformanceMetric.timestamp.desc())\
                .limit(1000)\
                .all()
            
            if not metrics:
                return {
                    "total_matches": 0,
                    "accuracy": 0,
                    "average_score": 0
                }
            
            total = len(metrics)
            correct = sum(1 for m in metrics if m.is_correct)
            avg_score = sum(m.score for m in metrics) / total
            
            return {
                "total_matches": total,
                "accuracy": correct / total,
                "average_score": avg_score,
                "recent_threshold": metrics[0].threshold_used
            }
        finally:
            session.close()
