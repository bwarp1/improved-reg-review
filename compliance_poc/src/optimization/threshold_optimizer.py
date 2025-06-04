"""Module for optimizing threshold values for regulation matching."""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

class OptimizationMetric(NamedTuple):
    """Performance metrics for threshold optimization."""
    precision: float
    recall: float
    f1: float
    support: int

@dataclass
class DomainPerformance:
    """Track performance metrics for a specific domain."""
    correct: int = 0
    incorrect: int = 0
    total: int = 0
    thresholds: List[float] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class OptimizationError(Exception):
    """Raised when threshold optimization fails."""
    pass

class ThresholdOptimizer:
    """
    Optimizes and manages threshold values for regulatory compliance matching.
    
    This class analyzes performance metrics and optimizes threshold values
    based on actual matching results to improve accuracy.
    """
    
    def __init__(self, config: Dict[str, Any],
                 db_manager: Optional['DatabaseManager'] = None):
        """
        Initialize the threshold optimizer.
        
        Args:
            config_path: Path to load/save optimized thresholds
            auto_save: Whether to automatically save optimized thresholds
            metrics_history: Number of data points to keep in history
        """
        """Initialize threshold optimizer with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config.get("optimization", {})
        self.db_manager = db_manager
        
        # Get optimization parameters from config
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)
        self.max_history = self.config.get("metrics_history", 100)
        
        # Initialize thresholds with defaults and confidence levels
        self.thresholds = self.config.get("thresholds", {
            "base": {"value": 0.60, "confidence": self.min_confidence},
            "financial": {"value": 0.70, "confidence": self.min_confidence},
            "healthcare": {"value": 0.65, "confidence": self.min_confidence},
            "data_privacy": {"value": 0.65, "confidence": self.min_confidence},
            "environmental": {"value": 0.65, "confidence": self.min_confidence},
            "general": {"value": 0.55, "confidence": self.min_confidence}
        })
        
        # Initialize tracking
        self.domain_performance: Dict[str, DomainPerformance] = {}
        self.last_optimization: Dict[str, float] = {}
        
    def save_metric(self, metric_data: Dict[str, Any]) -> None:
        """Save a performance metric to the database."""
        if self.db_manager:
            try:
                self.db_manager.add_performance_metric(
                    domain=metric_data["domain"],
                    threshold=metric_data["threshold_used"],
                    score=metric_data["score"],
                    is_correct=metric_data["is_correct"],
                    details=metric_data.get("details")
                )
            except Exception as e:
                self.logger.error(f"Error saving metric to database: {e}")
    
    def save_threshold_change(self, domain: str, old_value: float,
                            new_value: float, metrics: OptimizationMetric) -> None:
        """Save a threshold change event to the database."""
        if self.db_manager:
            try:
                self.db_manager.add_threshold_change(
                    domain=domain,
                    old_value=old_value,
                    new_value=new_value,
                    confidence=metrics.f1,
                    metrics={
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "support": metrics.support
                    }
                )
            except Exception as e:
                self.logger.error(f"Error saving threshold change to database: {e}")
            
    def add_match_result(self, match_result: Dict, is_correct: bool, domain: str = "base",
                        details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a match result to the performance metrics.
        
        Args:
            match_result: Result from the matcher
            is_correct: Whether the match was correct (validated)
            domain: The regulatory domain of the match
        """
        metric_data = {
            "domain": domain,
            "threshold_used": match_result.get("threshold_used", 
                                             self.thresholds[domain]["value"]),
            "score": match_result.get("adjusted_score", 
                                    match_result.get("score", 0.0)),
            "is_correct": is_correct,
            "details": details
        }
        
        # Save metric to database first
        self.save_metric(metric_data)
        
        # Update in-memory performance tracking
        if domain not in self.domain_performance:
            self.domain_performance[domain] = DomainPerformance()
        
        perf = self.domain_performance[domain]
        perf.total += 1
        if is_correct:
            perf.correct += 1
        else:
            perf.incorrect += 1
        
        # Track detailed metrics
        perf.thresholds.append(metric_data["threshold_used"])
        perf.scores.append(metric_data["score"])
        perf.timestamps.append(time.time())
        
        # Maintain history size
        while len(perf.thresholds) > self.max_history:
            perf.thresholds.pop(0)
            perf.scores.pop(0)
            perf.timestamps.pop(0)
        
        # Consider optimization if needed
        if self._should_optimize(domain):
            try:
                self.optimize_thresholds(domains=[domain])
            except OptimizationError as e:
                self.logger.warning(f"Optimization skipped for {domain}: {e}")
            
    def optimize_thresholds(self, min_samples: int = 10, 
                          domains: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Optimize threshold values based on collected metrics.
        
        Args:
            min_samples: Minimum number of samples required for domain optimization
            
        Returns:
            Dictionary of optimized thresholds by domain
        """
        # Determine which domains to optimize
        domains_to_optimize = (domains if domains is not None 
                             else list(self.domain_performance.keys()))
        
        optimized = False
        for domain in domains_to_optimize:
            if domain not in self.domain_performance:
                continue
                
            perf = self.domain_performance[domain]
            if len(perf.scores) < min_samples:
                continue
            
            try:
                # Find optimal threshold and evaluate performance
                best_threshold, metrics = self._find_optimal_threshold(domain)
                
                # Update threshold if significant improvement
                current = self.thresholds[domain]["value"]
                if self._is_significant_improvement(current, best_threshold, metrics):
                    # Apply gradual adaptation
                    new_threshold = (current * (1 - self.adaptation_rate) + 
                                  best_threshold * self.adaptation_rate)
                    
                    self.thresholds[domain]["value"] = new_threshold
                    self.thresholds[domain]["confidence"] = metrics.f1
                    
                    self.last_optimization[domain] = time.time()
                    optimized = True
                    
                    # Save threshold change to database
                    self.save_threshold_change(domain, current, new_threshold, metrics)
                    
                    self.logger.info(
                        f"Optimized {domain} threshold: {current:.3f} -> {new_threshold:.3f} "
                        f"(F1: {metrics.f1:.3f}, Support: {metrics.support})"
                    )
                    
            except OptimizationError as e:
                self.logger.warning(f"Optimization failed for {domain}: {e}")
        
        if optimized and self.auto_save:
            self.save_thresholds()
        
        return {d: t["value"] for d, t in self.thresholds.items()}
    
    def _find_optimal_threshold(self, metrics: List[Dict]) -> Tuple[float, float]:
        """
        Find the optimal threshold that maximizes F1 score.
        
        Args:
            metrics: List of match metrics for a domain
            
        Returns:
            Tuple of (best_threshold, best_f1_score)
        """
        # Get unique scores to test as thresholds
        all_scores = sorted(set(m["score"] for m in metrics))
        if not all_scores:
            return 0.6, 0.0  # Default values if no scores
            
        # Add some values between min and max for more granular testing
        min_score = min(all_scores)
        max_score = max(all_scores)
        test_thresholds = np.linspace(min_score, max_score, 20).tolist()
        test_thresholds = sorted(set(all_scores + test_thresholds))
        
        # Evaluate each threshold
        best_threshold = 0.6  # Default
        best_f1 = 0.0
        
        for threshold in test_thresholds:
            # Calculate precision, recall, F1
            tp = sum(1 for m in metrics if m["score"] >= threshold and m["is_correct"])
            fp = sum(1 for m in metrics if m["score"] >= threshold and not m["is_correct"])
            fn = sum(1 for m in metrics if m["score"] < threshold and m["is_correct"])
            
            # Handle edge cases
            if tp + fp == 0 or tp + fn == 0:
                continue
                
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall == 0:
                continue
                
            f1 = 2 * (precision * recall) / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold, best_f1
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics across all domains.
        
        Returns:
            Dictionary with performance metrics
        """
        # Prepare comprehensive performance report
        metrics = {
            "overall": {
                "correct": 0,
                "incorrect": 0,
                "total": 0,
                "accuracy": 0.0
            },
            "domains": {},
            "optimization_history": self.optimization_history,
            "current_thresholds": {
                domain: {
                    "value": data["value"],
                    "confidence": data["confidence"],
                    "last_optimized": self.last_optimization.get(domain, 0)
                }
                for domain, data in self.thresholds.items()
            }
        }
        
        # Calculate metrics for each domain
        for domain, perf in self.domain_performance.items():
            if perf.total == 0:
                continue
                
            accuracy = perf.correct / perf.total
            metrics["domains"][domain] = {
                "correct": perf.correct,
                "incorrect": perf.incorrect,
                "total": perf.total,
                "accuracy": accuracy,
                "recent_scores": {
                    "mean": np.mean(perf.scores[-10:]) if perf.scores else 0,
                    "std": np.std(perf.scores[-10:]) if perf.scores else 0
                }
            }
            
            # Update overall metrics
            metrics["overall"]["correct"] += perf.correct
            metrics["overall"]["incorrect"] += perf.incorrect
            metrics["overall"]["total"] += perf.total
        
        # Calculate overall accuracy
        if metrics["overall"]["total"] > 0:
            metrics["overall"]["accuracy"] = (
                metrics["overall"]["correct"] / metrics["overall"]["total"]
            )
        
        return metrics
    
    def _should_optimize(self, domain: str) -> bool:
        """Determine if optimization should be run for a domain."""
        if domain not in self.domain_performance:
            return False
            
        perf = self.domain_performance[domain]
        
        # Check if we have enough new data
        if len(perf.scores) < 10:
            return False
            
        # Check time since last optimization
        last_opt = self.last_optimization.get(domain, 0)
        if time.time() - last_opt < 3600:  # Don't optimize more than once per hour
            return False
            
        # Check if recent performance suggests need for optimization
        recent_correct = sum(1 for i in range(-10, 0)
                           if i < len(perf.scores) and
                           perf.scores[i] >= self.thresholds[domain]["value"])
        
        return recent_correct <= 7  # Optimize if accuracy drops below 70%
    
    def _is_significant_improvement(self, current: float, new: float, 
                                  metrics: OptimizationMetric) -> bool:
        """Determine if a threshold change represents significant improvement."""
        # Require higher confidence for larger changes
        min_improvement = abs(new - current) * 2
        
        # Check if F1 score improvement is significant
        current_f1 = self.thresholds.get(
            "base", {"confidence": self.min_confidence}
        )["confidence"]
        
        return (metrics.f1 - current_f1) > min_improvement
