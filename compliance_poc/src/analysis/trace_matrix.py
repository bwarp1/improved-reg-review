"""Module for generating and analyzing requirement-to-policy trace matrices."""

import logging
from typing import Dict, List, Set, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class TraceLink:
    """Represents a link between a requirement and policy."""
    requirement_id: str
    policy_id: str
    confidence: float
    match_type: str
    evidence: List[str]

class TraceMatrix:
    """
    Manages traceability between regulatory requirements and policies.
    Implements advanced matching and analysis capabilities.
    """
    
    def __init__(self):
        """Initialize the trace matrix."""
        self.logger = logging.getLogger(__name__)
        self.links: List[TraceLink] = []
        self.requirements: Dict[str, Dict] = {}
        self.policies: Dict[str, Dict] = {}
        
        # Track section-level relationships
        self.section_links: Dict[str, Set[str]] = {}
        
        # Confidence thresholds for different match types
        self.confidence_thresholds = {
            "exact": 0.9,
            "strong": 0.75,
            "moderate": 0.5,
            "weak": 0.25
        }
    
    def add_requirement(self, req_id: str, requirement: Dict) -> None:
        """Add a requirement to the matrix."""
        self.requirements[req_id] = requirement
        
        # Track section relationships
        section = requirement.get("section", "")
        if section:
            if section not in self.section_links:
                self.section_links[section] = set()
    
    def add_policy(self, policy_id: str, policy: Dict) -> None:
        """Add a policy to the matrix."""
        self.policies[policy_id] = policy
        
        # Track section relationships
        section = policy.get("section", "")
        if section:
            if section not in self.section_links:
                self.section_links[section] = set()
    
    def add_trace_link(self, link: TraceLink) -> None:
        """Add a trace link between requirement and policy."""
        self.links.append(link)
        
        # Update section relationships if applicable
        req = self.requirements.get(link.requirement_id, {})
        policy = self.policies.get(link.policy_id, {})
        
        req_section = req.get("section", "")
        policy_section = policy.get("section", "")
        
        if req_section and policy_section:
            self.section_links[req_section].add(policy_section)
    
    def get_requirement_coverage(self, requirement_id: str) -> Dict:
        """Get coverage analysis for a specific requirement."""
        req_links = [l for l in self.links if l.requirement_id == requirement_id]
        
        if not req_links:
            return {
                "status": "uncovered",
                "confidence": 0.0,
                "matching_policies": []
            }
        
        # Get highest confidence match
        best_match = max(req_links, key=lambda x: x.confidence)
        
        # Determine coverage status
        if best_match.confidence >= self.confidence_thresholds["strong"]:
            status = "fully_covered"
        elif best_match.confidence >= self.confidence_thresholds["moderate"]:
            status = "partially_covered"
        else:
            status = "weakly_covered"
            
        return {
            "status": status,
            "confidence": best_match.confidence,
            "matching_policies": [
                {
                    "policy_id": l.policy_id,
                    "confidence": l.confidence,
                    "match_type": l.match_type,
                    "evidence": l.evidence
                }
                for l in req_links
            ]
        }
    
    def get_policy_coverage(self, policy_id: str) -> Dict:
        """Get coverage analysis for a specific policy."""
        policy_links = [l for l in self.links if l.policy_id == policy_id]
        
        covered_reqs = [
            {
                "requirement_id": l.requirement_id,
                "confidence": l.confidence,
                "match_type": l.match_type,
                "evidence": l.evidence
            }
            for l in policy_links
        ]
        
        return {
            "num_requirements": len(covered_reqs),
            "average_confidence": np.mean([l.confidence for l in policy_links]) if policy_links else 0,
            "covered_requirements": covered_reqs
        }
    
    def analyze_coverage(self) -> Dict:
        """Perform comprehensive coverage analysis."""
        total_reqs = len(self.requirements)
        covered_reqs = len(set(l.requirement_id for l in self.links))
        
        analysis = {
            "total_requirements": total_reqs,
            "covered_requirements": covered_reqs,
            "coverage_percentage": (covered_reqs / total_reqs * 100) if total_reqs > 0 else 0,
            "coverage_by_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "uncovered_requirements": [],
            "section_coverage": self._analyze_section_coverage()
        }
        
        # Analyze coverage by confidence level
        for req_id in self.requirements:
            coverage = self.get_requirement_coverage(req_id)
            if coverage["status"] == "uncovered":
                analysis["uncovered_requirements"].append(req_id)
            elif coverage["confidence"] >= self.confidence_thresholds["strong"]:
                analysis["coverage_by_confidence"]["high"] += 1
            elif coverage["confidence"] >= self.confidence_thresholds["moderate"]:
                analysis["coverage_by_confidence"]["medium"] += 1
            else:
                analysis["coverage_by_confidence"]["low"] += 1
        
        return analysis
    
    def _analyze_section_coverage(self) -> Dict:
        """Analyze coverage at the section level."""
        section_coverage = {}
        
        for section, linked_sections in self.section_links.items():
            section_coverage[section] = {
                "linked_sections": list(linked_sections),
                "num_links": len(linked_sections),
                "requirements": len([r for r in self.requirements.values() 
                                  if r.get("section") == section]),
                "policies": len([p for p in self.policies.values() 
                               if p.get("section") == section])
            }
        
        return section_coverage
    
    def suggest_improvements(self) -> List[Dict]:
        """Suggest improvements for better coverage."""
        suggestions = []
        
        # Check for uncovered requirements
        uncovered = [req_id for req_id in self.requirements 
                    if not any(l.requirement_id == req_id for l in self.links)]
        
        if uncovered:
            suggestions.append({
                "type": "coverage_gap",
                "severity": "high",
                "description": f"Found {len(uncovered)} uncovered requirements",
                "affected_items": uncovered,
                "recommendation": "Create new policies to address these requirements"
            })
        
        # Check for weak matches
        weak_matches = [
            l for l in self.links 
            if l.confidence < self.confidence_thresholds["moderate"]
        ]
        
        if weak_matches:
            suggestions.append({
                "type": "weak_coverage",
                "severity": "medium",
                "description": f"Found {len(weak_matches)} weak requirement matches",
                "affected_items": [(l.requirement_id, l.policy_id) for l in weak_matches],
                "recommendation": "Review and strengthen these policy matches"
            })
        
        # Check section coverage
        section_coverage = self._analyze_section_coverage()
        poorly_covered_sections = [
            section for section, data in section_coverage.items()
            if data["num_links"] < data["requirements"] * 0.5  # Less than 50% coverage
        ]
        
        if poorly_covered_sections:
            suggestions.append({
                "type": "section_coverage",
                "severity": "medium",
                "description": f"Found {len(poorly_covered_sections)} poorly covered sections",
                "affected_items": poorly_covered_sections,
                "recommendation": "Review policy coverage for these sections"
            })
        
        return suggestions
