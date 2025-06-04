"""Compare regulatory obligations against internal policies."""

import logging
import re
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz

# Import ThresholdOptimizer for dynamic threshold management
from compliance_poc.src.optimization.threshold_optimizer import ThresholdOptimizer


from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple

class SimilarityScore(NamedTuple):
    """Represents different components of a similarity score."""
    semantic: float
    keyword: float
    entity: float
    domain: float
    final: float

@dataclass
class MatchResult:
    """Result of a compliance match."""
    obligation_id: str
    policy_id: str
    section_id: str
    section_index: int
    scores: SimilarityScore
    matched_text: str
    status: str
    details: Dict[str, Any]

class LRUCache:
    """LRU cache for similarity computations."""
    
    def __init__(self, maxsize: int = 1000):
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize
    
    def get(self, key: str) -> Optional[float]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: float) -> None:
        """Put value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

@dataclass
class ComplianceComparer:
    """Compare regulatory obligations against internal policies with enhanced context awareness."""
    
    config: Dict[str, Any]
    logger: logging.Logger = field(init=False)
    threshold_optimizer: ThresholdOptimizer = field(init=False)
    thresholds: Dict[str, float] = field(init=False)
    context_weights: Dict[str, float] = field(init=False)
    vectorizer: TfidfVectorizer = field(init=False)
    similarity_cache: LRUCache = field(init=False)
    context_memory: OrderedDict = field(init=False)
    
    def __post_init__(self):
        """Initialize after instance creation."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize threshold optimizer
        threshold_config_path = self.config.get("threshold_config_path")
        self.threshold_optimizer = ThresholdOptimizer(config_path=threshold_config_path)
        self.thresholds = self.threshold_optimizer.thresholds
        
        # Setup context weights with validation
        weights = {
            "keyword_match": self.config.get("weight_keyword", 0.3),
            "semantic_similarity": self.config.get("weight_semantic", 0.4),
            "entity_overlap": self.config.get("weight_entity", 0.2),
            "domain_match": self.config.get("weight_domain", 0.1)
        }
        
        # Validate weights sum to 1.0
        total = sum(weights.values())
        if not 0.99 <= total <= 1.01:  # Allow small floating point imprecision
            self.logger.warning(f"Weights sum to {total}, normalizing...")
            weights = {k: v/total for k, v in weights.items()}
        
        self.context_weights = weights
        
        # Initialize vectorizer with optimized parameters
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2),
            stop_words='english',
            max_features=10000  # Limit vocabulary size
        )
        
        # Initialize caches
        self.similarity_cache = LRUCache(
            maxsize=self.config.get("cache_size", 1000)
        )
        self.context_memory = OrderedDict()

    def compare(self, obligations: List[Dict], policies: Dict[str, str]) -> List[Dict]:
        """Compare obligations against policies with enhanced context awareness."""
        start_time = time.time()
        self.logger.info(f"Starting compliance comparison of {len(obligations)} obligations against {len(policies)} policies")
        
        if not obligations or not policies:
            self.logger.warning("No obligations or no policies provided for comparison")
            return []
            
        # Preprocess policies for efficient matching
        policy_sections = self._split_policies_into_sections(policies)
        
        # Start vectorization timer
        vec_start = time.time()
        
        # Create policy vectors
        policy_texts = list(policy_sections.values())
        policy_keys = list(policy_sections.keys())
        
        try:
            policy_vectors = self.vectorizer.fit_transform(policy_texts)
            
            if self.enable_performance_tracking:
                self.performance_stats["vectorization_time"] = time.time() - vec_start
        except Exception as e:
            self.logger.error(f"Error creating policy vectors: {e}")
            return []
        
        # Auto-calibrate thresholds if enabled and sufficient data exists
        if self.auto_calibrate and len(self.calibration_data) >= 10:
            self._calibrate_thresholds()
        
        # Process obligations in sequence, but with awareness of other obligations
        matches = []
        grouped_obligations = self._group_related_obligations(obligations)
        
        for group_idx, obligation_group in enumerate(grouped_obligations):
            self.logger.debug(f"Processing obligation group {group_idx+1}/{len(grouped_obligations)}")
            
            # Extract the primary obligation from each group
            primary_obligation = obligation_group[0]
            
            # Get domain context from the obligation
            domains = primary_obligation.get("domain", [])
            
            # Determine appropriate threshold based on domain
            threshold = self._get_domain_threshold(primary_obligation)
            
            # Find matches with context-awareness
            obligation_matches = self._find_matches_with_context(
                primary_obligation,
                policy_sections,
                policy_vectors,
                policy_keys,
                obligation_group,  # Pass the whole group for context
                threshold
            )
            
            # Add group information to matches
            for match in obligation_matches:
                match["obligation_group"] = [obl["id"] for obl in obligation_group]
                match["primary_obligation_id"] = primary_obligation["id"]
                
            matches.extend(obligation_matches)
        
        # Add non-compliant entries for obligations without matches
        matched_obligation_ids = set(match["obligation_id"] for match in matches)
        all_obligation_ids = set(obl["id"] for obl in obligations)
        unmatched_ids = all_obligation_ids - matched_obligation_ids
        
        for obl_id in unmatched_ids:
            # Find the obligation
            obligation = next((o for o in obligations if o["id"] == obl_id), None)
            if obligation:
                matches.append(self._create_non_compliant_entry(obligation))
        
        # Performance logging
        if self.enable_performance_tracking:
            total_time = time.time() - start_time
            self.logger.info(f"Comparison completed in {total_time:.2f} seconds")
            self.logger.info(f"Vectorization: {self.performance_stats['vectorization_time']:.2f}s, " +
                           f"Similarity calculations: {self.performance_stats['similarity_calculation_time']:.2f}s")
            self.logger.info(f"Total comparisons: {self.performance_stats['total_comparisons']}, " +
                           f"Cache hits: {self.performance_stats['cache_hits']}")
        
        # Apply continuous optimization if we have reviewed matches
        if self.reviewed_matches and self.adaptive_learning:
            self._apply_adaptive_learning()
            # Clear reviewed matches
            self.reviewed_matches = []
        
        return matches

    def record_match_review(self, obligation_id: str, policy_id: str, is_correct: bool) -> None:
        """
        Record human review feedback for a match to improve future matching.
        
        Args:
            obligation_id: ID of the obligation
            policy_id: ID of the matched policy
            is_correct: Whether the match was correct
        """
        # Find the match in our result history
        for match in self.context_memory:
            if match.get("obligation_id") == obligation_id and match.get("policy_id") == policy_id:
                # Store this review
                self.reviewed_matches.append({
                    "match": match,
                    "is_correct": is_correct
                })
                
                # Add to optimizer
                domain = match.get("domain", ["base"])[0] if isinstance(match.get("domain"), list) else "base"
                self.threshold_optimizer.add_match_result(match, is_correct, domain)
                
                self.logger.info(f"Recorded match review: obligation={obligation_id}, policy={policy_id}, correct={is_correct}")
                break

    def _get_domain_threshold(self, obligation: Dict) -> float:
        """Determine appropriate threshold based on obligation characteristics."""
        # First check explicit domains from domain recognition
        domains = obligation.get("domain", [])
        
        for domain in domains:
            if domain in self.thresholds:
                return self.thresholds[domain]
        
        # Fallback to keyword-based detection
        text = obligation["text"].lower()
        
        if any(word in text for word in ["financial", "monetary", "fiscal", "payment", "capital"]):
            return self.thresholds["financial"]
        elif any(word in text for word in ["health", "medical", "patient", "hipaa"]):
            return self.thresholds["healthcare"]
        elif any(word in text for word in ["data", "privacy", "confidential", "personal", "gdpr"]):
            return self.thresholds["data_privacy"]
        elif any(word in text for word in ["environment", "emission", "waste", "pollution"]):
            return self.thresholds["environmental"]
        
        return self.thresholds["base"]

    def _find_matches_with_context(self, 
                                 obligation: Dict, 
                                 policy_sections: Dict[str, str],
                                 policy_vectors: Any,
                                 policy_keys: List[str],
                                 obligation_group: List[Dict],
                                 threshold: float) -> List[Dict]:
        """Find matches considering both content and context."""
        matches = []
        
        # Extract obligation text and create vector
        obligation_text = obligation["text"]
        
        # Calculate similarity scores
        sim_start = time.time()
        obligation_vector = self.vectorizer.transform([obligation_text])
        similarities = cosine_similarity(obligation_vector, policy_vectors).flatten()
        
        if self.enable_performance_tracking:
            self.performance_stats["similarity_calculation_time"] += (time.time() - sim_start)
            self.performance_stats["total_comparisons"] += 1
        
        # Combine similarity scores with other matching factors
        for idx, policy_key in enumerate(policy_keys):
            base_similarity = similarities[idx]
            policy_text = policy_sections[policy_key]
            
            # Skip very low similarity scores for efficiency
            if base_similarity < threshold * 0.5:
                continue
                
            # Calculate comprehensive matching score with context factors
            adjusted_score = self._calculate_comprehensive_score(
                obligation,
                policy_text,
                base_similarity,
                obligation_group
            )
            
            # If score meets threshold, add as a match
            if adjusted_score >= threshold:
                # Split policy key to get original policy ID
                parts = policy_key.split('_section_')
                original_policy_id = parts[0]
                section_num = int(parts[1]) if len(parts) > 1 else 0
                
                # Create match entry with detailed context information
                matches.append({
                    "obligation_id": obligation["id"],
                    "policy_id": original_policy_id,
                    "section_id": policy_key,
                    "section_index": section_num,
                    "base_score": float(base_similarity),
                    "adjusted_score": float(adjusted_score),
                    "threshold_used": float(threshold),
                    "matched_text": policy_text,
                    "status": self._determine_compliance_status(adjusted_score),
                    "match_details": self._generate_match_details(
                        obligation,
                        policy_text,
                        base_similarity,
                        adjusted_score,
                        threshold
                    )
                })
        
        # Sort matches by adjusted score
        matches.sort(key=lambda x: x["adjusted_score"], reverse=True)
        
        # Store calibration data if we have a ground truth
        if obligation.get("ground_truth"):
            self._add_calibration_data(obligation, matches, threshold)
            
        return matches

    def _calculate_comprehensive_score(self, 
                                     obligation: Dict, 
                                     policy_text: str,
                                     base_similarity: float,
                                     obligation_group: List[Dict]) -> float:
        """Calculate comprehensive matching score with multiple factors."""
        # Start with base semantic similarity
        score_components = {
            "semantic_similarity": base_similarity
        }
        
        # Calculate keyword match score
        score_components["keyword_match"] = self._calculate_keyword_score(obligation, policy_text)
        
        # Calculate entity overlap score (named entities, concepts)
        score_components["entity_overlap"] = self._calculate_entity_overlap(obligation, policy_text)
        
        # Domain-specific matching
        score_components["domain_match"] = self._calculate_domain_match(obligation, policy_text)
        
        # Use group context - check if other obligations in the group match
        group_context_boost = 0
        for related_obligation in obligation_group:
            if related_obligation["id"] != obligation["id"]:
                # Calculate basic similarity for related obligation
                related_score = self._quick_similarity(related_obligation["text"], policy_text)
                if related_score > 0.5:  # If there's a decent match
                    group_context_boost = max(group_context_boost, 0.05)  # Small boost for related matches
        
        # Adjust for strength of the obligation
        strength_factor = {
            "strong": 0.05,  # Boost for strong obligations like "must", "shall"
            "medium": 0.0,   # No change for medium strength
            "weak": -0.05    # Slight penalty for weak obligations like "should"
        }
        strength_adjustment = strength_factor.get(obligation.get("strength", "medium"), 0.0)
        
        # Combine scores using weighting
        weighted_score = (
            score_components["semantic_similarity"] * self.context_weights["semantic_similarity"] +
            score_components["keyword_match"] * self.context_weights["keyword_match"] +
            score_components["entity_overlap"] * self.context_weights["entity_overlap"] +
            score_components["domain_match"] * self.context_weights["domain_match"]
        )
        
        # Enhanced contextual memory boost - learn from past matches
        memory_boost = self._get_contextual_memory_boost(obligation, policy_text)
        
        # Add group context boost and strength adjustment
        weighted_score += group_context_boost + strength_adjustment + memory_boost
        
        return min(1.0, max(0.0, weighted_score))  # Keep between 0 and 1

    def _get_contextual_memory_boost(self, obligation: Dict, policy_text: str) -> float:
        """
        Get a boost based on contextual memory of similar matches.
        
        Args:
            obligation: Current obligation
            policy_text: Policy text to match
            
        Returns:
            Boost value between 0 and 0.1
        """
        if not self.context_memory:
            return 0.0
            
        # Look for similar obligations in our memory
        max_boost = 0.0
        obligation_text = obligation["text"]
        
        for past_match in self.context_memory:
            if "obligation_text" not in past_match:
                continue
                
            past_obligation = past_match["obligation_text"]
            past_policy = past_match["matched_text"]
            
            # If obligation text is similar and policy text is similar,
            # and the match was correct, provide a boost
            obl_similarity = self._quick_similarity(obligation_text, past_obligation)
            policy_similarity = self._quick_similarity(policy_text, past_policy)
            
            if obl_similarity > 0.7 and policy_similarity > 0.6:
                if past_match.get("is_correct", False):
                    # Strong boost for confirmed correct similar matches
                    boost = min(0.1, (obl_similarity + policy_similarity) / 20.0)
                    max_boost = max(max_boost, boost)
                elif past_match.get("is_correct") is False:  # Explicitly False
                    # Negative boost for confirmed incorrect matches
                    boost = -min(0.1, (obl_similarity + policy_similarity) / 20.0)
                    max_boost = min(max_boost, boost)
        
        return max_boost

    def _calculate_keyword_score(self, obligation: Dict, policy_text: str) -> float:
        """Calculate keyword match score with improved weighting for important terms."""
        keywords = obligation.get("keywords", [])
        if not keywords:
            return 0.0
            
        # Calculate matches but weight more important keywords higher
        total_weight = 0
        matched_weight = 0
        
        for keyword in keywords:
            # Determine keyword importance (longer compound terms are more important)
            if ' ' in keyword:  # Multi-word term
                weight = 2.0
            elif len(keyword) > 6:  # Longer single words are more significant
                weight = 1.5
            else:
                weight = 1.0
                
            total_weight += weight
            
            # Check for keyword in policy text
            if re.search(r'\b' + re.escape(keyword) + r'\b', policy_text, re.IGNORECASE):
                matched_weight += weight
        
        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _calculate_entity_overlap(self, obligation: Dict, policy_text: str) -> float:
        """Calculate entity overlap score."""
        # Extract entities from obligation
        entities = obligation.get("entities", {})
        if not entities:
            return 0.0
            
        # Flatten entities into a list
        entity_terms = []
        for entity_type, values in entities.items():
            entity_terms.extend(values)
            
        if not entity_terms:
            return 0.0
            
        # Count matches in policy text
        matches = 0
        for entity in entity_terms:
            if re.search(r'\b' + re.escape(entity) + r'\b', policy_text, re.IGNORECASE):
                matches += 1
                
        return matches / len(entity_terms) if len(entity_terms) > 0 else 0.0

    def _calculate_domain_match(self, obligation: Dict, policy_text: str) -> float:
        """Calculate domain-specific match score."""
        # Get the domains from the obligation
        domains = obligation.get("domain", [])
        if not domains:
            return 0.0
            
        # Get domain features if available
        domain_features = obligation.get("domain_features", {})
        
        # Check for domain-specific terms in policy
        domain_term_matches = 0
        total_domain_terms = 0
        
        # Check for regulatory entities
        reg_entities = domain_features.get("regulatory_entities", [])
        for entity in reg_entities:
            total_domain_terms += 1
            if re.search(r'\b' + re.escape(entity) + r'\b', policy_text, re.IGNORECASE):
                domain_term_matches += 1
                
        # Check for regulatory concepts
        concepts = domain_features.get("regulatory_concepts", [])
        for concept in concepts:
            total_domain_terms += 1
            if re.search(r'\b' + re.escape(concept) + r'\b', policy_text, re.IGNORECASE):
                domain_term_matches += 1
                
        # Check for regulatory laws
        laws = domain_features.get("regulatory_laws", [])
        for law in laws:
            total_domain_terms += 1
            if re.search(r'\b' + re.escape(law) + r'\b', policy_text, re.IGNORECASE):
                domain_term_matches += 1.5  # Higher weight for specific law matches
                
        if total_domain_terms == 0:
            return 0.0
            
        return min(1.0, domain_term_matches / total_domain_terms)

    def _determine_compliance_status(self, score: float) -> str:
        """Determine compliance status based on match score."""
        if score >= 0.8:
            return "Compliant"
        elif score >= 0.5:
            return "Partial"
        else:
            return "Non-Compliant"

    def _quick_similarity(self, text1: str, text2: str) -> float:
        """Calculate a quick similarity score between two texts."""
        # Check cache first
        cache_key = hash(text1) ^ hash(text2)  # Simple XOR-based cache key
        
        if cache_key in self.similarity_cache:
            if self.enable_performance_tracking:
                self.performance_stats["cache_hits"] += 1
            return self.similarity_cache[cache_key]
            
        # Simple similarity based on fuzzy matching - faster than TF-IDF for small texts
        similarity = fuzz.token_sort_ratio(text1, text2) / 100.0
        
        # Store in cache
        self.similarity_cache[cache_key] = similarity
        
        return similarity

    def _split_policies_into_sections(self, policies: Dict[str, str]) -> Dict[str, str]:
        """Split policies into sections for more granular matching."""
        policy_sections = {}
        for policy_id, policy_text in policies.items():
            # Try to find section markers using regular expressions
            section_markers = re.finditer(r'\n\s*(?:Section\s+\d+|(?:\d+\.)+\d*|[A-Z\.\s]{2,50}:)\s*\n', policy_text)
            
            # Convert section markers to indices
            split_indices = [0]  # Always start with beginning of document
            for marker in section_markers:
                split_indices.append(marker.start())
            split_indices.append(len(policy_text))  # End of document
            
            # If no sections found, try another approach with blank lines
            if len(split_indices) <= 2:
                sections = re.split(r'\n\n+', policy_text)
                for i, section in enumerate(sections):
                    if len(section.strip()) > 10:  # Skip very short sections
                        section_id = f"{policy_id}_section_{i}"
                        policy_sections[section_id] = section
                continue
                
            # Extract sections based on indices
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i+1]
                section = policy_text[start:end].strip()
                
                if len(section) > 10:  # Skip very short sections
                    section_id = f"{policy_id}_section_{i}"
                    policy_sections[section_id] = section
        
        return policy_sections

    def _group_related_obligations(self, obligations: List[Dict]) -> List[List[Dict]]:
        """Group related obligations to improve context awareness in matching."""
        # Start with each obligation in its own group
        obligation_groups = [[obligation] for obligation in obligations]
        
        # Track which group each obligation is in
        obligation_to_group = {obligation["id"]: i for i, group in enumerate(obligation_groups) 
                             for obligation in group}
        
        # Combine groups based on explicit relationships
        for obligation in obligations:
            if "related_obligations" in obligation:
                # Current group index
                current_group_idx = obligation_to_group[obligation["id"]]
                current_group = obligation_groups[current_group_idx]
                
                # Find related obligations and merge groups
                for related_id in obligation.get("related_obligations", []):
                    if related_id in obligation_to_group and obligation_to_group[related_id] != current_group_idx:
                        related_group_idx = obligation_to_group[related_id]
                        
                        # Merge groups
                        current_group.extend(obligation_groups[related_group_idx])
                        obligation_groups[related_group_idx] = []  # Empty the merged group
                        
                        # Update group indices
                        for obl in obligation_groups[current_group_idx]:
                            obligation_to_group[obl["id"]] = current_group_idx
        
        # Filter out empty groups and sort by primary obligation confidence
        return [group for group in obligation_groups if group]

    def _generate_match_details(self, obligation: Dict, policy_text: str, 
                              base_score: float, adjusted_score: float,
                              threshold: float) -> Dict:
        """Generate detailed explanation for a match."""
        # Calculate individual component scores
        keyword_score = self._calculate_keyword_score(obligation, policy_text)
        entity_score = self._calculate_entity_overlap(obligation, policy_text)
        domain_score = self._calculate_domain_match(obligation, policy_text)
        
        # Identify top contributing keywords
        keywords = obligation.get("keywords", [])
        matched_keywords = []
        
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', policy_text, re.IGNORECASE):
                matched_keywords.append(keyword)
        
        # Extract sentence snippets that likely match
        snippets = []
        policy_sentences = re.split(r'[.!?]\s+', policy_text)
        for sentence in policy_sentences:
            if any(re.search(r'\b' + re.escape(kw) + r'\b', sentence, re.IGNORECASE) for kw in matched_keywords):
                if len(sentence) > 10:  # Skip very short sentences
                    snippets.append(sentence.strip())
        
        return {
            "component_scores": {
                "base_similarity": float(base_score),
                "keyword_match": float(keyword_score),
                "entity_overlap": float(entity_score),
                "domain_match": float(domain_score)
            },
            "matched_keywords": matched_keywords[:5],  # Top 5 keywords
            "relevant_snippets": snippets[:3],  # Top 3 snippets
            "confidence": float((adjusted_score - threshold) / max(0.2, 1 - threshold))  # Normalized confidence
        }

    def _add_calibration_data(self, obligation: Dict, matches: List[Dict], threshold: float) -> None:
        """Add data point for threshold calibration."""
        # Store information about this matching attempt for later calibration
        ground_truth = obligation.get("ground_truth", {})
        if ground_truth and "policy_id" in ground_truth:
            # Find if we matched the ground truth
            matched_correct = any(m.get("policy_id") == ground_truth["policy_id"] for m in matches)
            
            # Store data point
            self.calibration_data.append({
                "domain": obligation.get("domain", ["general"])[0],
                "strength": obligation.get("strength", "medium"),
                "threshold_used": threshold,
                "matched_correct": matched_correct,
                "top_score": matches[0].get("adjusted_score", 0) if matches else 0
            })

    def _calibrate_thresholds(self) -> None:
        """Analyze calibration data to optimize thresholds."""
        self.logger.info(f"Calibrating thresholds with {len(self.calibration_data)} data points")
        
        # Group by domain
        domain_data = {}
        for data_point in self.calibration_data:
            domain = data_point["domain"]
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(data_point)
        
        # Calculate optimal thresholds for each domain
        for domain, data_points in domain_data.items():
            if len(data_points) < 5:  # Skip domains with too little data
                continue
                
            # Try different thresholds
            best_f1 = 0
            best_threshold = self.thresholds.get(domain, self.thresholds["base"])
            
            test_thresholds = np.linspace(0.4, 0.9, 11)  # Test 11 values from 0.4 to 0.9
            
            for threshold in test_thresholds:
                # Calculate precision, recall, F1 for this threshold
                tp, fp, fn = 0, 0, 0
                
                for point in data_points:
                    top_score = point["top_score"]
                    is_correct = point["matched_correct"]
                    
                    if top_score >= threshold and is_correct:
                        tp += 1
                    elif top_score >= threshold and not is_correct:
                        fp += 1
                    elif top_score < threshold and is_correct:
                        fn += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            # Update threshold for this domain
            self.thresholds[domain] = best_threshold
            self.logger.info(f"Calibrated threshold for {domain}: {best_threshold:.2f} (F1: {best_f1:.2f})")
        
        # Clear calibration data for the next round
        self.calibration_data = []

    def _apply_adaptive_learning(self) -> None:
        """Apply adaptive learning to improve matching weights based on reviews."""
        correct_count = sum(1 for r in self.reviewed_matches if r["is_correct"])
        incorrect_count = len(self.reviewed_matches) - correct_count
        
        if not self.reviewed_matches:
            return
            
        # Analyze match components to see what features correlate with correct matches
        component_correlation = {
            "keyword_match": 0,
            "semantic_similarity": 0,
            "entity_overlap": 0,
            "domain_match": 0
        }
        
        # Calculate correlation between components and correctness
        for review in self.reviewed_matches:
            match = review["match"]
            is_correct = review["is_correct"]
            
            # Get component scores
            component_scores = match.get("match_details", {}).get("component_scores", {})
            
            # For each component, if high score correlates with correct matches, adjust up
            for component, score in component_scores.items():
                if component in component_correlation:
                    if (is_correct and score > 0.6) or (not is_correct and score < 0.4):
                        component_correlation[component] += 1
                    elif (is_correct and score < 0.4) or (not is_correct and score > 0.6):
                        component_correlation[component] -= 1
        
        # Adjust weights based on correlation
        total_reviews = len(self.reviewed_matches)
        for component, correlation in component_correlation.items():
            if correlation != 0 and component in self.context_weights:
                # Calculate adjustment - normalize by number of reviews
                adjustment = self.learning_rate * (correlation / total_reviews)
                
                # Apply adjustment
                self.context_weights[component] += adjustment
                
                # Enforce minimum weight
                self.context_weights[component] = max(self.min_weight, self.context_weights[component])
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(self.context_weights.values())
        if total_weight > 0:
            for component in self.context_weights:
                self.context_weights[component] /= total_weight
                
        self.logger.info(f"Applied adaptive learning. Updated weights: {self.context_weights}")
        
        # Also optimize thresholds
        self.threshold_optimizer.optimize_thresholds()
        self.thresholds = self.threshold_optimizer.thresholds

    def _create_non_compliant_entry(self, obligation: Dict) -> Dict:
        """Create a non-compliant entry for an obligation.
        
        Args:
            obligation: Regulatory obligation dictionary
            
        Returns:
            Non-compliant compliance entry dictionary
        """
        return {
            "obligation_id": obligation["id"],
            "requirement": obligation["text"],
            "keywords": obligation.get("keywords", []),
            "status": "Non-Compliant",
            "score": 0.0,
            "policy": None,
            "policy_text": None
        }

    def find_matches(self, obligations, policies):
        """Find matches between regulatory obligations and internal policies.
        
        This is a convenience wrapper around the compare method that returns results
        in a more accessible format.
        
        Args:
            obligations (list): List of regulatory obligations
            policies (dict): Dictionary of company policies
            
        Returns:
            dict: Mapping of obligations to matching policies with scores
        """
        results = self.compare(obligations, policies)
        
        matches = {}
        for result in results:
            obligation_id = result.get("obligation_id")
            if obligation_id not in matches:
                matches[obligation_id] = []
                
            if result.get("policy"):
                matches[obligation_id].append((
                    result.get("policy"),
                    result.get("score", 0)
                ))
                
        return matches
    
    def identify_gaps(self, matches):
        """Identify compliance gaps based on matches.
        
        Args:
            matches (dict): Mapping of obligations to matching policies
            
        Returns:
            list: Unmatched or poorly matched obligations
        """
        gaps = []
        for obligation_id, policy_matches in matches.items():
            if not policy_matches or max(m[1] for m in policy_matches) < 0.3:
                gaps.append({
                    "obligation_id": obligation_id,
                    "status": "unmatched" if not policy_matches else "poor_match",
                    "recommendations": self._generate_recommendations(obligation_id)
                })
        return gaps
    
    def _generate_recommendations(self, obligation_id):
        """Generate recommendations for addressing an unmatched obligation.
        
        Args:
            obligation_id (str): The ID of the unmatched obligation
            
        Returns:
            list: Recommended actions
        """
        # This is a placeholder implementation that would be expanded in a real system
        return ["Create a new policy document addressing this obligation"]
