"""Module for domain-specific language models and terminology."""
import logging
from typing import Dict, List, Set, Optional, Tuple, Any, Union # Added Any, Union for good measure
from collections import Counter # Added Counter
import re
import json
from pathlib import Path
import spacy
from spacy.tokens import Token
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span


class DomainSpecificProcessor:
    """
    Provides domain-specific language processing capabilities for regulatory text.
    
    This class enhances standard NLP models with industry-specific terminology,
    patterns, and rules to better understand specialized regulatory language.
    """
    
    def __init__(self, domains: Optional[List[str]] = None, base_model: str = "en_core_web_sm"):
        """
        Initialize with specified regulatory domains.
        
        Args:
            domains: List of domains to load (e.g., "financial", "healthcare")
            base_model: Base spaCy model to enhance
        """
        self.logger = logging.getLogger(__name__)
        self.domains = domains or ["financial", "healthcare", "data_privacy", "environmental"]
        
        # Load base spaCy model
        try:
            self.nlp = spacy.load(base_model)
            self.logger.info(f"Loaded base NLP model: {base_model}")
        except OSError:
            self.logger.warning(f"Model {base_model} not found. Downloading...")
            spacy.cli.download(base_model)
            self.nlp = spacy.load(base_model)
        
        # Load domain-specific resources
        self.terminology = self._load_domain_terminology()
        
        # Enhance the NLP pipeline with domain-specific components
        self._enhance_pipeline()
        
        # Add standard document sections for better context awareness
        self.standard_sections = {
            "definitions": ["definitions", "terms", "glossary"],
            "scope": ["scope", "applicability", "application", "coverage"],
            "requirements": ["requirements", "obligations", "mandates", "rules"],
            "procedures": ["procedures", "processes", "methods", "steps"],
            "exceptions": ["exceptions", "exemptions", "exclusions"],
            "penalties": ["penalties", "enforcement", "violations", "sanctions"]
        }
        
        # Initialize section context history
        self.section_context = []
    
    def _load_domain_terminology(self) -> Dict[str, Dict]:
        """Load domain-specific terminology and patterns."""
        terminology = {}
        
        # Financial domain
        financial_terms = {
            "entities": [
                {"label": "REG_ENTITY", "pattern": "Securities and Exchange Commission"},
                {"label": "REG_ENTITY", "pattern": "SEC"},
                {"label": "REG_ENTITY", "pattern": "Federal Reserve"},
                {"label": "REG_ENTITY", "pattern": "FDIC"},
                {"label": "REG_ENTITY", "pattern": "OCC"},
                {"label": "REG_ENTITY", "pattern": "FINRA"},
                {"label": "REG_ENTITY", "pattern": "CFPB"},
                {"label": "REG_DOCUMENT", "pattern": "Form 10-K"},
                {"label": "REG_DOCUMENT", "pattern": "Form 10-Q"},
                {"label": "REG_DOCUMENT", "pattern": "prospectus"},
                {"label": "REG_CONCEPT", "pattern": "capital adequacy"},
                {"label": "REG_CONCEPT", "pattern": "liquidity coverage ratio"},
                {"label": "REG_CONCEPT", "pattern": "tier 1 capital"},
                {"label": "REG_CONCEPT", "pattern": "leverage ratio"},
                {"label": "REG_LAW", "pattern": "Dodd-Frank Act"},
                {"label": "REG_LAW", "pattern": "Bank Secrecy Act"},
                {"label": "REG_LAW", "pattern": "Sarbanes-Oxley Act"},
                {"label": "REG_LAW", "pattern": "Gramm-Leach-Bliley Act"},
                {"label": "REG_LAW", "pattern": "Basel III"},
            ],
            "obligation_verbs": [
                "disclose", "report", "file", "maintain", "establish", "implement",
                "ensure", "certify", "attest", "verify", "submit", "demonstrate"
            ],
            "action_objects": [
                "capital", "reports", "statements", "controls", "procedures", "reserves",
                "records", "filings", "ratios", "systems", "risks", "policies"
            ],
            "patterns": [
                r"(banks?|financial institutions?|broker-dealers?) (must|shall|are required to)",
                r"(annual|quarterly|monthly) (reports?|filings?|statements?)",
                r"(maintain|establish) (adequate|appropriate|reasonable) (controls|procedures|systems)"
            ]
        }
        
        # Healthcare domain
        healthcare_terms = {
            "entities": [
                {"label": "REG_ENTITY", "pattern": "Department of Health and Human Services"},
                {"label": "REG_ENTITY", "pattern": "HHS"},
                {"label": "REG_ENTITY", "pattern": "FDA"},
                {"label": "REG_ENTITY", "pattern": "OCR"},
                {"label": "REG_ENTITY", "pattern": "CMS"},
                {"label": "REG_DOCUMENT", "pattern": "Notice of Privacy Practices"},
                {"label": "REG_DOCUMENT", "pattern": "Business Associate Agreement"},
                {"label": "REG_DOCUMENT", "pattern": "Authorization Form"},
                {"label": "REG_CONCEPT", "pattern": "protected health information"},
                {"label": "REG_CONCEPT", "pattern": "electronic health record"},
                {"label": "REG_CONCEPT", "pattern": "covered entity"},
                {"label": "REG_CONCEPT", "pattern": "business associate"},
                {"label": "REG_LAW", "pattern": "HIPAA"},
                {"label": "REG_LAW", "pattern": "HITECH Act"},
                {"label": "REG_LAW", "pattern": "Affordable Care Act"},
                {"label": "REG_LAW", "pattern": "Medicare"},
                {"label": "REG_LAW", "pattern": "Medicaid"},
            ],
            "obligation_verbs": [
                "safeguard", "protect", "secure", "maintain", "ensure", "provide",
                "disclose", "restrict", "limit", "comply", "implement", "train"
            ],
            "action_objects": [
                "patients", "records", "information", "data", "privacy", "confidentiality",
                "security", "access", "rights", "authorization", "consent", "measures"
            ],
            "patterns": [
                r"(covered entities?|business associates?|healthcare providers?) (must|shall|are required to)",
                r"(patient|individual) (rights|authorization|consent)",
                r"(protect|safeguard|secure) (confidential|sensitive|protected) (information|data)"
            ]
        }
        
        # Data privacy domain
        privacy_terms = {
            "entities": [
                {"label": "REG_ENTITY", "pattern": "Data Protection Authority"},
                {"label": "REG_ENTITY", "pattern": "European Data Protection Board"},
                {"label": "REG_ENTITY", "pattern": "EDPB"},
                {"label": "REG_ENTITY", "pattern": "Federal Trade Commission"},
                {"label": "REG_ENTITY", "pattern": "FTC"},
                {"label": "REG_ENTITY", "pattern": "Information Commissioner's Office"},
                {"label": "REG_ENTITY", "pattern": "ICO"},
                {"label": "REG_DOCUMENT", "pattern": "Privacy Policy"},
                {"label": "REG_DOCUMENT", "pattern": "Privacy Notice"},
                {"label": "REG_DOCUMENT", "pattern": "Data Processing Agreement"},
                {"label": "REG_CONCEPT", "pattern": "personal data"},
                {"label": "REG_CONCEPT", "pattern": "data subject"},
                {"label": "REG_CONCEPT", "pattern": "data controller"},
                {"label": "REG_CONCEPT", "pattern": "data processor"},
                {"label": "REG_CONCEPT", "pattern": "lawful basis"},
                {"label": "REG_LAW", "pattern": "GDPR"},
                {"label": "REG_LAW", "pattern": "CCPA"},
                {"label": "REG_LAW", "pattern": "CPRA"},
                {"label": "REG_LAW", "pattern": "LGPD"},
                {"label": "REG_LAW", "pattern": "PIPEDA"},
            ],
            "obligation_verbs": [
                "process", "collect", "notify", "inform", "obtain", "delete",
                "erase", "provide", "restrict", "transfer", "record", "document"
            ],
            "action_objects": [
                "consent", "personal data", "data subjects", "individuals", "information",
                "access", "erasure", "rectification", "processing", "rights", "requests"
            ],
            "patterns": [
                r"(controllers?|processors?|organizations?|businesses?) (must|shall|are required to)",
                r"(obtain|receive|get) (explicit|valid|unambiguous|specific) consent",
                r"(data subject|consumer|individual) (rights|requests|access)"
            ]
        }
        
        # Environmental domain
        environmental_terms = {
            "entities": [
                {"label": "REG_ENTITY", "pattern": "Environmental Protection Agency"},
                {"label": "REG_ENTITY", "pattern": "EPA"},
                {"label": "REG_ENTITY", "pattern": "Department of Energy"},
                {"label": "REG_ENTITY", "pattern": "DOE"},
                {"label": "REG_DOCUMENT", "pattern": "Environmental Impact Statement"},
                {"label": "REG_DOCUMENT", "pattern": "Environmental Assessment"},
                {"label": "REG_DOCUMENT", "pattern": "Permit Application"},
                {"label": "REG_DOCUMENT", "pattern": "Compliance Certification"},
                {"label": "REG_CONCEPT", "pattern": "emissions"},
                {"label": "REG_CONCEPT", "pattern": "hazardous waste"},
                {"label": "REG_CONCEPT", "pattern": "pollutant discharge"},
                {"label": "REG_CONCEPT", "pattern": "remediation"},
                {"label": "REG_LAW", "pattern": "Clean Air Act"},
                {"label": "REG_LAW", "pattern": "Clean Water Act"},
                {"label": "REG_LAW", "pattern": "RCRA"},
                {"label": "REG_LAW", "pattern": "CERCLA"},
                {"label": "REG_LAW", "pattern": "NEPA"},
            ],
            "obligation_verbs": [
                "monitor", "report", "reduce", "prevent", "control", "maintain",
                "limit", "mitigate", "remediate", "clean", "dispose", "treat"
            ],
            "action_objects": [
                "emissions", "discharges", "waste", "pollutants", "contamination",
                "permits", "limits", "thresholds", "standards", "requirements", "conditions"
            ],
            "patterns": [
                r"(facilities?|operators?|owners?|sources?) (must|shall|are required to)",
                r"(monitor|measure|track) (emissions|discharges|releases|pollutants)",
                r"(reduce|minimize|prevent|mitigate) (pollution|contamination|environmental impact)"
            ]
        }
        
        # Store all terminology
        terminology = {
            "financial": financial_terms,
            "healthcare": healthcare_terms,
            "data_privacy": privacy_terms,
            "environmental": environmental_terms
        }
        
        return terminology
    
    def _enhance_pipeline(self):
        """Enhance the NLP pipeline with domain-specific components."""
        # Add entity ruler if not already present
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = EntityRuler(self.nlp)
            
            # Add entities from all selected domains
            patterns = []
            for domain in self.domains:
                if domain in self.terminology:
                    patterns.extend(self.terminology[domain].get("entities", []))
            
            ruler.add_patterns(patterns)
            self.nlp.add_pipe("entity_ruler", before="ner")
            self.logger.info("Added entity ruler with domain-specific entities")
        
        # Add custom pipeline component for regulatory language
        if "regulatory_language_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("regulatory_language_detector", last=True)
            self.logger.info("Added regulatory language detector")
            
        # Add custom component for context-aware obligation detection
        if "obligation_context_analyzer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("obligation_context_analyzer", after="regulatory_language_detector")
            self.logger.info("Added context-aware obligation analyzer")

    def process(self, text: str) -> Doc:
        """
        Process text with domain-specific enhancements.
        
        Args:
            text: The text to process
            
        Returns:
            spaCy Doc object with enhanced annotations
        """
        return self.nlp(text)
    
    def extract_domain_specific_features(self, doc: Doc) -> Dict[str, List[str]]:
        """
        Extract domain-specific features from processed text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary of domain-specific features
        """
        features = {
            "regulatory_entities": [],
            "regulatory_documents": [],
            "regulatory_concepts": [],
            "regulatory_laws": [],
            "obligation_indicators": [],
            "domain_specific_terms": []
        }
        
        # Extract custom entities
        for ent in doc.ents:
            if ent.label_ == "REG_ENTITY":
                features["regulatory_entities"].append(ent.text)
            elif ent.label_ == "REG_DOCUMENT":
                features["regulatory_documents"].append(ent.text)
            elif ent.label_ == "REG_CONCEPT":
                features["regulatory_concepts"].append(ent.text)
            elif ent.label_ == "REG_LAW":
                features["regulatory_laws"].append(ent.text)
        
        # Extract obligation indicators
        for token in doc:
            if token._.is_obligation_verb:
                features["obligation_indicators"].append(token.text)
        
        # Extract domain-specific terms based on patterns
        for domain in self.domains:
            if domain not in self.terminology:
                continue
                
            domain_patterns = self.terminology[domain].get("patterns", [])
            for pattern in domain_patterns:
                matches = re.finditer(pattern, doc.text, re.IGNORECASE)
                for match in matches:
                    features["domain_specific_terms"].append(match.group(0))
        
        return features
    
    def get_domain_terminology(self, domain: str) -> Dict:
        """
        Get terminology for a specific domain.
        
        Args:
            domain: Domain name (e.g., "financial", "healthcare")
            
        Returns:
            Dictionary of domain-specific terminology
        """
        if domain in self.terminology:
            return self.terminology[domain]
        return {}
    
    def detect_domain(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect the most likely regulatory domain(s) for a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of tuples (domain_name, confidence_score) sorted by confidence
        """
        doc = self.nlp(text)
        domain_scores = {}
        
        # Calculate score for each domain based on terminology matches
        for domain in self.domains:
            if domain not in self.terminology:
                continue
                
            score = 0
            domain_terms = self.terminology[domain]
            
            # Check for entities
            for entity in domain_terms.get("entities", []):
                if entity["pattern"].lower() in text.lower():
                    score += 2
            
            # Check for obligation verbs
            for verb in domain_terms.get("obligation_verbs", []):
                if re.search(rf"\b{verb}\b", text, re.IGNORECASE):
                    score += 1
            
            # Check for action objects
            for obj in domain_terms.get("action_objects", []):
                if re.search(rf"\b{obj}\b", text, re.IGNORECASE):
                    score += 1
            
            # Check for patterns
            for pattern in domain_terms.get("patterns", []):
                if re.search(pattern, text, re.IGNORECASE):
                    score += 3
            
            # Normalize score based on text length
            word_count = len(text.split())
            normalized_score = score / max(1, word_count / 20)
            
            domain_scores[domain] = normalized_score
        
        # Sort domains by score in descending order
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_domains
    
    def analyze_semantic_obligation_context(self, text: str, window_size: int = 3) -> Dict:
        """
        Analyze the semantic context of obligations within the text.
        
        Args:
            text: Input text to analyze
            window_size: Number of sentences to consider for context
            
        Returns:
            Dictionary with semantic context analysis
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Track obligation sentences
        obligation_contexts = []
        
        for i, sent in enumerate(sentences):
            # Check if sentence contains obligation
            is_obligation = False
            for token in sent:
                if token._.is_obligation_verb or hasattr(sent.doc, "_") and sent.doc._.contains_obligation:
                    is_obligation = True
                    break
                    
            if not is_obligation:
                continue
                
            # Get context window
            start_idx = max(0, i - window_size)
            end_idx = min(len(sentences), i + window_size + 1)
            
            # Get preceding context
            preceding = [sentences[j].text.strip() for j in range(start_idx, i)]
            
            # Get following context
            following = [sentences[j].text.strip() for j in range(i+1, end_idx)]
            
            # Analyze the obligation in context
            obligation_contexts.append({
                "obligation_text": sent.text.strip(),
                "preceding_context": preceding,
                "following_context": following,
                "subject_matter": self._extract_subject_matter(sent, preceding)
            })
            
        return {
            "obligations_in_context": obligation_contexts,
            "domain_distribution": self.detect_domain(text)
        }
        
    def _extract_subject_matter(self, sent, preceding_context: List[str]) -> str:
        """Extract subject matter from a sentence and its preceding context."""
        # Check for explicit subject in the sentence
        subject_phrases = []
        for token in sent:
            if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ == "NOUN":
                # Get the full noun phrase
                subject_phrases.append(" ".join([t.text for t in token.subtree]))
                
        # If no subject found, look in preceding context
        if not subject_phrases and preceding_context:
            # Check the last preceding sentence for subjects
            last_context = self.nlp(preceding_context[-1])
            for token in last_context:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.pos_ == "NOUN":
                    subject_phrases.append(" ".join([t.text for t in token.subtree]))
        
        if subject_phrases:
            return subject_phrases[0]
        return ""
    
    def analyze_section_context(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for section-based context and structure.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with section analysis
        """
        doc = self.nlp(text)
        sections = []
        current_section = {"title": "", "level": 0, "content": [], "subsections": []}
        
        for sent in doc.sents:
            if self._is_section_header(sent.text):
                # Process previous section if it exists
                if current_section["title"]:
                    sections.append(current_section)
                
                # Start new section
                level = self._determine_section_level(sent.text)
                current_section = {
                    "title": sent.text.strip(),
                    "level": level,
                    "content": [],
                    "subsections": [],
                    "type": self._classify_section_type(sent.text)
                }
            else:
                current_section["content"].append(sent.text.strip())
        
        # Add final section
        if current_section["title"]:
            sections.append(current_section)
        
        return {
            "sections": sections,
            "structure": self._analyze_document_structure(sections),
            "hierarchy": self._build_section_hierarchy(sections)
        }
    
    def _is_section_header(self, text: str) -> bool:
        """Enhanced section header detection."""
        patterns = [
            r"^(?:Section|Article|§)\s+\d+",
            r"^\d+(?:\.\d+)*\s+[A-Z]",
            r"^[IVXLC]+\.\s+",
            r"^[A-Z][A-Za-z\s]{2,50}:$"
        ]
        
        return any(re.match(p, text) for p in patterns)
    
    def _determine_section_level(self, header: str) -> int:
        """Determine the hierarchical level of a section header."""
        if re.match(r"^(?:Section|Article|§)\s+\d+$", header):
            return 1
        if re.match(r"^\d+\.\d+", header):
            return len(header.split("."))
        if re.match(r"^[a-z]\)", header):
            return 3
        return 2
    
    def _classify_section_type(self, header: str) -> str:
        """Classify the type of section based on its header."""
        header_lower = header.lower()
        
        for section_type, keywords in self.standard_sections.items():
            if any(keyword in header_lower for keyword in keywords):
                return section_type
                
        return "other"
    
    def _analyze_document_structure(self, sections: List[Dict]) -> Dict[str, Any]:
        """Analyze the document's structural patterns."""
        return {
            "num_sections": len(sections),
            "max_depth": max(s["level"] for s in sections),
            "section_types": Counter(s["type"] for s in sections),
            "avg_section_length": sum(len(s["content"]) for s in sections) / len(sections) if sections else 0
        }
    
    def _build_section_hierarchy(self, sections: List[Dict]) -> Dict[str, Any]:
        """Build hierarchical representation of document sections."""
        hierarchy = {}
        current_path = []
        
        for section in sections:
            level = section["level"]
            # Reset path if we're at top level
            if level == 1:
                current_path = []
            # Truncate path if we're going up in the hierarchy
            elif level <= len(current_path):
                current_path = current_path[:level-1]
            
            # Add current section to path
            current_path.append(section["title"])
            
            # Build nested dictionary path
            current_dict = hierarchy
            for path_item in current_path[:-1]:
                if path_item not in current_dict:
                    current_dict[path_item] = {}
                current_dict = current_dict[path_item]
            current_dict[current_path[-1]] = section
        
        return hierarchy


# Custom pipeline component for regulatory language detection
@spacy.Language.component("regulatory_language_detector")
def regulatory_language_detector(doc):
    """
    Custom pipeline component to detect regulatory language features.
    
    Adds custom attributes to tokens and spans related to regulatory language.
    """
    # Add custom attributes to tokens
    Doc.set_extension("regulatory_domain", default=None, force=True)
    Doc.set_extension("contains_obligation", default=False, force=True)
    
    if not Token.has_extension("is_obligation_verb"):
        Token.set_extension("is_obligation_verb", default=False)
    
    if not Token.has_extension("is_action_object"):
        Token.set_extension("is_action_object", default=False)
    
    # Obligation verbs
    obligation_verbs = {
        "must", "shall", "should", "require", "comply", "ensure", "maintain",
        "provide", "submit", "report", "disclose", "notify", "establish",
        "implement", "demonstrate", "document", "record"
    }
    
    # Regulatory action objects
    action_objects = {
        "records", "documentation", "policies", "procedures", "controls",
        "measures", "systems", "reports", "information", "data", "compliance"
    }
    
    # Detect obligation verbs and action objects
    for token in doc:
        if token.lemma_.lower() in obligation_verbs:
            token._.is_obligation_verb = True
            doc._.contains_obligation = True
        
        if token.lemma_.lower() in action_objects:
            token._.is_action_object = True
    
    # Detect if document contains obligations
    if not doc._.contains_obligation:
        for token in doc:
            # Look for passive constructions like "is required to"
            if token.dep_ == "auxpass" and token.head.lemma_.lower() in {"require", "mandate", "oblige"}:
                doc._.contains_obligation = True
                break
            
            # Look for phrases like "in accordance with" or "pursuant to"
            if token.text.lower() in {"accordance", "pursuant"} and any(t.text.lower() in {"with", "to"} for t in token.children):
                doc._.contains_obligation = True
                break
    
    return doc


# New custom pipeline component for context-aware obligation analysis
@spacy.Language.component("obligation_context_analyzer")
def obligation_context_analyzer(doc):
    """Custom pipeline component for analyzing obligations in context."""
    # Add span-level extensions
    if not Span.has_extension("obligation_type"):
        Span.set_extension("obligation_type", default=None)
    
    if not Span.has_extension("obligation_strength"):
        Span.set_extension("obligation_strength", default=None)
    
    if not Span.has_extension("has_condition"):
        Span.set_extension("has_condition", default=False)
    
    if not Span.has_extension("has_exception"):
        Span.set_extension("has_exception", default=False)
    
    # Define obligation strength indicators
    strong_indicators = {"must", "shall", "required", "mandated", "mandatory"}
    medium_indicators = {"should", "expected", "recommended", "encouraged"}
    weak_indicators = {"may", "can", "might", "could", "optional"}
    
    # Process each sentence
    for sent in doc.sents:
        # Set defaults
        sent._.obligation_type = "none"
        sent._.obligation_strength = "none"
        
        # Check for conditions
        for token in sent:
            if token.dep_ == "mark" and token.text.lower() in {"if", "when", "unless", "provided"}:
                sent._.has_condition = True
            
            # Check for exceptions
            if token.text.lower() in {"except", "excluding", "exempt", "unless"}:
                sent._.has_exception = True
        
        # Detect obligation type and strength
        text_lower = sent.text.lower()
        
        # Check for strong obligation indicators
        if any(ind in text_lower for ind in strong_indicators):
            sent._.obligation_type = "directive"
            sent._.obligation_strength = "strong"
        # Check for medium obligation indicators
        elif any(ind in text_lower for ind in medium_indicators):
            sent._.obligation_type = "directive"
            sent._.obligation_strength = "medium"
        # Check for weak/permissive obligation indicators
        elif any(ind in text_lower for ind in weak_indicators):
            sent._.obligation_type = "permissive"
            sent._.obligation_strength = "weak"
        # Check for passive forms (e.g., "is required to")
        elif re.search(r"\b(is|are)\s+(required|obligated|expected)\s+to\b", text_lower):
            sent._.obligation_type = "directive"
            sent._.obligation_strength = "strong"
        # Check for imperative forms (e.g., "Ensure compliance...")
        elif re.match(r"^(Ensure|Maintain|Provide|Submit|Report|Implement)", sent.text):
            sent._.obligation_type = "directive"
            sent._.obligation_strength = "medium"
        
    return doc
