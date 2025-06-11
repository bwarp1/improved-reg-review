"""Module for extracting regulatory obligations from text."""
import re
import logging
import os
from typing import List, Dict, Any, Optional, Set, Tuple
import yaml
import spacy
from pdfminer.high_level import extract_text as pdf_extract_text
from .domain_models import DomainSpecificProcessor
from compliance_poc.src.processing.chunker import DocumentChunker, ParallelProcessor

class ObligationExtractor:
    """
    Extracts regulatory obligations from text documents.
    
    Uses NLP techniques to identify sentences containing obligations,
    typically indicated by modal verbs like "must", "shall", etc.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialize the extractor with configuration settings.
        
        Args:
            config (dict, optional): Configuration dictionary
            config_path (str, optional): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration either from dict or file path
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            # Enhanced default configuration with more patterns
            self.config = {
                "model": "en_core_web_sm",
                "obligation_keywords": [
                    "must", "shall", "required", "mandatory", "should",
                    "obligated", "responsible for", "duty to", "bound to",
                    "necessary to", "need to", "expected to"
                ],
                "conditional_markers": [
                    "if", "when", "unless", "except", "provided that",
                    "in the event", "subject to", "as applicable"
                ],
                "temporal_markers": [
                    "within", "by", "before", "after", "no later than",
                    "prior to", "annually", "quarterly", "monthly"
                ]
            }
        
        # Add defaults for missing config values (KISS principle)
        if "obligation_keywords" not in self.config:
            self.config["obligation_keywords"] = ["must", "shall", "required", "mandatory", "should"]
        
        # Enhanced obligation keywords with stronger context awareness
        self.obligation_keywords = set(self.config["obligation_keywords"])
        self.strong_obligation_keywords = set(["must", "shall", "required", "mandated", "mandatory", "obligation"])
        self.medium_obligation_keywords = set(["should", "expected", "encouraged", "recommended"])
        self.conditional_obligation_keywords = set(["if", "when", "where", "in case", "provided that"])
        self.exception_keywords = set(["except", "unless", "excluding", "not required", "exempted"])
        self.temporal_keywords = set(["within", "by", "before", "after", "no later than", "prior to"])
        
        # Enhanced domain-specific regulatory keywords with more industry terms
        self.domain_keywords = {
            "financial": ["report", "disclose", "audit", "capital", "liquidity", "risk", "compliance", "regulation", 
                         "transaction", "deposit", "withdrawal", "security", "investment", "asset", "liability",
                         "dividend", "shareholder", "earnings", "profit", "loss", "disclosure", "quarterly", "annually"],
            "healthcare": ["privacy", "patient", "health", "record", "confidential", "hipaa", "consent", 
                          "treatment", "diagnosis", "provider", "hospital", "clinic", "physician", "nurse",
                          "medicare", "medicaid", "insurance", "prescription", "procedure", "sanitation"],
            "data_privacy": ["data", "privacy", "consent", "personal", "breach", "notification", "processing",
                            "controller", "processor", "subject", "rights", "access", "rectification", "erasure",
                            "gdpr", "ccpa", "cpra", "cookie", "tracking", "profiling", "third-party", "encryption"],
            "environmental": ["emission", "waste", "discharge", "pollutant", "contamination", "remediation",
                             "hazardous", "toxic", "spill", "sustainable", "conservation", "recycling", 
                             "biodegradable", "carbon", "greenhouse", "ozone", "mitigation", "permit", "disposal"]
        }
        
        # Add new domains for better coverage
        self.domain_keywords["employment"] = ["employee", "employer", "workplace", "discrimination", "harassment",
                                             "termination", "wages", "hours", "benefits", "leave", "workers", 
                                             "union", "labor", "compensation", "retirement", "safety", "osha"]
        
        self.domain_keywords["cybersecurity"] = ["security", "breach", "vulnerability", "threat", "incident", 
                                               "malware", "phishing", "encryption", "firewall", "authentication",
                                               "authorization", "control", "access", "protect", "patch", "update"]
        
        # Enhanced obligation categories by type and severity
        self.obligation_categories = {
            "reporting": ["report", "notify", "disclose", "submit", "file", "inform", "document"],
            "protection": ["protect", "secure", "safeguard", "encrypt", "shield", "preserve"],
            "compliance": ["comply", "adhere", "follow", "conform", "satisfy", "fulfill"],
            "prohibition": ["prohibit", "restrict", "prevent", "forbid", "ban", "disallow"],
            "permission": ["may", "can", "allow", "permit", "authorize", "enable"],
            "requirement": ["must", "shall", "require", "need", "necessary", "essential"]
        }
        
        # Severity indicators for obligation classification
        self.severity_indicators = {
            "high": ["immediately", "critical", "severe", "serious", "significant", "material", "substantial"],
            "medium": ["important", "relevant", "moderate", "notable", "marked"],
            "low": ["minor", "minimal", "slight", "limited", "small"]
        }
        
        # Timeframe indicators for temporal analysis
        self.timeframe_indicators = {
            "immediate": ["immediately", "promptly", "without delay", "at once", "forthwith"],
            "short_term": ["within 24 hours", "within 48 hours", "within 7 days", "within 30 days", "monthly"],
            "medium_term": ["within 90 days", "within 6 months", "quarterly", "semi-annually"],
            "long_term": ["annually", "yearly", "within 1 year", "within 5 years", "in the future"]
        }
        
        # Initialize domain-specific processor if enabled
        self.use_domain_specific = self.config.get("use_domain_specific", True)
        if self.use_domain_specific:
            domains = self.config.get("domains", ["financial", "healthcare", "data_privacy", "environmental"])
            self.domain_processor = DomainSpecificProcessor(domains=domains, base_model=self.config.get("model", "en_core_web_sm"))
            self.logger.info(f"Initialized domain-specific processor for domains: {domains}")
            # Use the enhanced model for NLP processing
            self.nlp = self.domain_processor.nlp
        else:
            # Load standard spaCy model
            self._load_nlp_model()
            
        # Compile regex patterns for faster matching
        self._compile_patterns()
        
        # Initialize chunking and parallel processing
        chunker_config = self.config.get("chunking", {})
        self.enable_chunking = chunker_config.get("enabled", True)
        self.max_chunk_size = chunker_config.get("max_size", 100000)
        self.chunk_overlap = chunker_config.get("overlap", 5000)
        self.chunker = DocumentChunker(
            max_chunk_size=self.max_chunk_size,
            overlap_size=self.chunk_overlap,
            preserve_sections=chunker_config.get("preserve_sections", True)
        )
        
        # Parallel processing configuration
        processor_config = self.config.get("parallel_processing", {})
        self.enable_parallel = processor_config.get("enabled", True)
        self.max_workers = processor_config.get("max_workers", None)
        self.processor = ParallelProcessor(
            max_workers=self.max_workers,
            show_progress=processor_config.get("show_progress", True)
        )
    
    def _load_nlp_model(self):
        """Load the specified NLP model, downloading if necessary."""
        model_name = self.config.get("model", "en_core_web_sm")
        try:
            self.nlp = spacy.load(model_name)
            self.logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            self.logger.warning(f"Model {model_name} not found. Downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
    
    def _compile_patterns(self):
        """Compile regex patterns for obligation detection."""
        # Basic obligation pattern
        basic_keywords = "|".join(self.obligation_keywords)
        self.obligation_pattern = re.compile(f"\\b({basic_keywords})\\b", re.IGNORECASE)
        
        # Strong obligation pattern
        strong_keywords = "|".join(self.strong_obligation_keywords)
        self.strong_pattern = re.compile(f"\\b({strong_keywords})\\b", re.IGNORECASE)
        
        # Medium obligation pattern
        medium_keywords = "|".join(self.medium_obligation_keywords)
        self.medium_pattern = re.compile(f"\\b({medium_keywords})\\b", re.IGNORECASE)
        
        # Conditional pattern
        conditional_keywords = "|".join(self.conditional_obligation_keywords)
        self.conditional_pattern = re.compile(f"\\b({conditional_keywords})\\b", re.IGNORECASE)
        
        # Exception pattern
        exception_keywords = "|".join(self.exception_keywords)
        self.exception_pattern = re.compile(f"\\b({exception_keywords})\\b", re.IGNORECASE)
        
        # Temporal pattern
        temporal_keywords = "|".join(self.temporal_keywords)
        self.temporal_pattern = re.compile(f"\\b({temporal_keywords})\\b", re.IGNORECASE)
        
        # Domain-specific patterns
        self.domain_patterns = {}
        for domain, keywords in self.domain_keywords.items():
            domain_keywords = "|".join(keywords)
            self.domain_patterns[domain] = re.compile(f"\\b({domain_keywords})\\b", re.IGNORECASE)
    
    def extract_obligations(self, text):
        """
        Extract regulatory obligations from text with enhanced domain awareness.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: List of obligation dictionaries
        """
        self.logger.info(f"Extracting obligations from text ({len(text)} chars)")
        
        # Use chunking for large documents
        if self.enable_chunking and len(text) > self.max_chunk_size:
            return self._extract_with_chunking(text)
        
        # For smaller documents, use the standard approach
        obligations = []
        
        # Process the document with appropriate NLP model
        doc = self.nlp(text)
        
        # Process text in context
        sentences = list(doc.sents)
        
        # Keep track of document context
        context = {
            "current_section": "",
            "current_topic": "",
            "prev_obligations": []
        }
        
        # Detect document-level domains for context
        document_domains = []
        if self.use_domain_specific:
            domain_scores = self.domain_processor.detect_domain(text)
            document_domains = [domain for domain, score in domain_scores if score > 0.1]
            self.logger.info(f"Detected document domains: {document_domains}")
        
        # Extract sentences that contain obligation keywords
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            
            # Skip very short sentences
            if len(sent_text) < 10:
                continue
            
            # Get surrounding context from nearby sentences
            surrounding_context = self._get_sentence_context(sentences, i)
                
            # Check if the sentence contains any obligation indicators using enhanced detection
            is_obligation = self._is_obligation_sentence(sent_text)
            
            # Additional domain-specific obligation detection if enabled
            if not is_obligation and self.use_domain_specific:
                # Check if sentence contains domain-specific regulatory language
                sent_doc = self.nlp(sent_text)
                if hasattr(sent_doc, "_") and sent_doc._.contains_obligation:
                    is_obligation = True
                    
            if is_obligation:
                # Analyze the sentence for obligation elements with domain context
                obligation_elements = self._analyze_obligation(sent, surrounding_context, document_domains)
                
                # Update current context if this appears to be a section header
                if self._is_section_header(sent_text):
                    context["current_section"] = sent_text
                
                # Detect the most specific domains for this sentence
                sentence_domains = []
                if self.use_domain_specific:
                    domain_scores = self.domain_processor.detect_domain(sent_text)
                    sentence_domains = [domain for domain, score in domain_scores if score > 0.2]
                    
                    # If no specific domains detected for the sentence, use document domains
                    if not sentence_domains:
                        sentence_domains = document_domains
                else:
                    sentence_domains = self._detect_domain(sent_text)
                
                # Extract domain-specific features if available
                domain_features = {}
                if self.use_domain_specific:
                    sent_doc = self.nlp(sent_text)
                    domain_features = self.domain_processor.extract_domain_specific_features(sent_doc)
                
                # Create an enhanced obligation object with domain awareness
                obligation = {
                    "id": f"OBL-{i+1:03d}",
                    "text": sent_text,
                    "keywords": self._extract_keywords(sent),
                    "entities": self._extract_entities(sent),
                    "subject": obligation_elements["subject"],
                    "action": obligation_elements["action"],
                    "condition": obligation_elements["condition"],
                    "deadline": obligation_elements["deadline"],
                    "strength": obligation_elements["strength"],
                    "confidence_score": obligation_elements["confidence"],
                    "domain": sentence_domains,
                    "section": context["current_section"],
                    "surrounding_context": surrounding_context,
                    "domain_features": domain_features
                }
                
                obligations.append(obligation)
                context["prev_obligations"].append(obligation)
        
        # Post-process to identify related obligations
        self._identify_related_obligations(obligations)
        
        # Add cross-references between related obligations for better context
        if obligations:
            self._add_obligation_cross_references(obligations)
        
        self.logger.info(f"Found {len(obligations)} obligations")
        return obligations
    
    def _extract_with_chunking(self, text):
        """Extract obligations from a large text using chunking."""
        self.logger.info("Using document chunking for large text")
        
        # Divide text into manageable chunks
        chunks = self.chunker.chunk_document(text)
        self.logger.info(f"Divided document into {len(chunks)} chunks")
        
        # Process chunks in parallel if enabled
        if self.enable_parallel and len(chunks) > 1:
            self.logger.info("Processing chunks in parallel")
            
            # Create processor instance with proper parameters
            processor = ParallelProcessor(max_workers=self.max_workers, show_progress=True)
            
            # Process chunks using the processor
            chunk_results = processor.process_chunks(
                chunks, 
                self._process_single_chunk
            )
            
            # Merge results from all chunks
            merged_results = self._merge_chunk_results(chunk_results)
            return merged_results.get('obligations', [])
        else:
            # Process chunks sequentially
            self.logger.info("Processing chunks sequentially")
            all_obligations = []
            for chunk in chunks:
                chunk_result = self._process_single_chunk(chunk)
                all_obligations.extend(chunk_result.get('obligations', []))
            
            return all_obligations

    def _process_single_chunk(self, chunk):
        """Process a single document chunk."""
        # Extract the chunk text
        chunk_text = chunk.get('text', '')
        chunk_section = chunk.get('section')
        
        # Process the document with spaCy
        doc = self.nlp(chunk_text)
        
        # Process text in context
        sentences = list(doc.sents)
        
        # Keep track of document context
        context = {
            "current_section": chunk_section or "",
            "current_topic": "",
            "prev_obligations": []
        }
        
        # Detect document-level domains for context
        document_domains = []
        if self.use_domain_specific:
            domain_scores = self.domain_processor.detect_domain(chunk_text)
            document_domains = [domain for domain, score in domain_scores if score > 0.1]
        
        # Extract obligations from this chunk using the standard logic
        chunk_obligations = []
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip()
            
            # Skip very short sentences
            if len(sent_text) < 10:
                continue
            
            # Get surrounding context from nearby sentences
            surrounding_context = self._get_sentence_context(sentences, i)
            
            # Check if the sentence contains any obligation indicators
            is_obligation = self._is_obligation_sentence(sent_text)
            
            # Additional domain-specific obligation detection if enabled
            if not is_obligation and self.use_domain_specific:
                sent_doc = self.nlp(sent_text)
                if hasattr(sent_doc, "_") and sent_doc._.contains_obligation:
                    is_obligation = True
            
            if is_obligation:
                # Analyze the sentence for obligation elements with domain context
                obligation_elements = self._analyze_obligation(sent, surrounding_context, document_domains)
                
                # Update current context if this appears to be a section header
                if self._is_section_header(sent_text):
                    context["current_section"] = sent_text
                
                # Detect the most specific domains for this sentence
                sentence_domains = []
                if self.use_domain_specific:
                    domain_scores = self.domain_processor.detect_domain(sent_text)
                    sentence_domains = [domain for domain, score in domain_scores if score > 0.2]
                    
                    # If no specific domains detected for the sentence, use document domains
                    if not sentence_domains:
                        sentence_domains = document_domains
                else:
                    sentence_domains = self._detect_domain(sent_text)
                
                # Extract domain-specific features if available
                domain_features = {}
                if self.use_domain_specific:
                    sent_doc = self.nlp(sent_text)
                    domain_features = self.domain_processor.extract_domain_specific_features(sent_doc)
                
                # Create obligation with chunk-specific identifier
                chunk_idx = chunk.get('index', 0)
                obligation = {
                    "id": f"OBL-{chunk_idx}-{i+1:03d}",
                    "text": sent_text,
                    "keywords": self._extract_keywords(sent),
                    "entities": self._extract_entities(sent),
                    "subject": obligation_elements["subject"],
                    "action": obligation_elements["action"],
                    "condition": obligation_elements["condition"],
                    "deadline": obligation_elements["deadline"],
                    "strength": obligation_elements["strength"],
                    "confidence_score": obligation_elements["confidence"],
                    "domain": sentence_domains,
                    "section": context["current_section"],
                    "surrounding_context": surrounding_context,
                    "domain_features": domain_features,
                    # Add chunk metadata for later reference
                    "chunk_index": chunk_idx,
                    "chunk_section": chunk_section,
                    "start_pos": chunk.get('start_pos', 0) + sent.start_char,
                    "end_pos": chunk.get('start_pos', 0) + sent.end_char
                }
                
                chunk_obligations.append(obligation)
                context["prev_obligations"].append(obligation)
        
        # Return the chunk results
        return {
            'obligations': chunk_obligations,
            'chunk_info': chunk
        }

    def _merge_chunk_results(self, chunk_results) -> Dict[str, List]:
        """
        Merge results from multiple document chunks.
        
        Args:
            chunk_results: Results from chunk processing
            
        Returns:
            Consolidated dictionary with merged results
        """
        self.logger.info(f"Merging results from {len(chunk_results)} chunks")
        
        # Initialize merged results structure
        merged_results = {
            'obligations': []
        }
        
        # Track seen obligations to avoid duplicates
        seen_obligations = set()
        
        # Process all chunk results
        for result in chunk_results:
            if not result:
                continue
                
            # Add new obligations, avoiding duplicates
            for obligation in result.get('obligations', []):
                # Create a simple hash for obligation text to detect duplicates
                obligation_hash = hash(obligation.get('text', ''))
                
                if obligation_hash not in seen_obligations:
                    seen_obligations.add(obligation_hash)
                    merged_results['obligations'].append(obligation)
        
        # Post-process the merged obligations
        if merged_results['obligations']:
            # Renumber the obligation IDs to be sequential
            for i, obligation in enumerate(merged_results['obligations']):
                obligation['id'] = f"OBL-{i+1:03d}"
            
            # Identify related obligations across chunks
            self._identify_related_obligations(merged_results['obligations'])
            
            # Add cross-references between related obligations
            self._add_obligation_cross_references(merged_results['obligations'])
            
        self.logger.info(f"Merged into {len(merged_results['obligations'])} unique obligations")
        return merged_results

    def _is_obligation_sentence(self, text: str) -> bool:
        """
        Determine if a sentence contains obligation indicators.
        Enhanced to detect more subtle obligation patterns.
        
        Args:
            text (str): The sentence text
            
        Returns:
            bool: True if the sentence contains obligation indicators
        """
        # Direct obligation keyword check
        if self.obligation_pattern.search(text):
            return True
        
        # Check for passive obligation forms: "is required to", "are expected to"
        passive_forms = [
            r"\b(is|are)\s+required\s+to\b",
            r"\b(is|are)\s+expected\s+to\b",
            r"\b(is|are)\s+obligated\s+to\b",
            r"\b(is|are)\s+mandated\s+to\b",
            r"\b(is|are)\s+responsible\s+for\b",
            r"\bneed\s+to\b",
            r"\bmust\s+be\b"
        ]
        for pattern in passive_forms:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for imperative verbs at beginning of sentence (common in regulations)
        imperative_verbs = [
            r"^(Ensure|Maintain|Provide|Submit|Report|Implement|Establish|Develop|Perform|Conduct)\b"
        ]
        for pattern in imperative_verbs:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for regulatory references that often indicate obligations
        regulatory_refs = [
            r"\bpursuant\s+to\b",
            r"\bin\s+accordance\s+with\b",
            r"\bas\s+required\s+by\b",
            r"\bin\s+compliance\s+with\b",
        ]
        for pattern in regulatory_refs:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for responsibility assignment patterns
        responsibility_patterns = [
            r"\b(responsible\s+for|accountable\s+for|tasked\s+with|charged\s+with)\b",
            r"\b(duty|obligation|responsibility)\s+to\b",
            r"\b(shall|will)\s+be\s+responsible\b"
        ]
        for pattern in responsibility_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for numerical requirements which often indicate obligations
        numerical_reqs = [
            r"\b(minimum|maximum|at\s+least|not\s+less\s+than|not\s+more\s+than|up\s+to|not\s+exceed)\s+\d+",
            r"\b\d+\s+percent\b",
            r"\b\d+\s+(days|months|years)\b"
        ]
        for pattern in numerical_reqs:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
                
    def _analyze_obligation(self, sent, context: str, domains: List[str] = None) -> Dict[str, Any]:
        """
        Analyze an obligation sentence to extract its components.
        Enhanced with context awareness and more detailed extraction.
        
        Args:
            sent: The spaCy sentence
            context: Surrounding context
            domains: List of detected domains
            
        Returns:
            dict: Components of the obligation
        """
        sent_text = sent.text.strip().lower()
        result = {
            "subject": "",
            "action": "",
            "condition": "",
            "deadline": "",
            "strength": "medium",  # Default strength
            "confidence": 0.5,     # Default confidence
            "obligation_type": "", # New: type classification
            "severity": "",        # New: severity level
            "timeframe": "",       # New: timeframe classification
            "applies_to": []       # New: entities to which the obligation applies
        }
        
        # Extract subject (who has the obligation)
        result["subject"] = self._extract_subject(sent)
        
        # Extract action (what must be done)
        result["action"] = self._extract_action(sent)
        
        # Extract condition with improved pattern recognition
        conditions = []
        # Check for conditional markers at sentence start
        if re.search(r"^(If|When|Where|Unless|Provided that)", sent.text, re.IGNORECASE):
            # The entire sentence is likely conditional for another obligation
            result["is_condition_only"] = True
        
        # Extract conditional clauses within the sentence
        if self.conditional_pattern.search(sent_text):
            # Look for conditional clauses
            for token in sent:
                if token.dep_ == "mark" and token.text.lower() in ["if", "when", "unless", "until", "provided"]:
                    # Extract the subordinate clause
                    clause = [t.text for t in token.subtree]
                    conditions.append(" ".join(clause))
        result["condition"] = "; ".join(conditions) if conditions else ""
        
        # Extract deadlines with improved temporal recognition
        deadlines = []
        if self.temporal_pattern.search(sent_text):
            # Look for temporal expressions
            temporal_phrases = [
                r"within\s+\d+\s+(days?|weeks?|months?|years?)",
                r"no\s+later\s+than\s+\w+\s+\d+",
                r"by\s+\w+\s+\d+",
                r"before\s+\w+\s+\d+",
                r"annually|quarterly|monthly|weekly|daily",
                r"every\s+\d+\s+(days?|weeks?|months?|years?)",
                r"once\s+(a|per)\s+(day|week|month|quarter|year)"
            ]
            for phrase in temporal_phrases:
                matches = re.finditer(phrase, sent_text, re.IGNORECASE)
                for match in matches:
                    deadlines.append(match.group(0))
        result["deadline"] = "; ".join(deadlines) if deadlines else ""
        
        # Determine obligation strength
        if self.strong_pattern.search(sent_text):
            result["strength"] = "strong"
        elif self.medium_pattern.search(sent_text):
            result["strength"] = "medium"
        else:
            result["strength"] = "weak"
        
        # NEW: Determine obligation type
        result["obligation_type"] = self._classify_obligation_type(sent)
        
        # NEW: Determine severity level
        result["severity"] = self._determine_severity(sent_text)
        
        # NEW: Determine timeframe category
        result["timeframe"] = self._classify_timeframe(result["deadline"], sent_text)
        
        # NEW: Extract entities to which the obligation applies
        result["applies_to"] = self._extract_obligation_targets(sent, context, domains)
        
        # Determine confidence score with improved heuristics
        confidence = 0.5  # Base confidence
        # Strong obligation terms increase confidence
        if result["strength"] == "strong":
            confidence += 0.3
        elif result["strength"] == "medium":
            confidence += 0.1
        # Having a subject increases confidence
        if result["subject"]:
            confidence += 0.1
        # Having an action increases confidence
        if result["action"]:
            confidence += 0.1
        # Having a clear obligation type increases confidence
        if result["obligation_type"]:
            confidence += 0.05
        # Exception terms decrease confidence
        if self.exception_pattern.search(sent_text):
            confidence -= 0.2
        # Adjust confidence based on domain specificity
        if domains and hasattr(self, 'domain_processor'):
            # Check for domain-specific regulatory terms
            for domain in domains:
                domain_terms = self.domain_processor.get_domain_terminology(domain)
                # Look for domain-specific obligation verbs
                for verb in domain_terms.get('obligation_verbs', []):
                    if re.search(rf"\b{verb}\b", sent.text, re.IGNORECASE):
                        confidence += 0.05
                        break
                # Look for domain-specific action objects
                for obj in domain_terms.get('action_objects', []):
                    if re.search(rf"\b{obj}\b", sent.text, re.IGNORECASE):
                        confidence += 0.05
                        break
        # Cap confidence between 0 and 1
        result["confidence"] = max(0.0, min(1.0, confidence))
        
        return result

    def _extract_subject(self, sent_doc: spacy.tokens.doc.Doc) -> str:
        """
        Extract the subject(s) of an obligation from a spaCy sentence.
        Looks for nsubj (nominal subject) or nsubjpass (passive nominal subject).
        """
        subjects = []
        # Ensure sent_doc is a spaCy Span/Doc object
        if not hasattr(sent_doc, 'ents'): # Basic check if it's a spaCy object
            self.logger.warning(f"_extract_subject received non-spaCy object: {type(sent_doc)}")
            return ""

        for token in sent_doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                # Capture the whole noun phrase for the subject
                # Collect text from all tokens in the subtree of the subject token
                subject_phrase = "".join(t.text_with_ws for t in token.subtree).strip()
                subjects.append(subject_phrase)
        
        if not subjects and sent_doc.root.pos_ == "VERB": # If no clear subject, check root verb's children
            for child in sent_doc.root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject_phrase = "".join(t.text_with_ws for t in child.subtree).strip()
                    subjects.append(subject_phrase)

        return ", ".join(list(set(subjects))) if subjects else ""

    def _extract_action(self, sent_doc: spacy.tokens.doc.Doc) -> str:
        """
        Extract the action phrase from an obligation sentence.
        Focuses on the main verb and its direct objects or complements.
        """
        action_phrases = []
        # Ensure sent_doc is a spaCy Span/Doc object
        if not hasattr(sent_doc, 'ents'): # Basic check if it's a spaCy object
            self.logger.warning(f"_extract_action received non-spaCy object: {type(sent_doc)}")
            return ""

        # Find the root verb of the sentence
        root_verb = None
        for token in sent_doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root_verb = token
                break
        
        if not root_verb: # If no clear root verb, try to find main verb related to obligation
            for token in sent_doc:
                if token.lemma_ in self.obligation_keywords and token.head.pos_ == "VERB":
                    root_verb = token.head
                    break
                elif token.pos_ == "VERB" and token.dep_ != "aux" and token.lemma_ not in ["be", "have"]:
                    # Fallback to first non-auxiliary verb if no clear root/obligation verb
                    if not root_verb:
                         root_verb = token


        if root_verb:
            # Collect the verb and its objects/complements
            action_tokens = [root_verb]
            for child in root_verb.children:
                if child.dep_ in ("dobj", "obj", "attr", "oprd", "acomp", "xcomp"):
                    action_tokens.extend(list(child.subtree))
                # Include particles for phrasal verbs
                elif child.dep_ == "prt":
                    action_tokens.append(child)
            
            # Sort tokens by their original index in the sentence to maintain order
            action_tokens.sort(key=lambda t: t.i)
            action_phrase = "".join(t.text_with_ws for t in action_tokens).strip()
            # Further refinement: try to capture more of the verb phrase
            # e.g., if root_verb is part of a larger verb phrase like "is required to submit"
            # we want "submit" and its complements.
            # The current logic might get "is" or "required".
            # A more robust way is to find the main content verb.
            
            # Re-evaluate: find modal, then its head verb, then that verb's phrase.
            modal_verb = None
            main_content_verb = None

            for token in sent_doc:
                if token.lemma_ in self.obligation_keywords and token.pos_ == "AUX": # e.g. must, shall, should
                    modal_verb = token
                    if modal_verb.head.pos_ == "VERB":
                        main_content_verb = modal_verb.head
                        break
            
            if not main_content_verb: # If no modal, use the root verb if it's not an aux
                 if root_verb and root_verb.pos_ != "AUX":
                      main_content_verb = root_verb

            if main_content_verb:
                action_tokens = [main_content_verb]
                q = list(main_content_verb.children)
                visited_children = {main_content_verb}
                while q:
                    child = q.pop(0)
                    if child in visited_children:
                        continue
                    visited_children.add(child)
                    # Include direct objects, clausal complements, adjectival complements, particles
                    if child.dep_ in ("dobj", "obj", "ccomp", "xcomp", "acomp", "prt", "oprd"):
                        action_tokens.extend(list(child.subtree))
                    # For prepositional objects, include the preposition and its object's subtree
                    elif child.dep_ == "prep":
                        action_tokens.append(child) # add the preposition
                        action_tokens.extend(list(c for c_sub in child.children if c_sub.dep_ == "pobj" for c in c_sub.subtree))


                action_tokens.sort(key=lambda t: t.i)
                action_phrase = "".join(t.text_with_ws for t in action_tokens).strip()
                action_phrases.append(action_phrase)

        return "; ".join(list(set(action_phrases))) if action_phrases else ""
    
    def _classify_obligation_type(self, sent) -> str:
        """
        Classify the type of obligation (reporting, protection, compliance, etc.).
        
        Args:
            sent: The spaCy sentence
            
        Returns:
            str: The obligation type
        """
        sent_text = sent.text.lower()
        
        # Check each category
        for category, indicators in self.obligation_categories.items():
            for indicator in indicators:
                if re.search(rf"\b{indicator}(?:\w*)\b", sent_text):
                    return category
        
        # Default to "requirement" if no specific type is identified
        if any(word in sent_text for word in ("must", "shall", "required")):
            return "requirement"
        elif any(word in sent_text for word in ("should", "recommend")):
            return "recommendation"
        elif any(word in sent_text for word in ("may", "can", "optionally")):
            return "permission"
            
        return "general"
    
    def _determine_severity(self, text: str) -> str:
        """
        Determine the severity level of an obligation based on keywords.
        
        Args:
            text: The text to analyze
            
        Returns:
            str: Severity level (high, medium, low)
        """
        text = text.lower()
        
        # Check explicit severity indicators
        for level, indicators in self.severity_indicators.items():
            if any(indicator in text for indicator in indicators):
                return level
                
        # Infer severity from other factors
        if "immediate" in text or "urgent" in text or "critical" in text:
            return "high"
        elif "must" in text or "shall" in text or "required" in text:
            return "high"
        elif "should" in text or "recommend" in text:
            return "medium"
        elif "may" in text or "can" in text or "option" in text:
            return "low"
            
        return "medium"  # Default to medium if no indicators
    
    def _classify_timeframe(self, deadline: str, text: str) -> str:
        """
        Classify the timeframe of an obligation.
        
        Args:
            deadline: Extracted deadline string
            text: Full text of the obligation
            
        Returns:
            str: Timeframe classification
        """
        if not deadline and not text:
            return ""
            
        combined_text = (deadline + " " + text).lower()
        
        # Check for immediate timeframe indicators
        for indicator in self.timeframe_indicators["immediate"]:
            if indicator in combined_text:
                return "immediate"
                
        # Check for short-term indicators
        for indicator in self.timeframe_indicators["short_term"]:
            if indicator in combined_text:
                return "short_term"
                
        # Check for medium-term indicators
        for indicator in self.timeframe_indicators["medium_term"]:
            if indicator in combined_text:
                return "medium_term"
                
        # Check for long-term indicators
        for indicator in self.timeframe_indicators["long_term"]:
            if indicator in combined_text:
                return "long_term"
        
        # Check for numeric timeframes
        days_match = re.search(r"within\s+(\d+)\s+days?", combined_text)
        if days_match:
            days = int(days_match.group(1))
            if days <= 7:
                return "immediate"
            elif days <= 30:
                return "short_term"
            elif days <= 180:
                return "medium_term"
            else:
                return "long_term"
                
        return ""  # No clear timeframe
    
    def _extract_obligation_targets(self, sent, context: str, domains: List[str] = None) -> List[str]:
        """
        Extract the entities to which the obligation applies.
        
        Args:
            sent: The spaCy sentence
            context: Surrounding context
            domains: List of detected domains
            
        Returns:
            list: Entities to which the obligation applies
        """
        targets = []
        
        # Common entity types that could be targets of obligations
        target_entities = ["ORG", "PERSON", "GPE", "FAC", "LOC", "NORP"]
        
        # Look for entities that might be targets
        for ent in sent.ents:
            if ent.label_ in target_entities:
                targets.append(ent.text)
        
        # If no entities found, look for common subject terms
        if not targets:
            common_subjects = {
                "financial": ["bank", "financial institution", "credit union", "broker", "dealer", "issuer"],
                "healthcare": ["provider", "hospital", "covered entity", "business associate", "clinic"],
                "data_privacy": ["controller", "processor", "business", "organization", "company"],
                "employment": ["employer", "business", "company", "organization"],
                "cybersecurity": ["operator", "organization", "entity", "business", "company"]
            }
            
            # Check if any domain-specific subjects are mentioned
            if domains:
                for domain in domains:
                    if domain in common_subjects:
                        for subject in common_subjects[domain]:
                            if re.search(rf"\b{subject}s?\b", sent.text, re.IGNORECASE):
                                targets.append(subject)
        
        return list(set(targets))  # Remove duplicates
    
    def _extract_from_file(self, file_path):
        """Extract obligations from a file (supports text, PDF)."""
        self.logger.info(f"Extracting obligations from file: {file_path}")
        
        # Determine file type and extract text accordingly
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.pdf':
            text = self._extract_text_from_pdf(file_path)
        else:
            # Assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return self.extract_obligations(text)
                
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        try:
            return pdf_extract_text(pdf_path)
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def _extract_keywords(self, span):
        """
        Extract important keywords from a sentence with improved relevance.
        """
        keywords = []
        # Extract nouns, verbs, and adjectives as keywords with improved relevance filtering
        for token in span:
            # Include nouns, verbs that aren't auxiliary, and descriptive adjectives
            if (token.pos_ == 'NOUN' and len(token.text) > 2) or \
               (token.pos_ == 'VERB' and token.dep_ not in ["aux", "auxpass"] and len(token.text) > 2) or \
               (token.pos_ == 'ADJ' and not token.is_stop and len(token.text) > 3):
                # Lemmatize for better matching
                keywords.append(token.lemma_.lower())
        # Add compound terms (multi-word expressions)
        for token in span:
            if token.pos_ == 'NOUN' and any(child.dep_ == 'compound' for child in token.children):
                compounds = [child.text.lower() for child in token.children if child.dep_ == 'compound']
                compounds.append(token.text.lower())
                keywords.append(' '.join(compounds))
        # Remove duplicates while preserving order
        seen = set()
        return [kw for kw in keywords if not (kw in seen or seen.add(kw))]

    def _extract_entities(self, span):
        """Extract named entities from a sentence."""
        entities = {}
        for ent in span.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
            
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            # Return default config
            return {
                "model": "en_core_web_sm",
                "obligation_keywords": ["must", "shall", "required", "mandatory", "should"]
            }

    def _get_sentence_context(self, sentences, sentence_idx, window=2) -> str:
        """
        Get surrounding sentences as context.
        
        Args:
            sentences: List of all sentences
            sentence_idx: Index of the current sentence
            window: Number of sentences to include before and after
            
        Returns:
            String containing context sentences
        """
        start_idx = max(0, sentence_idx - window)
        end_idx = min(len(sentences) - 1, sentence_idx + window)
        context_sentences = []
        for i in range(start_idx, end_idx + 1):
            if i != sentence_idx:  # Skip the current sentence
                context_sentences.append(sentences[i].text.strip())
        return " ".join(context_sentences)
                
    def _is_section_header(self, text) -> bool:
        """
        Identify if a sentence is likely a section header.
        
        Args:
            text: The sentence text
            
        Returns:
            True if the text appears to be a section header
        """
        # Check for common section header patterns
        section_patterns = [
            r"^(?:Section|Article|§)\s+\d+[\.\d]*\s*[:-]",
            r"^\d+[\.\d]*\s+[A-Z][a-zA-Z\s]+$",
            r"^[IVXLC]+\.\s+[A-Z][a-zA-Z\s]+$"
        ]
        for pattern in section_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for short all-caps lines that might be headers
        if text.isupper() and len(text.split()) <= 5 and len(text) <= 50:
            return True
        return False
            
    def _detect_domain(self, text) -> List[str]:
        """
        Detect regulatory domains applicable to the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of domain names that match the text
        """
        matching_domains = []
        for domain, pattern in self.domain_patterns.items():
            if pattern.search(text):
                matching_domains.append(domain)
        return matching_domains
                
    def _identify_related_obligations(self, obligations):
        """
        Identify related obligations and add cross-references.
        
        Args:
            obligations: List of obligation dictionaries
        """
        # Build keyword index for efficient matching
        keyword_index = {}
        for idx, obligation in enumerate(obligations):
            for keyword in obligation.get("keywords", []):
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(idx)
        
        # Look for related obligations by shared keywords and proximity
        for i, obligation in enumerate(obligations):
            related = set()
            for keyword in obligation.get("keywords", []):
                if keyword in keyword_index:
                    related.update(keyword_index[keyword])
            # Remove self-reference
            if i in related:
                related.remove(i)
            # Add cross-references if not already present
            if "related_obligations" not in obligation:
                obligation["related_obligations"] = []
            for rel_idx in related:
                rel_id = obligations[rel_idx]["id"]
                if rel_id not in obligation["related_obligations"]:
                    obligation["related_obligations"].append(rel_id)

    def _add_obligation_cross_references(self, obligations: List[Dict]) -> None:
        """
        Add cross-references between related obligations for better context analysis.
        
        Args:
            obligations: List of extracted obligations
        """
        # Create an index of obligations by section
        section_index = {}
        for i, obligation in enumerate(obligations):
            section = obligation.get("section", "")
            if section not in section_index:
                section_index[section] = []
            section_index[section].append(i)
        
        # Add section-based cross-references
        for i, obligation in enumerate(obligations):
            section = obligation.get("section", "")
            if section and section in section_index:
                # Add same-section obligations as related
                section_obligations = [
                    obligations[idx]["id"] for idx in section_index[section]
                    if idx != i  # Don't include self
                ]
                # Add to existing related obligations
                related = obligation.get("related_obligations", [])
                for obl_id in section_obligations:
                    if obl_id not in related:
                        related.append(obl_id)
                obligation["related_obligations"] = related
                
    def extract_complex_obligations(self, text):
        """
        Extract obligations with conditions and dependencies.
        Enhanced with context relationship detection.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            list: List of complex obligation dictionaries with relationship info
        """
        self.logger.info(f"Extracting complex obligations from text ({len(text)} chars)")
        complex_obligations = []
        
        # Process the document with spaCy
        doc = self.nlp(text)
        
        # Build a context map of sections and subjects for better relationship tracking
        context_map = self._build_context_map(doc)
        
        # Track obligations by ID for establishing relationships
        obligation_by_id = {}
        
        # Extract sentences that contain obligation keywords
        for i, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            
            # Skip very short sentences
            if len(sent_text) < 15:  # Complex obligations tend to be longer
                continue
            
            # Check if the sentence contains any obligation keywords
            if self.obligation_pattern.search(sent_text):
                # Look for conditional markers
                has_condition = any(token.dep_ == 'mark' and token.text.lower() in 
                                   ['if', 'when', 'unless', 'until', 'provided', 'whereas'] 
                                   for token in sent)
                
                # Extract detailed conditions
                conditions = self._extract_detailed_conditions(sent)
                
                # Look for temporal markers
                temporal_markers = [token.text for token in sent 
                                  if token.dep_ in ('prep', 'advmod') and 
                                  token.text.lower() in ['before', 'after', 'during', 'within', 'by']]
                
                # Look for dependencies between clauses
                dependencies = []
                for token in sent:
                    if token.dep_ == 'advcl' and token.head.pos_ == 'VERB':
                        dependencies.append(f"{token.text} -> {token.head.text}")
                
                # Get surrounding context from context map
                section_context = context_map["current_section_at_index"].get(i, "")
                subject_context = context_map["current_subject_at_index"].get(i, "")
                
                # Calculate the obligation strength
                strength = "medium"  # Default
                if any(word in sent_text.lower() for word in self.strong_obligation_keywords):
                    strength = "strong"
                elif any(word in sent_text.lower() for word in self.medium_obligation_keywords):
                    strength = "medium"
                else:
                    strength = "weak"
                
                # Create a complex obligation object with enhanced relationship info
                obligation = {
                    "id": f"COBL-{i+1:03d}",
                    "text": sent_text,
                    "keywords": self._extract_keywords(sent),
                    "entities": self._extract_entities(sent),
                    "is_conditional": has_condition,
                    "conditions": conditions,
                    "temporal_markers": temporal_markers,
                    "dependencies": dependencies,
                    "complexity_score": self._calculate_complexity(sent),
                    "section_context": section_context,
                    "subject_context": subject_context,
                    "strength": strength,
                    "related_obligations": [],
                    "potentially_exempted_by": [],
                    "exempts": []
                }
                
                # Store the obligation
                complex_obligations.append(obligation)
                obligation_by_id[obligation["id"]] = obligation
        
        # Analyze relationships between obligations
        self._analyze_obligation_relationships(complex_obligations, obligation_by_id, context_map)
        
        self.logger.info(f"Found {len(complex_obligations)} complex obligations")
        return complex_obligations
    
    def _build_context_map(self, doc) -> Dict[str, Dict[int, str]]:
        """
        Build a map of context information throughout the document.
        
        Args:
            doc: spaCy document
            
        Returns:
            Dict with context information indexed by sentence position
        """
        context_map = {
            "current_section_at_index": {},
            "current_subject_at_index": {},
            "sentence_by_index": {}
        }
        
        current_section = ""
        current_subject = ""
        
        # Process each sentence to build the context map
        for i, sent in enumerate(doc.sents):
            # Store the sentence
            context_map["sentence_by_index"][i] = sent
            
            # Check if this is a section header
            if self._is_section_header(sent.text):
                current_section = sent.text.strip()
            
            # Update section context for this sentence
            context_map["current_section_at_index"][i] = current_section
            
            # Try to identify main subjects in this sentence
            subjects = []
            for token in sent:
                if token.dep_ == "nsubj" and token.pos_ == "NOUN":
                    subjects.append(token.text)
            
            # If subjects found, update the current subject context
            if subjects:
                current_subject = ", ".join(subjects)
            
            # Update subject context for this sentence
            context_map["current_subject_at_index"][i] = current_subject
            
        return context_map
    
    def _extract_detailed_conditions(self, sent) -> List[Dict[str, str]]:
        """
        Extract detailed condition information from a sentence.
        
        Args:
            sent: spaCy sentence
            
        Returns:
            List of condition dictionaries with type and text
        """
        conditions = []
        
        # Look for conditional markers
        for token in sent:
            if token.dep_ == "mark" and token.text.lower() in ["if", "when", "unless", "until", "provided"]:
                # Determine condition type
                condition_type = "requirement"  # Default
                if token.text.lower() == "unless":
                    condition_type = "exception"
                elif token.text.lower() == "until":
                    condition_type = "temporal"
                
                # Get the conditional clause text
                clause_tokens = [t.text for t in token.subtree]
                clause_text = " ".join(clause_tokens)
                
                # Add the condition
                conditions.append({
                    "type": condition_type,
                    "text": clause_text,
                    "root_verb": token.head.text if token.head and token.head.pos_ == "VERB" else ""
                })
        
        return conditions
    
    def _analyze_obligation_relationships(self, obligations, obligation_by_id, context_map):
        """
        Analyze and establish relationships between obligations.
        
        Args:
            obligations: List of extracted obligations
            obligation_by_id: Dictionary mapping obligation IDs to obligations
            context_map: Context information map
        """
        # Find related obligations based on context and references
        for i, obligation in enumerate(obligations):
            # Get the sentence index from the obligation ID
            sent_idx = int(obligation["id"].split("-")[-1]) - 1
            
            # Check for references to other obligations in text
            for j, other_obl in enumerate(obligations):
                if i == j:
                    continue
                
                other_idx = int(other_obl["id"].split("-")[-1]) - 1
                
                # Check if they share the same section context
                if (context_map["current_section_at_index"].get(sent_idx) == 
                    context_map["current_section_at_index"].get(other_idx)):
                    # If they're nearby, they're likely related
                    if abs(sent_idx - other_idx) <= 3:
                        obligation["related_obligations"].append(other_obl["id"])
                
                # Check for exception relationships
                if obligation["is_conditional"]:
                    # If this is an "unless" condition and references another obligation's subject
                    for condition in obligation.get("conditions", []):
                        if condition["type"] == "exception":
                            # Check if the condition might exempt the other obligation
                            if any(kw in condition["text"].lower() for kw in other_obl["keywords"]):
                                obligation["exempts"].append(other_obl["id"])
                                other_obl["potentially_exempted_by"].append(obligation["id"])
        
        # Remove duplicates in relationship lists
        for obligation in obligations:
            obligation["related_obligations"] = list(set(obligation["related_obligations"]))
            obligation["exempts"] = list(set(obligation["exempts"]))
            obligation["potentially_exempted_by"] = list(set(obligation["potentially_exempted_by"]))

    def categorize_obligations(self, obligations):
        """
        Categorize obligations by type or topic.
        Enhanced with multi-dimensional categorization.
        
        Args:
            obligations (list): List of obligation dictionaries
            
        Returns:
            dict: Obligations grouped by multiple categories
        """
        self.logger.info(f"Categorizing {len(obligations)} obligations")
        
        # Initialize categorization dimensions
        categorized = {
            "by_type": {},       # By obligation type (reporting, protection, etc.)
            "by_domain": {},     # By regulatory domain (financial, healthcare, etc.)
            "by_severity": {},   # By severity (high, medium, low)
            "by_timeframe": {},  # By timeframe (immediate, short-term, etc.)
            "by_strength": {}    # By obligation strength (strong, medium, weak)
        }
        
        # Initialize categories in each dimension
        for obligation_type in set([o.get("obligation_type", "general") for o in obligations] + list(self.obligation_categories.keys())):
            categorized["by_type"][obligation_type] = []
            
        for domain in set([d for o in obligations for d in o.get("domain", ["general"])]):
            categorized["by_domain"][domain] = []
            
        for severity in ["high", "medium", "low", "unspecified"]:
            categorized["by_severity"][severity] = []
            
        for timeframe in ["immediate", "short_term", "medium_term", "long_term", "unspecified"]:
            categorized["by_timeframe"][timeframe] = []
            
        for strength in ["strong", "medium", "weak"]:
            categorized["by_strength"][strength] = []
        
        # Categorize each obligation across all dimensions
        for obligation in obligations:
            # By type
            obligation_type = obligation.get("obligation_type", "general")
            if obligation_type in categorized["by_type"]:
                categorized["by_type"][obligation_type].append(obligation)
            else:
                categorized["by_type"]["general"].append(obligation)
                
            # By domain
            domains = obligation.get("domain", ["general"])
            for domain in domains:
                if domain in categorized["by_domain"]:
                    categorized["by_domain"][domain].append(obligation)
                else:
                    categorized["by_domain"]["general"].append(obligation)
            
            # By severity
            severity = obligation.get("severity", "unspecified")
            if severity in categorized["by_severity"]:
                categorized["by_severity"][severity].append(obligation)
            else:
                categorized["by_severity"]["unspecified"].append(obligation)
                
            # By timeframe
            timeframe = obligation.get("timeframe", "unspecified")
            if timeframe in categorized["by_timeframe"]:
                categorized["by_timeframe"][timeframe].append(obligation)
            else:
                categorized["by_timeframe"]["unspecified"].append(obligation)
                
            # By strength
            strength = obligation.get("strength", "medium")
            categorized["by_strength"][strength].append(obligation)
        
        return categorized
    
    def calculate_confidence_score(self, match):
        """
        Calculate a confidence score for a match.
        
        Determines how confidently the system believes this is a true obligation.
        
        Args:
            match (dict): A potential obligation match
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Start with base confidence
        confidence = 0.5
        
        # Strong obligation keywords increase confidence
        text = match["text"].lower()
        if any(word in text for word in ["must", "shall", "required", "mandated"]):
            confidence += 0.2
        elif any(word in text for word in ["should", "recommended"]):
            confidence += 0.1
        
        # Negative modifiers decrease confidence
        if any(phrase in text for phrase in ["not required", "not mandatory", "no need", "optional"]):
            confidence -= 0.3
        
        # Length of the sentence
        words = len(text.split())
        if 10 <= words <= 50:  # Typical length for obligations
            confidence += 0.1
        elif words > 50:  # Very long sentences might be complex but less precise
            confidence -= 0.1
        
        # If entities are present, especially organizations (usually regulators)
        entities = match.get("entities", {})
        if "ORG" in entities:
            confidence += 0.05
        
        # Presence of specific action verbs
        action_verbs = ["ensure", "provide", "maintain", "establish", "implement", "comply"]
        if any(verb in text for verb in action_verbs):
            confidence += 0.05
        
        # Clamp confidence between 0 and 1
        return max(0.0, min(1.0, confidence))

    def explain_match_reasoning(self, obligation, policy_match):
        """
        Provide explanation of why a match was made.
        
        Args:
            obligation (dict): The extracted obligation
            policy_match (dict): The matched policy or requirement
            
        Returns:
            dict: Explanation of the match with highlighted key points
        """
        explanation = {
            "match_strength": "unknown",
            "key_factors": [],
            "shared_keywords": [],
            "summary": ""
        }
        
        # Get text from both items
        obligation_text = obligation["text"].lower()
        policy_text = policy_match["text"].lower() if "text" in policy_match else ""
        
        # Find shared keywords
        obligation_keywords = set(obligation.get("keywords", []))
        policy_keywords = set(policy_match.get("keywords", []))
        shared_keywords = obligation_keywords.intersection(policy_keywords)
        explanation["shared_keywords"] = list(shared_keywords)
        
        # Calculate match strength
        if len(shared_keywords) >= 3:
            explanation["match_strength"] = "strong"
            explanation["key_factors"].append("Multiple shared keywords")
        elif len(shared_keywords) > 0:
            explanation["match_strength"] = "moderate"
            explanation["key_factors"].append("Some shared keywords")
        else:
            explanation["match_strength"] = "weak"
        
        # Check for shared entities
        obligation_entities = obligation.get("entities", {})
        policy_entities = policy_match.get("entities", {})
        shared_entity_types = set(obligation_entities.keys()).intersection(set(policy_entities.keys()))
        if shared_entity_types:
            for entity_type in shared_entity_types:
                obl_ents = set(obligation_entities[entity_type])
                pol_ents = set(policy_entities[entity_type])
                shared_ents = obl_ents.intersection(pol_ents)
                if shared_ents:
                    explanation["key_factors"].append(f"Shared {entity_type} entities: {', '.join(shared_ents)}")
                    explanation["match_strength"] = "strong"
        
        # Check for similar obligation verbs
        obligation_verbs = ["must", "shall", "required", "should", "may", "can"]
        for verb in obligation_verbs:
            if verb in obligation_text and verb in policy_text:
                explanation["key_factors"].append(f"Both contain obligation verb: '{verb}'")
        
        # Generate summary
        if explanation["match_strength"] == "strong":
            explanation["summary"] = "Strong match based on shared keywords and entities"
        elif explanation["match_strength"] == "moderate":
            explanation["summary"] = "Moderate match with some shared elements"
        else:
            explanation["summary"] = "Weak match with few similarities"
            
        return explanation
            
    def _calculate_complexity(self, span):
        """Calculate the complexity of a sentence based on its structure."""
        # Basic metrics for complexity
        word_count = len([token for token in span])
        clause_count = len([token for token in span if token.dep_ == "ROOT"]) 
        clause_count += len([token for token in span if token.dep_ == "ccomp" or token.dep_ == "xcomp"])
        
        # Count depth of dependency tree
        max_depth = 0
        for token in span:
            depth = 1
            current = token
            while current.head != current:  # While not at root
                current = current.head
                depth += 1
            max_depth = max(max_depth, depth)
        
        # Calculate complexity score (1-10)
        complexity = 1
        if word_count > 15: complexity += 1
        if word_count > 25: complexity += 1
        if word_count > 40: complexity += 1
        if clause_count > 1: complexity += 1
        if clause_count > 2: complexity += 2
        if max_depth > 5: complexity += 1
        if max_depth > 8: complexity += 2
        return min(10, complexity)  # Cap at 10
