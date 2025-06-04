"""
NLP module for extracting regulatory obligations.
"""

from compliance_poc.src.nlp.extractor import ObligationExtractor
from compliance_poc.src.nlp.domain_models import DomainSpecificProcessor

__all__ = ['ObligationExtractor', 'DomainSpecificProcessor']
