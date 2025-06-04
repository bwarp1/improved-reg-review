"""
Processing module for handling large documents and parallel execution.
"""

from compliance_poc.src.processing.chunker import DocumentChunker, ParallelProcessor

__all__ = ['DocumentChunker', 'ParallelProcessor']
