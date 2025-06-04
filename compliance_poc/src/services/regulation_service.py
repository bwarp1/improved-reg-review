import os
import logging
from typing import List, Dict, Any, Optional
from ..processing.chunker import DocumentChunker
from ..utils.parallel_processor import ParallelProcessor

logger = logging.getLogger(__name__)

class RegulationService:
    def __init__(self, nlp_service=None, db_service=None):
        # ...existing code...
        self.document_chunker = DocumentChunker(max_chunk_size=4000, overlap_size=200)
        self.parallel_processor = ParallelProcessor(max_workers=os.cpu_count() or 4)
        # ...existing code...
    
    def process_regulation(self, regulation_text: str, regulation_id: str = None) -> Dict[str, Any]:
        """Process a single regulation and extract relevant information."""
        # ...existing code...
        # For very large regulations, use chunking
        if len(regulation_text) > 10000:  # Threshold for chunking
            chunks = self.document_chunker.chunk_document(regulation_text)
            logger.info(f"Regulation split into {len(chunks)} chunks for processing")
            
            # Define a function to process a single chunk
            def process_chunk(chunk):
                chunk_result = self.nlp_service.extract_entities(chunk['text'])
                return chunk_result
            
            # Process chunks in parallel
            chunk_results = self.parallel_processor.process_batch(
                chunks, process_chunk, use_threads=True
            )
            
            # Merge results
            result = self._merge_chunk_results(chunk_results)
        else:
            # Process normally for smaller regulations
            result = self.nlp_service.extract_entities(regulation_text)
        
        # ...existing code...
        return result

    def _merge_chunk_results(self, chunk_results: List[Dict]) -> Dict[str, Any]:
        """Merge results from multiple chunks into a single result."""
        merged_result = {
            'entities': [],
            'sentiment': 0,
            'categories': {},
            'requirements': []
        }
        
        if not chunk_results:
            return merged_result
            
        for result in chunk_results:
            merged_result['entities'].extend(result.get('entities', []))
            merged_result['sentiment'] += result.get('sentiment', 0)
            
            # Merge categories
            for category, value in result.get('categories', {}).items():
                if category in merged_result['categories']:
                    merged_result['categories'][category] += value
                else:
                    merged_result['categories'][category] = value
            
            merged_result['requirements'].extend(result.get('requirements', []))
        
        # Normalize sentiment
        merged_result['sentiment'] /= len(chunk_results)
        
        # Deduplicate entities and requirements
        merged_result['entities'] = list({e['text']: e for e in merged_result['entities']}.values())
        merged_result['requirements'] = list({r['text']: r for r in merged_result['requirements']}.values())
        
        return merged_result
    
    def process_regulations_batch(self, regulations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process multiple regulations in parallel."""
        logger.info(f"Processing batch of {len(regulations)} regulations")
        
        # Define function to process a single regulation
        def process_single_reg(reg):
            try:
                return self.process_regulation(reg['text'], reg.get('id'))
            except Exception as e:
                logger.error(f"Error processing regulation {reg.get('id')}: {str(e)}")
                return {'error': str(e), 'regulation_id': reg.get('id')}
        
        # Use parallel processor to process all regulations
        results = self.parallel_processor.process_batch(
            regulations, process_single_reg, use_threads=False
        )
        
        return results
