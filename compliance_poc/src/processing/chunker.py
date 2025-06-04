"""Module for handling large documents through advanced chunking strategies."""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import concurrent.futures
from functools import partial


class DocumentChunker:
    """
    Handles large documents by splitting them into manageable chunks while
    preserving context and structure for accurate analysis.
    """
    
    def __init__(self, max_chunk_size: int = 100000, 
                overlap_size: int = 5000,
                preserve_sections: bool = True):
        """
        Initialize the document chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap_size: Overlap between chunks to preserve context
            preserve_sections: Whether to preserve section boundaries
        """
        self.logger = logging.getLogger(__name__)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_sections = preserve_sections
        
        # Section header patterns for identifying structure
        self.section_patterns = [
            r'^(?:Section|Article|§)\s+\d+[\.\d]*\s*[:-]',
            r'^\d+[\.\d]*\s+[A-Z][a-zA-Z\s]+$',
            r'^[IVXLC]+\.\s+[A-Z][a-zA-Z\s]+$',
            r'^[A-Z][A-Z\s\-]+:?$'  # All-caps headers
        ]
    
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Split document into manageable chunks while preserving structure.
        
        Args:
            text: The full document text
            
        Returns:
            List of chunks with metadata (position, context)
        """
        if len(text) <= self.max_chunk_size:
            self.logger.info("Document within size limits, no chunking needed")
            return [{
                "text": text, 
                "index": 0,
                "start_pos": 0,
                "end_pos": len(text),
                "section": None
            }]
        
        self.logger.info(f"Chunking document of {len(text)} characters")
        
        # Find all section boundaries for smart splitting
        sections = self._identify_sections(text)
        
        # If no clear sections or not preserving sections, use paragraph-based chunking
        if not sections or not self.preserve_sections:
            return self._chunk_by_paragraphs(text)
        
        # Use section-aware chunking
        return self._chunk_by_sections(text, sections)
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify section boundaries in the document.
        
        Args:
            text: The document text
            
        Returns:
            List of section dictionaries with positions and headers
        """
        sections = []
        combined_pattern = '|'.join(self.section_patterns)
        
        # Find all potential section headers
        for match in re.finditer(combined_pattern, text, re.MULTILINE):
            sections.append({
                "start": match.start(),
                "header": match.group(0).strip(),
                "level": self._determine_section_level(match.group(0))
            })
        
        # Add document end as a boundary
        if sections:
            for i in range(len(sections) - 1):
                sections[i]["end"] = sections[i+1]["start"]
            sections[-1]["end"] = len(text)
            
        return sections
    
    def _determine_section_level(self, header: str) -> int:
        """Determine the hierarchical level of a section header."""
        # Simple heuristics for hierarchy level
        if re.match(r'^(?:Section|Article|§)\s+\d+\s*[:-]', header):
            return 1  # Top level
        elif re.match(r'^\d+\.\d+', header):
            return 2  # Second level (e.g., 1.2)
        elif re.match(r'^\d+\.\d+\.\d+', header):
            return 3  # Third level (e.g., 1.2.3)
        return 1  # Default to top level
    
    def _chunk_by_sections(self, text: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks based on section boundaries."""
        chunks = []
        current_size = 0
        start_idx = 0
        current_section = None
        chunk_index = 0
        
        for section in sections:
            section_size = section["end"] - section["start"]
            
            # If adding this section exceeds chunk size, create a new chunk
            if current_size + section_size > self.max_chunk_size and current_size > 0:
                # Create chunk up to this section
                chunks.append({
                    "text": text[start_idx:section["start"]],
                    "index": chunk_index,
                    "start_pos": start_idx,
                    "end_pos": section["start"],
                    "section": current_section
                })
                chunk_index += 1
                start_idx = section["start"]
                current_size = 0
            
            current_size += section_size
            current_section = section["header"]
            
            # If this single section exceeds chunk size, split it
            if section_size > self.max_chunk_size:
                self.logger.warning(f"Section '{section['header']}' exceeds max chunk size, splitting within section")
                section_chunks = self._split_large_section(
                    text[section["start"]:section["end"]], 
                    section["start"],
                    section["header"],
                    chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
                start_idx = section["end"]
                current_size = 0
        
        # Add final chunk if there's remaining content
        if start_idx < len(text):
            chunks.append({
                "text": text[start_idx:],
                "index": chunk_index,
                "start_pos": start_idx,
                "end_pos": len(text),
                "section": current_section
            })
            
        return chunks
        
    def _split_large_section(self, text: str, offset: int, section: str, start_index: int) -> List[Dict[str, Any]]:
        """Split an oversized section into paragraph-based chunks."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_start = 0
        chunk_index = start_index
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size and current_chunk:
                # Add current chunk
                chunks.append({
                    "text": current_chunk,
                    "index": chunk_index,
                    "start_pos": offset + chunk_start,
                    "end_pos": offset + chunk_start + len(current_chunk),
                    "section": section
                })
                chunk_index += 1
                current_chunk = para
                chunk_start += len(current_chunk) + 2  # +2 for the paragraph break
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "index": chunk_index,
                "start_pos": offset + chunk_start,
                "end_pos": offset + len(current_chunk),
                "section": section
            })
            
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split document into chunks based on paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, create a new chunk
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size and current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "index": chunk_index,
                    "start_pos": chunk_start,
                    "end_pos": chunk_start + len(current_chunk),
                    "section": None
                })
                chunk_index += 1
                current_chunk = para
                chunk_start += len(current_chunk) + 2
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "index": chunk_index,
                "start_pos": chunk_start,
                "end_pos": chunk_start + len(current_chunk),
                "section": None
            })
            
        return chunks


class ParallelProcessor:
    """
    Handles parallel processing of document chunks or multiple regulations
    to improve performance on multi-core systems.
    """
    
    def __init__(self, max_workers: Optional[int] = None, 
                show_progress: bool = True):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads
            show_progress: Whether to display a progress bar
        """
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.show_progress = show_progress
    
    def process_chunks(self, chunks: List[Dict[str, Any]], 
                      process_func: callable, **kwargs) -> List[Any]:
        """
        Process document chunks in parallel.
        
        Args:
            chunks: List of document chunks to process
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments to pass to the process function
            
        Returns:
            Combined results from all chunks
        """
        if not chunks:
            return []
            
        self.logger.info(f"Processing {len(chunks)} chunks in parallel")
        results = []
        
        # Create a partial function with the kwargs
        chunk_processor = partial(process_func, **kwargs)
        
        # Use ThreadPoolExecutor for I/O-bound operations or ProcessPoolExecutor for CPU-bound
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Map the function to all chunks
            futures = {executor.submit(chunk_processor, chunk): i for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            if self.show_progress:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks"):
                    results.append((futures[future], future.result()))
            else:
                for future in concurrent.futures.as_completed(futures):
                    results.append((futures[future], future.result()))
        
        # Sort results by original chunk index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def merge_chunk_results(self, results: List[Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from processed chunks into a single coherent result.
        
        Args:
            results: List of results from chunk processing
            chunks: Original chunks with metadata
            
        Returns:
            Consolidated results
        """
        if not results:
            return {}
            
        self.logger.info("Merging results from parallel processing")
        
        # Merge obligations from all chunks
        all_obligations = []
        for i, chunk_result in enumerate(results):
            chunk_obligations = chunk_result.get('obligations', [])
            
            # Adjust IDs and positions to avoid duplicates
            for obligation in chunk_obligations:
                # Adjust position based on chunk start position
                if 'start_pos' in obligation:
                    obligation['start_pos'] += chunks[i].get('start_pos', 0)
                if 'end_pos' in obligation:
                    obligation['end_pos'] += chunks[i].get('start_pos', 0)
                
                # Ensure unique IDs by prefixing with chunk index
                if 'id' in obligation and not obligation['id'].startswith(f"C{i+1}-"):
                    obligation['id'] = f"C{i+1}-{obligation['id']}"
                
                # Add section information if available
                if 'section' not in obligation and chunks[i].get('section'):
                    obligation['section'] = chunks[i].get('section')
                
                all_obligations.append(obligation)
        
        # Deduplicate obligations that may appear in overlapping chunks
        deduplicated = self._deduplicate_obligations(all_obligations)
        
        # Combine all other result properties
        combined_result = {
            'obligations': deduplicated
        }
        
        # Add any other fields from the results
        for field in results[0].keys():
            if field != 'obligations':
                combined_result[field] = results[0][field]
        
        return combined_result
    
    def _deduplicate_obligations(self, obligations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate obligations from overlapping chunks."""
        # Simple deduplication based on text similarity
        unique_obligations = []
        seen_texts = set()
        
        for obligation in obligations:
            text = obligation.get('text', '')
            text_hash = hash(text)
            
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_obligations.append(obligation)
        
        return unique_obligations
