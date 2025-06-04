"""
Utility for parallel processing of regulatory analysis tasks.
"""
import concurrent.futures
import logging
from typing import Callable, Dict, List, Any, Union
import time

logger = logging.getLogger(__name__)

class ParallelProcessor:
    def __init__(self, max_workers: int = 4, timeout: int = 300):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            timeout: Maximum time in seconds to wait for a task
        """
        self.max_workers = max_workers
        self.timeout = timeout

    def process_batch(self, 
                      items: List[Any], 
                      process_func: Callable[[Any], dict],
                      use_threads: bool = True) -> List[dict]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function that processes a single item
            use_threads: Whether to use threads (True) or processes (False)
            
        Returns:
            List of results from processing each item
        """
        results = []
        errors = []
        
        executor_class = (concurrent.futures.ThreadPoolExecutor if use_threads 
                          else concurrent.futures.ProcessPoolExecutor)
        
        start_time = time.time()
        logger.info(f"Starting parallel processing of {len(items)} items with {self.max_workers} workers")
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(process_func, item): item for item in items}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_item, timeout=self.timeout):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Successfully processed item: {getattr(item, 'id', str(item)[:50])}")
                except Exception as exc:
                    errors.append({
                        'item': getattr(item, 'id', str(item)[:50]),
                        'error': str(exc)
                    })
                    logger.error(f"Error processing item {getattr(item, 'id', str(item)[:50])}: {exc}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed parallel processing in {elapsed_time:.2f} seconds. "
                   f"Successful: {len(results)}, Errors: {len(errors)}")
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during parallel processing.")
            
        return results

    def chunk_and_process(self,
                          large_items: List[Any],
                          chunker_func: Callable[[Any], List[Any]],
                          process_func: Callable[[Any], dict],
                          merge_func: Callable[[List[dict]], dict],
                          use_threads: bool = True) -> List[dict]:
        """
        Chunk large items, process in parallel, then merge results.
        
        Args:
            large_items: List of large items to process
            chunker_func: Function to split an item into chunks
            process_func: Function to process a single chunk
            merge_func: Function to merge chunk results back together
            use_threads: Whether to use threads or processes
            
        Returns:
            List of merged results
        """
        all_chunks = []
        chunk_map = {}  # Maps chunk to original item index
        
        # Chunk all items
        for idx, item in enumerate(large_items):
            chunks = chunker_func(item)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_map[id(chunk)] = idx
        
        # Process all chunks in parallel
        chunk_results = self.process_batch(all_chunks, process_func, use_threads)
        
        # Group results by original item
        grouped_results = {}
        for chunk, result in zip(all_chunks, chunk_results):
            item_idx = chunk_map[id(chunk)]
            if item_idx not in grouped_results:
                grouped_results[item_idx] = []
            grouped_results[item_idx].append(result)
        
        # Merge results for each original item
        final_results = []
        for idx in range(len(large_items)):
            if idx in grouped_results:
                merged_result = merge_func(grouped_results[idx])
                final_results.append(merged_result)
            else:
                # If we have no results for this item, add None
                final_results.append(None)
        
        return final_results
