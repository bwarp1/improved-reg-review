"""
Client for interacting with the Regulations.gov API.
"""

import os
import json
import logging
import time
import io
from pathlib import Path
from urllib.parse import urljoin
from typing import Dict, List, Optional, Union
from functools import wraps
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_hour: int = 1000):
        self.requests_per_hour = requests_per_hour
        self.request_times = []
        self.logger = logging.getLogger(__name__)

    def wait_if_needed(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Clean up old requests
        self.request_times = [t for t in self.request_times if t > hour_ago]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_hour:
            wait_time = self.request_times[0] - hour_ago
            self.logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time + 0.1)  # Add small buffer
            return self.wait_if_needed()
        
        # Add current request time
        self.request_times.append(current_time)

class APICache:
    """Cache handler for API responses."""
    
    def __init__(self, cache_dir: Path, max_age: int = 86400):
        self.cache_dir = cache_dir
        self.max_age = max_age  # Cache expiry in seconds
        self.logger = logging.getLogger(__name__)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def get(self, key: str) -> Optional[str]:
        """Get cached content if it exists and is not expired."""
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None

        # Check cache age
        age = time.time() - cache_file.stat().st_mtime
        if age > self.max_age:
            self.logger.debug(f"Cache expired for {key}")
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                self.logger.debug(f"Cache hit for {key}")
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading cache for {key}: {e}")
            return None

    def set(self, key: str, content: str) -> None:
        """Store content in cache."""
        cache_file = self._get_cache_path(key)
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.debug(f"Cached content for {key}")
        except Exception as e:
            self.logger.error(f"Error caching content for {key}: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        # Create subdirectories based on key parts for better organization
        parts = key.split('-')
        if len(parts) >= 3:
            return self.cache_dir / parts[0] / '-'.join(parts[1:3]) / f"{'-'.join(parts[3:])}.txt"
        return self.cache_dir / f"{key}.txt"

class RegulationsAPI:
    """Client for fetching data from the Regulations.gov API."""
    
    def __init__(self, api_key=None, use_demo_data=False, cache_max_age=86400):
        """
        Initialize the Regulations.gov API client.
        
        Args:
            api_key: API key for Regulations.gov
            use_demo_data: Whether to use demo data instead of real API calls
            cache_max_age: Maximum age of cached content in seconds (default 24 hours)
        """
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://api.regulations.gov/v4/"
        self.use_demo_data = use_demo_data
        
        # Get API key from parameter, env var, or use demo key
        self.api_key = api_key or os.environ.get("REGULATIONS_API_KEY", "DEMO_KEY")
        
        # Setup HTTP headers for API requests
        self.headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json"
        }
        
        # Initialize cache handler
        self.cache = APICache(Path("cache/regulations"), cache_max_age)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Setup session with retry mechanism
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,  # number of retries
            backoff_factor=0.5,  # wait 0.5s * (2 ^ (retry - 1)) between retries
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger.info(f"RegulationsAPI initialized, use_demo_data={use_demo_data}, API key={'provided' if api_key else 'from environment or default'}")
    
    def search_by_docket(self, docket_id: str, limit: int = 10) -> List[Dict]:
        """
        Search for documents within a specific docket.
        
        Args:
            docket_id: The docket ID to search
            limit: Maximum number of results to return
            
        Returns:
            List of document metadata dictionaries
        """
        if self.use_demo_data:
            self.logger.info(f"Demo mode: Returning sample documents for docket {docket_id}")
            return [{
                "id": "DEMO-DOC-2023-0001",
                "type": "documents",
                "attributes": {
                    "title": "Sample Regulation Document",
                    "agency": "EPA",
                    "docketId": docket_id,
                    "documentType": "Rule"
                }
            }]
        
        # Check rate limit before making API call
        self._enforce_rate_limit()
        
        # Build the API request
        endpoint = "documents"
        params = {
            "filter[docketId]": docket_id,
            "sort": "-postedDate",
            "page[size]": limit
        }
        
        try:
            self.logger.info(f"Searching for documents in docket: {docket_id}")
            response = self._make_api_request(endpoint, params)
            
            documents = response.get("data", [])
            self.logger.info(f"Found {len(documents)} documents for docket {docket_id}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error searching docket {docket_id}: {e}")
            return []
    
    def get_document_content(self, document_id: str) -> str:
        """
        Get the text content of a document.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document text content
        """
        # Check cache first
        safe_id = self._sanitize_document_id(document_id)
        cached_content = self.cache.get(safe_id)
        if cached_content:
            return cached_content
        
        # In demo mode, return sample content
        if self.use_demo_data:
            self.logger.info(f"Demo mode: Returning sample content for document {document_id}")
            sample_content = "Section 1.1: Organizations must maintain records for at least 5 years.\n"
            sample_content += "Section 1.2: All employees shall receive annual security training.\n"
            sample_content += "Section 1.3: Companies are required to submit quarterly compliance reports."
            
            # Cache the sample content
            self._cache_document(document_id, sample_content)
            return sample_content
        
        try:
            # First get document metadata
            self.logger.info(f"Fetching document metadata: {document_id}")
            endpoint = f"documents/{document_id}"
            
            # Make API request with rate limiting
            response = self._make_api_request(endpoint)
            
            # Extract document content through different methods
            doc_content = self._extract_document_content(response)
            
            # Cache the content if found
            if doc_content:
                self._cache_document(document_id, doc_content)
                
            return doc_content or ""
            
        except Exception as e:
            self.logger.error(f"Error getting document content for {document_id}: {e}")
            return ""
    
    def _extract_document_content(self, doc_response: Dict) -> str:
        """Extract content from document response using various methods."""
        if not doc_response or "data" not in doc_response:
            return ""
            
        doc_data = doc_response.get("data", {})
        attributes = doc_data.get("attributes", {})
        content = ""
        
        # Try to get content directly from the response
        if "fileContent" in attributes:
            return attributes["fileContent"]
        
        # Try HTML content
        html_url = attributes.get("htmlUrl")
        if html_url:
            self.logger.info(f"Retrieving HTML content from URL: {html_url}")
            content = self._download_html_content(html_url)
            if content:
                return content
            
        # Try PDF content
        if "fileFormats" in attributes and "pdf" in attributes["fileFormats"]:
            pdf_url = attributes["fileFormats"]["pdf"]
            self.logger.info(f"Retrieving PDF content from URL")
            content = self._download_pdf_content(pdf_url)
            if content:
                return content
                
        # Check for attachments
        if "included" in doc_response:
            for item in doc_response["included"]:
                if item.get("type") == "attachments" and "attributes" in item:
                    attachment = item.get("attributes", {})
                    formats = attachment.get("fileFormats", {})
                    
                    # Try PDF attachment
                    if "pdf" in formats:
                        self.logger.info(f"Retrieving PDF attachment")
                        content = self._download_pdf_content(formats["pdf"])
                        if content:
                            return content
                            
                    # Try HTML attachment
                    if "html" in formats:
                        self.logger.info(f"Retrieving HTML attachment")
                        content = self._download_html_content(formats["html"])
                        if content:
                            return content
        
        self.logger.warning("No content found in document")
        return content
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make an API request with error handling and retry logic."""
        url = urljoin(self.base_url, endpoint)
        params = params or {}
        
        # Check rate limits before making request
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            if status_code == 401:
                self.logger.error("API key unauthorized. Check your API key.")
            elif status_code == 403:
                self.logger.error("API key forbidden. Check your permissions.")
            elif status_code == 429:
                # Retry logic is handled by the retry strategy in session
                self.logger.warning("Rate limit exceeded. Retrying with backoff...")
            
            self.logger.error(f"HTTP error {status_code}: {str(e)}")
            raise
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            raise
    
    def _download_pdf_content(self, url: str) -> str:
        """
        Download and extract text from a PDF URL.
        
        Args:
            url: URL to the PDF document
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text from PDF using PyMuPDF
            pdf_data = io.BytesIO(response.content)
            pdf = fitz.open(stream=pdf_data, filetype="pdf")
            
            text = ""
            for page_num in range(len(pdf)):
                text += pdf[page_num].get_text()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error downloading PDF from {url}: {e}")
            return ""
    
    def _download_html_content(self, url: str) -> str:
        """
        Download and extract text from an HTML URL.
        
        Args:
            url: URL to the HTML document
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract text from HTML using BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "meta", "link", "header", "footer"]):
                element.decompose()
                
            return soup.get_text(separator="\n", strip=True)
            
        except Exception as e:
            self.logger.error(f"Error downloading HTML from {url}: {e}")
            return ""
    
    def _cache_document(self, document_id: str, content: str) -> None:
        """
        Cache document content to file using a standardized naming scheme.
        
        Args:
            document_id: Document ID from the API
            content: Text content to cache
        """
        try:
            # Sanitize document ID for file system use
            safe_id = self._sanitize_document_id(document_id)
            
            # Create subdirectory based on first part of the ID for better organization
            # For example, EPA-HQ-OAR-2021-0257 becomes /cache/regulations/EPA-HQ-OAR/2021-0257.txt
            id_parts = safe_id.split('-')
            if len(id_parts) >= 3:
                # Create agency/docket-specific subdirectory
                agency_docket = '-'.join(id_parts[:3])
                doc_subdir = self.cache_dir / agency_docket
                doc_subdir.mkdir(exist_ok=True, parents=True)
                
                # Use remaining parts for filename
                filename = '-'.join(id_parts[3:]) if len(id_parts) > 3 else safe_id
                cache_file = doc_subdir / f"{filename}.txt"
            else:
                # Fallback for IDs that don't match expected format
                cache_file = self.cache_dir / f"{safe_id}.txt"
            
            # Save content with metadata header
            with open(cache_file, "w", encoding="utf-8") as f:
                # Add metadata header for easier identification and tracking
                f.write(f"# Document ID: {document_id}\n")
                f.write(f"# Cached: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Source: Regulations.gov API\n")
                f.write("#" + "-" * 50 + "\n\n")
                f.write(content)
                
            self.logger.debug(f"Cached content for document {document_id} at {cache_file}")
            return cache_file
            
        except Exception as e:
            self.logger.error(f"Error caching document {document_id}: {e}")
            return None
    
    def _sanitize_document_id(self, document_id: str) -> str:
        """
        Sanitize document ID to create a valid and consistent filename.
        
        Args:
            document_id: Raw document ID from the API
            
        Returns:
            Sanitized ID safe for filesystem use
        """
        # Replace unsafe characters
        safe_id = document_id.replace('/', '-').replace('\\', '-')
        safe_id = safe_id.replace(':', '_').replace('*', '_')
        safe_id = safe_id.replace('?', '_').replace('"', '_')
        safe_id = safe_id.replace('<', '_').replace('>', '_')
        safe_id = safe_id.replace('|', '_').replace(' ', '_')
        
        return safe_id
    
    def _enforce_rate_limit(self) -> None:
        """
        Check and enforce API rate limiting.
        """
        # Clean up old requests
        current_time = time.time()
        hour_ago = current_time - 3600
        self.request_times = [t for t in self.request_times if t > hour_ago]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            self.logger.warning(f"Rate limit reached ({self.rate_limit}/hour). Waiting...")
            time.sleep(5)  # Wait 5 seconds before next request
            
        # Add current request time
        self.request_times.append(time.time())

    def search_documents(self, published_since=None, search_terms=None, **kwargs):
        """
        Search for documents with optional filtering.
        
        Args:
            published_since: Date string to filter documents published since
            search_terms: Terms to search for in documents
            **kwargs: Additional filter parameters for the API
            
        Returns:
            list: Documents matching the search criteria
        """
        # Apply rate limiting before making API call
        self._enforce_rate_limit()
        
        # Build search params (DRY principle - reuse existing code structure)
        params = kwargs.copy()
        
        if published_since:
            params['filter[postedDate][ge]'] = published_since
        
        if search_terms:
            # If search_terms is a list, join them
            if isinstance(search_terms, list):
                params['filter[searchTerm]'] = ' '.join(search_terms)
            else:
                params['filter[searchTerm]'] = search_terms
        
        # Use demo data if configured (KISS principle)
        if self.use_demo_data:
            self.logger.info(f"Demo mode: Returning sample search results")
            return self._get_demo_search_results(params)
        
        try:
            # Make the API request using our existing method (DRY principle)
            response = self._make_api_request("documents", params)
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []
            
    def _get_demo_search_results(self, params=None):
        """Return demo search results based on the search params."""
        from datetime import datetime  # Add missing import
        
        # Simple demo data generator (YAGNI principle - no need for complex demo data)
        results = [{
            "id": f"DEMO-DOC-2023-{i}",
            "type": "documents",
            "attributes": {
                "title": f"Sample Regulation Document {i}",
                "agency": "EPA",
                "documentType": "Rule",
                "postedDate": datetime.now().strftime("%Y-%m-%d")
            }
        } for i in range(1, 6)]
        
        if params and 'filter[searchTerm]' in params:
            # Filter demo results by search term
            term = params['filter[searchTerm]'].lower()
            results = [r for r in results if term in r["attributes"]["title"].lower()]
            
        return results
