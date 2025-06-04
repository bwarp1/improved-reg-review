"""Load and preprocess internal policy documents."""

import glob
import io
import logging
import os
from pathlib import Path
from typing import Dict, List, Union, Optional

import fitz  # PyMuPDF
from bs4 import BeautifulSoup


class PolicyLoader:
    """Load and preprocess internal policy documents."""
    
    def __init__(self):
        """Initialize the policy loader."""
        self.logger = logging.getLogger(__name__)
        
        # Map file extensions to handler methods (OCP from SOLID)
        self.file_handlers = {
            ".pdf": self._extract_text_from_pdf,
            ".txt": self._extract_text_from_txt,
            ".html": self._extract_text_from_html_file,
            ".htm": self._extract_text_from_html_file,
        }
        
        # Try to add DOCX support if available
        try:
            import docx
            self.file_handlers[".docx"] = self._extract_text_from_docx
            self.docx_available = True
        except ImportError:
            self.docx_available = False
            self.logger.warning("python-docx module not available, DOCX files will be skipped")
    
    def load_policies(self, policy_dir: str) -> Dict[str, str]:
        """Load all policy documents from a directory.
        
        Args:
            policy_dir: Directory containing policy documents
            
        Returns:
            Dictionary mapping policy names to their text content
        """
        policy_dir_path = Path(policy_dir)
        if not policy_dir_path.exists():
            self.logger.error(f"Policy directory not found: {policy_dir}")
            return {}
        
        self.logger.info(f"Loading policies from {policy_dir}")
        policies = {}
        
        # Process all files in directory recursively
        for file_path in policy_dir_path.glob("**/*"):
            if not file_path.is_file():
                continue
                
            extension = file_path.suffix.lower()
            if extension in self.file_handlers:
                try:
                    self.logger.debug(f"Loading policy: {file_path}")
                    policy_text = self.file_handlers[extension](file_path)
                    if policy_text:
                        policies[str(file_path.relative_to(policy_dir_path))] = policy_text
                except Exception as e:
                    self.logger.error(f"Error loading policy {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(policies)} policy documents")
        return policies
    
    def _extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        try:
            pdf_doc = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text += page.get_text()
                
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            
        return text
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content.
        
        Args:
            html_content: HTML content as string
            
        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove scripts, styles, etc.
        for tag in soup(["script", "style", "meta", "link"]):
            tag.extract()
            
        return soup.get_text(separator=" ", strip=True)
    
    def _extract_text_from_docx(self, docx_path: Union[str, Path]) -> str:
        """Extract text from a DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text content
        """
        import docx
        
        text = []
        doc = docx.Document(docx_path)
        
        for para in doc.paragraphs:
            text.append(para.text)
            
        return "\n".join(text)
    
    def _extract_text_from_txt(self, txt_path: Union[str, Path]) -> str:
        """Extract text from a TXT file.
        Args:
            txt_path: Path to TXT file
        Returns:
            Text content of the file
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error extracting text from TXT file: {e}")
            return ""
            
    def _extract_text_from_html_file(self, html_path: Union[str, Path]) -> str:
        """Extract text from an HTML file.
        
        Args:
            html_path: Path to HTML file
            
        Returns:
            Extracted text content
        """
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return self._extract_text_from_html(html_content)
        except Exception as e:
            self.logger.error(f"Error extracting text from HTML file: {e}")
            return ""
    
    def preprocess_policy(self, policy_text: str) -> str:
        """Preprocess policy text for analysis.
        
        Args:
            policy_text: Raw policy text
            
        Returns:
            Preprocessed policy text
        """
        # Remove excessive whitespace
        policy_text = " ".join(policy_text.split())
        
        # Replace common Unicode characters
        replacements = {
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--"
        }
        for old, new in replacements.items():
            policy_text = policy_text.replace(old, new)
            
        return policy_text
    
    def split_policy_into_sections(self, policy_text: str) -> Dict[str, str]:
        """Split a policy document into sections.
        
        Args:
            policy_text: Policy text content
            
        Returns:
            Dictionary mapping section identifiers to section text
        """
        # Basic section splitting based on numbered sections or headings
        # This is a simplistic implementation, would need refinement for production
        sections = {}
        
        # Try to find section headers (e.g., "Section 1.2", "Article 5", etc.)
        section_pattern = r'(?:Section|Article|§)\s+(\d+(?:\.\d+)?)\s*[:\.]\s*([^\n]+)'
        
        import re
        section_matches = re.finditer(section_pattern, policy_text)
        
        last_pos = 0
        last_section_id = "intro"
        
        for match in section_matches:
            # Extract text from last position to current match as a section
            if match.start() > last_pos:
                section_text = policy_text[last_pos:match.start()].strip()
                if section_text:
                    sections[last_section_id] = section_text
            
            # Extract section number and title
            section_id = match.group(1)
            section_title = match.group(2).strip()
            last_section_id = f"section_{section_id}"
            last_pos = match.end()
        
        # Add the final section
        if last_pos < len(policy_text):
            sections[last_section_id] = policy_text[last_pos:].strip()
            
        # If no sections found, just use sentences as sections
        if not sections:
            sentences = re.split(r'(?<=[.!?])\s+', policy_text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    sections[f"sent_{i+1}"] = sentence.strip()
        
        return sections
