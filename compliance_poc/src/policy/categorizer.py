import os
import re
from typing import Dict, List, Optional
from pathlib import Path

class PolicyLoader:
    """Loads and categorizes policy documents for compliance analysis."""
    
    def __init__(self, default_categories=None):
        """
        Initialize the policy loader with default categories.
        
        Args:
            default_categories: Dict of categories to initialize with
        """
        # Initialize with standard regulatory categories
        self.categories = default_categories or {
            "privacy": [],
            "financial": [],
            "environmental": [],
            "healthcare": [],
            "general": []  # Default category
        }
        
        # Regex patterns for category detection (extracted for Single Responsibility)
        self._category_patterns = {
            "privacy": r"privacy|data\s+protection|gdpr|ccpa|personal\s+data|confidential",
            "financial": r"financial|banking|investment|payment|accounting|tax|finance|budget",
            "environmental": r"environmental|sustainability|emission|waste|ecology|pollution|climate",
            "healthcare": r"health|medical|patient|hipa+|treatment|clinical|care\s+provider",
        }
    
    def categorize_policies(self, policy_dir: str) -> Dict[str, List[str]]:
        """
        Group policies by category based on metadata or content.
        
        Args:
            policy_dir: Directory containing policy files
            
        Returns:
            Dictionary mapping categories to lists of policy file paths
        """
        # Reset categories to empty lists
        categories = {k: [] for k in self.categories.keys()}
        
        # Verify directory exists (KISS - simple validation)
        dir_path = Path(policy_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            return categories
            
        # Iterate through files and categorize
        for file_path in dir_path.iterdir():
            if not self._is_policy_file(file_path.name):
                continue
                
            # Determine category from metadata or content analysis
            category = self._detect_policy_category(file_path)
            categories[category].append(str(file_path))
        
        return categories
    
    def _is_policy_file(self, filename: str) -> bool:
        """
        Check if a file is a policy file based on its extension.
        
        Args:
            filename: Name of file to check
            
        Returns:
            Boolean indicating if file has valid policy extension
        """
        return filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx'))

    def _detect_policy_category(self, filepath: Path) -> str:
        """
        Detect policy category based on file content or metadata.
        
        Args:
            filepath: Path to policy file
            
        Returns:
            Category name as string
        """
        # First check the filename for obvious category markers
        filename = filepath.name.lower()
        for category, pattern in self._category_patterns.items():
            if re.search(pattern, filename, re.IGNORECASE):
                return category
        
        # If file is small enough, check content (avoid large files)
        if filepath.suffix.lower() == '.txt' and filepath.stat().st_size < 500000:
            try:
                # Check first 1000 chars for category indicators
                with open(filepath, 'r', errors='ignore') as f:
                    content = f.read(1000).lower()
                    
                for category, pattern in self._category_patterns.items():
                    if re.search(pattern, content, re.IGNORECASE):
                        return category
            except Exception:
                # If we can't read the file, default to general category
                pass
                
        return "general"  # Default when no category is detected
    
    def get_policy_metadata(self, policy_path: str) -> Dict:
        """
        Extract metadata from policy file if available.
        
        Args:
            policy_path: Path to policy file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "title": Path(policy_path).stem,
            "category": "general",
            "last_updated": None
        }
        
        # Extract category based on content
        metadata["category"] = self._detect_policy_category(Path(policy_path))
        
        # Get file modification time as last updated
        try:
            mtime = Path(policy_path).stat().st_mtime
            metadata["last_updated"] = mtime
        except Exception:
            pass
            
        return metadata
