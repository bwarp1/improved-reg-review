#!/usr/bin/env python3
"""
Demo script for Regulatory Compliance Analysis PoC.
This uses sample data rather than calling the actual Regulations.gov API.
"""
import os
import logging
import yaml
from pathlib import Path

from compliance_poc.src.api.regulations_api import RegulationsAPI
from compliance_poc.src.nlp.extractor import ObligationExtractor
from compliance_poc.src.policy.loader import PolicyLoader
from compliance_poc.src.matching.comparer import ComplianceComparer
from compliance_poc.src.reporting.report_generator import ReportGenerator

def load_config(config_path="compliance_poc/config/config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        # Provide minimal default config
        return {}

def main():
    """Run the demo pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Regulatory Compliance Analysis Demo")
    
    # Load demo configuration or use defaults if loading fails
    try:
        config = load_config()
    except Exception as e:
        logger.warning(f"Error loading config: {e}. Using defaults.")
        config = {}
    
    # Set default configurations for each component if not present
    if "nlp" not in config:
        config["nlp"] = {}
    if "matching" not in config:
        config["matching"] = {}
    if "reporting" not in config:
        config["reporting"] = {
            "format": "console",
            "directory": "reports"
        }
    
    # Load sample regulatory text
    logger.info("Loading sample regulation document")
    with open("sample_data/regulations/sample_regulation.txt", "r") as f:
        regulation_text = f.read()
    
    # Extract regulatory obligations
    logger.info("Extracting regulatory obligations")
    extractor = ObligationExtractor()
    obligations = extractor.extract_obligations(regulation_text)
    logger.info(f"Found {len(obligations)} regulatory obligations")
    
    # Load internal policies
    logger.info("Loading internal policies")
    policy_loader = PolicyLoader()
    policy_dir = config.get("policy_dir", "sample_data/policies")
    policies = policy_loader.load_policies(policy_dir)
    logger.info(f"Loaded {len(policies)} policy documents")
    
    # Compare obligations against policies
    logger.info("Comparing regulatory obligations against internal policies")
    comparer = ComplianceComparer(config["matching"])
    compliance_matrix = comparer.compare(obligations, policies)
    
    # Generate compliance report
    logger.info("Generating compliance report")
    report_generator = ReportGenerator(config["reporting"])
    report_generator.generate_report({
        "obligations": obligations,
        "policies": policies,
        "matches": compliance_matrix
    })
    
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    main()
