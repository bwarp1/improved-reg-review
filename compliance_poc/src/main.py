"""Main entry point for the Regulatory Compliance Analysis PoC."""
from __future__ import annotations
import os
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, TypedDict # Added TypedDict
from dataclasses import dataclass

from compliance_poc.src.utils.config_manager import ConfigManager
from compliance_poc.src.utils.initialization import (
    initialize_system,
    cleanup_system,
    get_system_status
)
from compliance_poc.src.api.regulations_api import RegulationsAPI
from compliance_poc.src.nlp.extractor import ObligationExtractor
from compliance_poc.src.policy.loader import PolicyLoader
from compliance_poc.src.matching.comparer import ComplianceComparer
from compliance_poc.src.reporting.report_generator import ReportGenerator
from compliance_poc.src.utils.database import DatabaseManager

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass

@dataclass
class ProcessingContext:
    """Context object for regulation processing."""
    config: Dict[str, Any]
    logger: logging.Logger
    api_client: RegulationsAPI
    extractor: ObligationExtractor
    policy_loader: PolicyLoader
    comparer: ComplianceComparer
    report_generator: Optional[ReportGenerator] = None

# Define the path for the processed dockets tracking file within the cache directory
CACHE_DIR = Path("cache")
PROCESSED_DOCKETS_FILE = CACHE_DIR / "processed_dockets.txt"

def get_docket_ids_from_policy_dir(policy_dir_path: Path) -> List[str]:
    """Scan the policy directory for subdirectories, treating each as a docket ID."""
    if not policy_dir_path.is_dir():
        return []
    return [d.name for d in policy_dir_path.iterdir() if d.is_dir()]

def load_processed_dockets() -> set[str]:
    """Load processed docket IDs from the tracking file."""
    if not PROCESSED_DOCKETS_FILE.exists():
        return set()
    with open(PROCESSED_DOCKETS_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}

def mark_docket_as_processed(docket_id: str, logger: logging.Logger) -> None:
    """Mark_docket_as_processed a docket ID as processed by appending it to the tracking file."""
    try:
        # Ensure the cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_DOCKETS_FILE, "a") as f:
            f.write(f"{docket_id}\n")
        logger.info(f"Marked docket {docket_id} as processed. Tracking file: {PROCESSED_DOCKETS_FILE}")
    except IOError as e:
        logger.error(f"Error writing to processed dockets file {PROCESSED_DOCKETS_FILE}: {e}")


class ComplianceResult(TypedDict):
    """Type definition for compliance analysis results."""
    docket_id: str
    document_id: Optional[str]
    obligations: List[Dict]
    policies: Dict[str, str]
    matches: List[Dict]

def setup_logging(config_manager: ConfigManager) -> logging.Logger:
    """Configure logging based on configuration."""
    log_dir = Path(config_manager.get("logging.file")).parent
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_level = config_manager.get("logging.level", "INFO")
    log_format = config_manager.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "app.log")
    ]
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # Create and return application logger
    logger = logging.getLogger("compliance-poc")
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Regulatory Compliance Analysis Tool")
    parser.add_argument("--env", default="development",
                        help="Environment to use (development/production)")
    parser.add_argument("--docket", help="Regulation docket ID")
    parser.add_argument("--doc-id", help="Specific document ID")
    parser.add_argument("--policy-dir", default=None, # Changed default to None
                        help="Directory with company policies. Defaults to value from config file.")
    parser.add_argument("--force-reprocess", action="store_true", default=False,
                        help="Force reprocessing of all dockets, ignoring the processed dockets log.")
    return parser.parse_args()

def setup_processing_context(config_manager: ConfigManager, logger: logging.Logger) -> ProcessingContext:
    """Initialize and configure all processing components."""
    config = config_manager.config
    
    # Check for API key and demo mode
    api_key = config_manager.get("api.key")
    use_demo_data = config_manager.get("api.use_demo_data", False) or not api_key
    
    if use_demo_data:
        logger.info("No API key found or demo mode configured. Using sample data.")
    
    # Initialize all components
    return ProcessingContext(
        config=config,
        logger=logger,
        api_client=RegulationsAPI(api_key=api_key, use_demo_data=use_demo_data),
        extractor=ObligationExtractor(config=config["nlp"]),
        policy_loader=PolicyLoader(),
        comparer=ComplianceComparer(config["matching"]),
        # Use "output" key for reporting configuration as per base.yaml and development.yaml structure
        report_generator=(ReportGenerator(config["output"]) 
                        if config.get("output", {}).get("format") != "none" else None)
    )

def fetch_regulation_text(context: ProcessingContext, docket_id: str, 
                        document_id: Optional[str] = None) -> str:
    """Fetch regulation text from API or sample data."""
    if document_id:
        context.logger.info(f"Fetching specific document: {document_id}")
        return context.api_client.get_document_content(document_id)
    
    context.logger.info(f"Searching for documents in docket: {docket_id}")
    documents = context.api_client.search_by_docket(docket_id)
    if not documents:
        context.logger.warning(f"No documents found for docket: {docket_id}")
        return ""
    
    document_id = documents[0].get("id")
    context.logger.info(f"Using document: {document_id}")
    return context.api_client.get_document_content(document_id)

def get_sample_regulation_text(context: ProcessingContext) -> str:
    """Get sample regulation text for demo mode."""
    context.logger.info("Using sample regulation text in demo mode")
    sample_path = Path("sample_data/regulations/sample_regulation.txt")
    
    if sample_path.exists():
        with open(sample_path, "r") as f:
            return f.read()
            
    return ("Section 1.1: Organizations must maintain records for at least 5 years.\n"
            "Section 1.2: All employees shall receive annual security training.")

def process_regulation(docket_id: str, 
                      document_id: Optional[str] = None,
                      policy_dir: Optional[str] = None,
                      config_manager: Optional[ConfigManager] = None,
                      db_manager: Optional[DatabaseManager] = None,
                      optimizer: Optional[ThresholdOptimizer] = None) -> ComplianceResult:
    """
    Process a regulation document and compare against policies.
    
    Args:
        docket_id: The docket ID for the regulation
        document_id: Specific document ID (optional)
        policy_dir: Directory containing policy documents (optional)
        config_manager: Configuration manager instance (optional)
        db_manager: Database manager instance (optional)
        optimizer: Threshold optimizer instance (optional)
        
    Returns:
        ComplianceResult containing analysis results
    """
    # Setup configuration if not provided
    if config_manager is None:
        config_manager = ConfigManager()
        config_manager.load_config()
    
    # Setup logging and processing context
    logger = setup_logging(config_manager)
    context = setup_processing_context(config_manager, logger)
    
    try:
        logger.info(f"Processing regulation: Docket ID={docket_id}, Document ID={document_id}")
        
        # Fetch regulation text
        regulation_text = fetch_regulation_text(context, docket_id, document_id)
        if not regulation_text and context.api_client.use_demo_data:
            regulation_text = get_sample_regulation_text(context)
        
        # Process regulation text with performance tracking
        if not regulation_text:
            raise ValueError("Failed to retrieve regulation text")
            
        # Extract obligations with enhanced NLP
        context.logger.info("Extracting obligations from regulation text")
        obligations = context.extractor.extract_obligations(regulation_text)
        context.logger.info(f"Found {len(obligations)} obligations")
        
        # Load and analyze policies from configured or specified directory
        policy_directory = policy_dir or config_manager.get("paths.policy_dir")
        context.logger.info(f"Loading policies from directory: {policy_directory}")
        policies = context.policy_loader.load_policies(policy_directory)
        context.logger.info(f"Loaded {len(policies)} policy documents")
        
        # Compare obligations against policies using optimizer if available
        context.logger.info("Comparing obligations against policies")
        if optimizer:
            # Ensure the optimizer's thresholds are up to date.
            # optimize_thresholds might take arguments like min_samples, domains.
            # Calling it without args will use its defaults.
            context.logger.info("Optimizing thresholds using the main optimizer instance.")
            optimizer.optimize_thresholds() 
            
            # Propagate the optimized thresholds to the comparer.
            # ThresholdOptimizer.thresholds is like {"base": {"value": 0.60, "confidence": ...}}
            # ComplianceComparer.thresholds expects a flat dict like {"base": 0.60}
            new_thresholds = {domain: data["value"] for domain, data in optimizer.thresholds.items()}
            context.logger.info(f"Updating comparer thresholds with: {new_thresholds}")
            context.comparer.thresholds = new_thresholds
        
        # Always use the comparer to perform the actual matching.
        # If optimizer was present, comparer now uses its updated thresholds.
        # If optimizer was not present, comparer uses its internally initialized thresholds.
        matches = context.comparer.compare(obligations, policies)
            
        # Store results in database if available
        if db_manager:
            try:
                for match in matches:
                    db_manager.add_performance_metric(
                        domain=match.get("domain", "base"),
                        threshold=match.get("threshold_used", 0.0),
                        score=match.get("adjusted_score", 0.0),
                        is_correct=match.get("status") == "Compliant",
                        details=match.get("match_details")
                    )
            except Exception as e:
                context.logger.warning(f"Failed to store metrics in database: {e}")
        
        # Prepare results
        result: ComplianceResult = {
            "docket_id": docket_id,
            "document_id": document_id,
            "obligations": obligations,
            "policies": policies,
            "matches": matches
        }
        
        # Generate report if configured
        if context.report_generator:
            context.logger.info("Generating compliance report")
            context.report_generator.generate_report({
                "obligations": obligations,
                "policies": policies,
                "matches": matches
            })
        
        return result
    
    except Exception as e:
        context.logger.error(f"Error processing regulation: {e}", exc_info=True)
        raise RuntimeError(f"Failed to process regulation: {str(e)}")

def print_compliance_summary(results: ComplianceResult) -> None:
    """Print a summary of compliance results to the console."""
    print("\n=== COMPLIANCE SUMMARY ===")
    
    # Count different compliance statuses
    compliance_status = {
        "compliant": 0,
        "partial": 0,
        "non_compliant": 0
    }
    
    for match in results["matches"]:
        status = match.get("status", "unknown").lower()
        if status in compliance_status:
            compliance_status[status] += 1
    
    # Print results
    print(f"Total Obligations: {len(results['obligations'])}")
    print(f"Compliant requirements: {compliance_status['compliant']}")
    print(f"Partially compliant: {compliance_status['partial']}")
    print(f"Non-compliant: {compliance_status['non_compliant']}")
    print("========================\n")

def main() -> None:
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.env)
        
        # Setup logging
        logger = setup_logging(config_manager)
        logger.info(f"Starting regulatory compliance analysis in {args.env} environment")
        
        # Initialize system components
        # The setup_processing_context is called within process_regulation if needed.
        # ConfigManager is passed to process_regulation, which then handles context setup.
        # Pass the ConfigManager instance to initialize_system
        db_manager, optimizer = initialize_system(config_manager)

        # Determine the base policy directory from arguments or configuration.
        # This base directory is expected to contain subdirectories, each representing a docket.
        base_policy_dir_path = Path(args.policy_dir if args.policy_dir else config_manager.get("paths.policy_dir", "company_policies"))
        logger.info(f"Using base policy directory: {base_policy_dir_path}")

        if not base_policy_dir_path.exists() or not base_policy_dir_path.is_dir():
            logger.error(f"Base policy directory '{base_policy_dir_path}' does not exist or is not a directory. Exiting.")
            sys.exit(1)
        
        if args.docket:
            # Manual mode: Process a single, specified docket.
            # The policy_dir for this docket will be base_policy_dir_path / args.docket.
            logger.info(f"Manual mode: Processing specified docket '{args.docket}'")
            docket_specific_policy_path = base_policy_dir_path / args.docket

            if not docket_specific_policy_path.is_dir():
                logger.error(f"Specified docket directory '{docket_specific_policy_path}' not found under base policy directory. Exiting.")
                sys.exit(1)
            
            try:
                # Note: process_regulation expects policy_dir to be the path to the actual policy files for that docket.
                results = process_regulation(
                    docket_id=args.docket,
                    document_id=args.doc_id,
                    policy_dir=str(docket_specific_policy_path), 
                    config_manager=config_manager, # Pass manager to avoid re-init of logger/context
                    db_manager=db_manager,
                    optimizer=optimizer
                )
                print_compliance_summary(results)
                mark_docket_as_processed(args.docket, logger) # Mark after successful processing
                logger.info(f"Successfully processed docket: {args.docket}")
            except Exception as e:
                logger.error(f"Error processing docket {args.docket}: {e}", exc_info=True)
                # In manual mode, an error for the single docket should likely halt.
                # Depending on requirements, could allow continuing if this was part of a batch.
        
        else:
            # Automatic mode: Scan base_policy_dir_path for dockets (subdirectories).
            logger.info(f"Automatic mode: Scanning '{base_policy_dir_path}' for dockets.")
            detected_docket_ids = get_docket_ids_from_policy_dir(base_policy_dir_path)
            
            if not detected_docket_ids:
                logger.info(f"No dockets (subdirectories) found in '{base_policy_dir_path}'.")
                # Optionally, print system status here as per original behavior if nothing to do.
                # Pass the ConfigManager instance to get_system_status
                status = get_system_status(config_manager, db_manager, optimizer)
                print("\n=== SYSTEM STATUS (No Dockets Found) ===")
                print(f"Environment: {config_manager.get('app.env')}")
                print(f"Database: {status['database']['status']}")
                print(f"Cache: {status['cache']['status']}")
                print(f"Demo Mode: {'Enabled' if config_manager.get('api.use_demo_data') else 'Disabled'}")
                print(f"Policy Directory Searched: {base_policy_dir_path}")
                print("=======================================\n")
                sys.exit(0)

            logger.info(f"Found dockets: {', '.join(detected_docket_ids)}")
            processed_dockets_set = load_processed_dockets()
            
            if args.force_reprocess:
                logger.info("Force reprocessing enabled. All detected dockets will be processed regardless of prior status.")

            processed_count = 0
            skipped_count = 0
            error_count = 0

            for docket_id_to_process in detected_docket_ids:
                if docket_id_to_process in processed_dockets_set and not args.force_reprocess:
                    logger.info(f"Skipping already processed docket: {docket_id_to_process}")
                    skipped_count += 1
                    continue
                
                logger.info(f"Processing docket: {docket_id_to_process}")
                docket_specific_policy_path = base_policy_dir_path / docket_id_to_process
                try:
                    results = process_regulation(
                        docket_id=docket_id_to_process,
                        document_id=None, # In auto mode, doc_id is not specified per docket.
                        policy_dir=str(docket_specific_policy_path),
                        config_manager=config_manager,
                        db_manager=db_manager,
                        optimizer=optimizer
                    )
                    print_compliance_summary(results)
                    mark_docket_as_processed(docket_id_to_process, logger)
                    logger.info(f"Successfully processed docket: {docket_id_to_process}")
                    processed_count +=1
                except Exception as e:
                    logger.error(f"Error processing docket {docket_id_to_process}: {e}", exc_info=True)
                    error_count +=1
                    # Continue to the next docket in automatic mode.
            
            logger.info(f"Automatic processing summary: "
                        f"Processed {processed_count} dockets, "
                        f"Skipped {skipped_count} dockets, "
                        f"Errors on {error_count} dockets.")

        logger.info("Compliance analysis run completed.")
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Ensure proper cleanup
        try:
            cleanup_system(db_manager)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
