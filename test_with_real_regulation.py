#!/usr/bin/env python3
"""Test the compliance tool with a real regulation."""
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from compliance_poc.src.main import process_regulation

def setup_logging():
    """Configure logging for the test script."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    return logging.getLogger("test-script")

def main():
    """Run the test script."""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Test with a real regulation")
    parser.add_argument("--docket", default="EPA-HQ-OAR-2021-0257", 
                        help="Regulation docket ID")
    parser.add_argument("--doc-id", help="Specific document ID")
    parser.add_argument("--policy-dir", default="sample_data/policies",
                        help="Directory with company policies")
    parser.add_argument("--output", choices=["table", "json", "csv"], default="table",
                        help="Output format")
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get("REGULATIONS_API_KEY")
    if not api_key:
        logger.warning("No API key found in environment. Running in demo mode.")
        logger.info("To use real regulations, set the REGULATIONS_API_KEY environment variable.")
    else:
        logger.info("API key found. Using real regulations API.")
    
    # Validate policy directory
    policy_dir = Path(args.policy_dir)
    if not policy_dir.exists():
        logger.error(f"Policy directory not found: {policy_dir}")
        logger.info("Using sample policies instead.")
        args.policy_dir = "sample_data/policies"
        
        # Create the directory if it doesn't exist
        Path("sample_data/policies").mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Processing regulation: Docket={args.docket}, Doc ID={args.doc_id}")
    
    # Process regulation
    results = process_regulation(args.docket, args.doc_id, args.policy_dir)
    
    # Check for errors
    if "error" in results:
        logger.error(f"Error processing regulation: {results['error']}")
        sys.exit(1)
    
    # Summarize results
    obligations = results.get("obligations", [])
    matches = results.get("matches", [])
    
    logger.info(f"Analysis complete. Found {len(obligations)} obligations.")
    
    # Calculate compliance metrics
    if matches:
        compliant = sum(1 for m in matches if m.get("status") == "Compliant")
        partial = sum(1 for m in matches if m.get("status") == "Partial")
        non_compliant = sum(1 for m in matches if m.get("status") == "Non-Compliant")
        
        print("\n=== COMPLIANCE SUMMARY ===")
        print(f"Compliant requirements: {compliant}")
        print(f"Partially compliant: {partial}")
        print(f"Non-compliant: {non_compliant}")
        print("========================\n")
        
        # Display detailed results based on output format
        if args.output == "table":
            from tabulate import tabulate
            table_data = []
            for match in matches:
                table_data.append([
                    match.get("obligation_id", ""),
                    match.get("status", "Unknown"),
                    f"{match.get('score', 0):.2f}",
                    match.get("requirement", "")[:70] + "...",
                    match.get("policy", "None")
                ])
            
            headers = ["ID", "Status", "Score", "Requirement", "Policy"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
        elif args.output == "json":
            import json
            output_file = f"results_{args.docket}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
            
        elif args.output == "csv":
            import csv
            output_file = f"results_{args.docket}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Obligation ID", "Requirement", "Status", "Score", "Policy"])
                for match in matches:
                    writer.writerow([
                        match.get("obligation_id", ""),
                        match.get("requirement", ""),
                        match.get("status", "Unknown"),
                        match.get("score", 0),
                        match.get("policy", "None")
                    ])
            print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
