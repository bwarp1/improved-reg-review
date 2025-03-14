#!/usr/bin/env python3
"""
Configuration utility for the Regulatory Compliance Analysis tool.
This script helps set up the tool for production use with real APIs.
"""

import os
import sys
import yaml
from pathlib import Path
import argparse

def setup_config():
    """Set up the configuration for production use."""
    parser = argparse.ArgumentParser(description="Configure Regulatory Compliance Tool")
    parser.add_argument("--api-key", help="Regulations.gov API key")
    parser.add_argument("--policy-dir", help="Path to company policies directory")
    parser.add_argument("--output-dir", help="Path to output reports directory")
    args = parser.parse_args()
    
    # Find the config file
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    # Load existing config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error loading config file: {e}")
        return 1
    
    # Initialize config sections if needed
    if "api" not in config:
        config["api"] = {}
    if "paths" not in config:
        config["paths"] = {}
    
    # Update API key
    if args.api_key:
        config["api"]["key"] = args.api_key
        print(f"✓ API key updated")
    else:
        # Check if an API key is already in environment or config
        env_key = os.environ.get("REGULATIONS_API_KEY")
        config_key = config.get("api", {}).get("key")
        
        if env_key:
            print(f"ℹ Using API key from environment variable")
        elif config_key and config_key != "DEMO_KEY":
            print(f"ℹ Using existing API key from config")
        else:
            print("⚠ No API key set. The tool will run in demo mode.")
            print("  To use real regulations, either:")
            print("  - Set the REGULATIONS_API_KEY environment variable")
            print("  - Run this script with --api-key YOUR_API_KEY")
    
    # Update policy directory
    if args.policy_dir:
        policy_path = Path(args.policy_dir)
        if not policy_path.exists():
            print(f"⚠ Policy directory {args.policy_dir} does not exist. Creating it...")
            try:
                policy_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory: {e}")
                return 1
        
        config["paths"]["policy_dir"] = str(policy_path)
        print(f"✓ Policy directory set to: {args.policy_dir}")
    
    # Update output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
        if not output_path.exists():
            print(f"⚠ Output directory {args.output_dir} does not exist. Creating it...")
            try:
                output_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory: {e}")
                return 1
        
        config["paths"]["output_dir"] = str(output_path)
        print(f"✓ Output directory set to: {args.output_dir}")
    
    # Save the updated config
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config: {e}")
        return 1
    
    print("\n✓ Configuration complete!")
    print("\nTo run the tool:")
    print("  - Web interface: streamlit run app.py")
    print("  - Command line: python run.py --docket EPA-HQ-OAR-2021-0257")
    print("  - Test script: python test_with_real_regulation.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(setup_config())
