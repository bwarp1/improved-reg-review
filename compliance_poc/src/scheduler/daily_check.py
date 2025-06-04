import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from apscheduler.schedulers.blocking import BlockingScheduler
from compliance_poc.src.api.regulations_api import RegulationsAPI
from compliance_poc.src.policy.categorizer import PolicyLoader
from compliance_poc.src.main import process_regulation_by_category, generate_summary_report, notify_team
from compliance_poc.config.loader import load_config

# Load configuration once at module level (DRY principle)
config = load_config()

class RegulationChecker:
    """
    Centralized class for checking new regulations and running compliance analysis.
    Following SOLID principles by organizing related functionality in one class.
    """
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.api_client = None
        self.history_file = Path("compliance_poc/data/regulation_history.json")
        self.history_file.parent.mkdir(exist_ok=True, parents=True)
    
    def initialize_api(self):
        """Initialize API client (lazy loading - YAGNI principle)."""
        if not self.api_client:
            self.api_client = RegulationsAPI(api_key=self.config["api"].get("key"))
        return self.api_client
    
    def check_new_regulations(self, domains=None, days_back=None):
        """
        Check for new regulations in specified domains.
        
        Args:
            domains: List of regulatory domains to check
            days_back: How many days back to check (defaults to config setting)
            
        Returns:
            List of new regulations
        """
        today = datetime.now()
        days_lookback = days_back if days_back is not None else self.config["scheduler"].get("days_lookback", 1)
        since_date = (today - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
        
        api_client = self.initialize_api()
        
        # Use the search method with appropriate parameters
        new_regulations = self.poll_for_new_regulations(api_client, since_date, domains)
        
        if not new_regulations:
            self.logger.info(f"No new regulations found since {since_date}")
            return []
            
        self.logger.info(f"Found {len(new_regulations)} new regulations")
        return new_regulations
    
    def poll_for_new_regulations(self, api_client, since_date=None, domains=None):
        """
        Poll Regulations.gov for new regulations since last check.
        
        Args:
            api_client: Initialized RegulationsAPI client
            since_date: ISO format date string of last check
            domains: List of domains to filter by
            
        Returns:
            list: New regulations found
        """
        # Set up search parameters - simplified approach (KISS principle)
        search_params = {
            "filter[documentType]": "Rule",
            "sort": "postedDate,desc"
        }
        
        # Add date filter if we have a last check date
        if since_date:
            search_params["filter[postedDate][ge]"] = since_date
        
        # Add domain/agency filter if specified
        if domains:
            # Convert domain list to relevant agencies
            search_params["filter[agencies]"] = ",".join(domains)
        
        # Execute search
        return api_client.search_documents(**search_params)
    
    def get_historical_data(self):
        """Load historical regulation data."""
        if not self.history_file.exists():
            return {}
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            self.logger.warning("History file corrupted, starting fresh")
            return {}

    def save_historical_data(self, data):
        """Save regulation data to history."""
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_differential_report(self, new_results, historical_data):
        """Generate a report showing changes from previous checks."""
        changes = {
            'new_regulations': [],
            'updated_regulations': [],
            'summary': {}
        }
        
        for result in new_results:
            reg_id = result.get('id')
            if reg_id not in historical_data:
                changes['new_regulations'].append(result)
            elif result != historical_data[reg_id]:
                changes['updated_regulations'].append(result)
        
        changes['summary'] = {
            'new_count': len(changes['new_regulations']),
            'updated_count': len(changes['updated_regulations']),
            'timestamp': datetime.now().isoformat()
        }
        
        return changes

    def run_daily_check(self):
        """Enhanced daily check with historical comparison."""
        self.logger.info(f"Starting daily regulation check: {datetime.now()}")
        
        # Load historical data
        historical_data = self.get_historical_data()
        
        # Get policy categories
        policy_loader = PolicyLoader()
        policy_dir = self.config["paths"].get("policy_dir", "company_policies")
        policy_categories = policy_loader.categorize_policies(policy_dir)
        
        # Check for new regulations
        new_regulations = self.check_new_regulations()
        
        # Exit early if nothing found (YAGNI principle - avoid unnecessary processing)
        if not new_regulations:
            self.logger.info("No new regulations to process.")
            return []
        
        # Process each regulation against relevant policies only
        results = []
        for reg in new_regulations:
            result = process_regulation_by_category(reg, policy_categories)
            results.append(result)
            
        # Generate differential report
        diff_report = self.generate_differential_report(results, historical_data)
        
        # Update historical data
        for result in results:
            if reg_id := result.get('id'):
                historical_data[reg_id] = result
        self.save_historical_data(historical_data)
        
        # Send notifications if there are changes
        if diff_report['summary']['new_count'] > 0 or diff_report['summary']['updated_count'] > 0:
            self._send_report_notification(diff_report)
        
        # Generate and send report if any new regulations found
        if results:
            generate_summary_report(results)
            notify_team(f"Found {len(results)} new regulations with compliance implications")
        
        # Update last check date
        self._update_check_date()
        
        self.logger.info(f"Completed daily regulation check: {datetime.now()}")
        return {
            'status': 'success',
            'differential_report': diff_report,
            'documents_found': len(results),
            'documents_with_matches': sum(1 for r in results if r.get('matches'))
        }

    def _send_report_notification(self, diff_report):
        """Send email notification about changes."""
        from compliance_poc.src.utils.email_notifier import EmailNotifier
        
        subject = f"Regulation Changes Found - {datetime.now().date()}"
        body = (f"New regulations found: {diff_report['summary']['new_count']}\n"
               f"Updated regulations: {diff_report['summary']['updated_count']}\n\n"
               "Please check the dashboard for details.")
        
        EmailNotifier().send_notification(subject, body)

    def _get_last_check_date(self):
        """Get the date of the last regulation check."""
        check_history_path = Path("compliance_poc/data/check_history.json")
        if not check_history_path.exists():
            return None
            
        with open(check_history_path, "r") as f:
            history = json.load(f)
            return history.get("last_check_date")
    
    def _update_check_date(self):
        """Update the last check date to now."""
        check_history_path = Path("compliance_poc/data/check_history.json")
        check_history_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Use proper datetime format (fixed bug)
        current_date = datetime.now().isoformat()
        
        history = {"last_check_date": current_date}
        if check_history_path.exists():
            try:
                with open(check_history_path, "r") as f:
                    history = json.load(f)
                    history["last_check_date"] = current_date
            except json.JSONDecodeError:
                # Handle corrupt file (KISS principle - simple error handling)
                self.logger.warning("Check history file corrupted, creating new one")
        
        with open(check_history_path, "w") as f:
            json.dump(history, f)

# Create a singleton instance for use throughout the application (DRY principle)
regulation_checker = RegulationChecker(config)

# Simple function interfaces for backward compatibility
def check_new_regulations(domains=None, days_back=None):
    """Proxy function to maintain backwards compatibility."""
    return regulation_checker.check_new_regulations(domains, days_back)

def daily_regulation_check():
    """Daily job to check for new regulations and process them."""
    return regulation_checker.run_daily_check()

def run_daily_check():
    """Simple entry point for scripts."""
    return regulation_checker.run_daily_check()

# Set up scheduler only if this module is run directly (YAGNI principle)
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # Use configuration for scheduling (DRY principle)
    check_time = config["scheduler"].get("check_time", "01:00")
    hour, minute = check_time.split(":")
    scheduler.add_job(daily_regulation_check, 'cron', hour=int(hour), minute=int(minute))
    scheduler.start()
