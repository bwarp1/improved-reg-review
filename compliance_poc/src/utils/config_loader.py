import os
import yaml
from pathlib import Path
from typing import Dict, Any

def get_env_or_config(key: str, config_value: Any, default: Any = None) -> Any:
    """Get value from environment or config with fallback to default
    
    Args:
        key: The configuration key (will be prefixed with COMPLIANCE_ for env vars)
        config_value: The value from the config file
        default: Default value if neither environment nor config has the value
        
    Returns:
        The value from environment variable, config, or default (in that order of precedence)
    """
    env_key = f"COMPLIANCE_{key.upper()}"
    return os.environ.get(env_key) or config_value or default

def load_config(config_path=None) -> Dict[str, Any]:
    """Load configuration from YAML file and environment variables"""
    if config_path is None:
        config_path = Path(__file__).parents[2] / "config" / "config.yaml"
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Ensure notifications config exists
    if 'notifications' not in config:
        config['notifications'] = {}
        
    # Override with environment variables if present
    notifications = config['notifications']
    notifications['smtp_server'] = get_env_or_config('SMTP_SERVER', notifications.get('smtp_server'))
    notifications['smtp_port'] = int(get_env_or_config('SMTP_PORT', notifications.get('smtp_port'), 587))
    notifications['smtp_username'] = get_env_or_config('SMTP_USERNAME', notifications.get('smtp_username'))
    notifications['smtp_password'] = get_env_or_config('SMTP_PASSWORD', notifications.get('smtp_password'))
    notifications['sender_email'] = get_env_or_config('SMTP_SENDER', notifications.get('sender_email'), 'compliance@example.com')
    notifications['enabled'] = get_env_or_config('NOTIFICATIONS_ENABLED', notifications.get('enabled'), 'false').lower() == 'true'
    
    return config