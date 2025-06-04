"""Configuration management for the Regulatory Compliance Analysis tool."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

@dataclass
class EnvConfig:
    """Environment variable configuration mapping."""
    REGULATIONS_API_KEY: str = "api.key"
    SMTP_SERVER: str = "email.smtp_server"
    SMTP_USERNAME: str = "email.username"
    SMTP_PASSWORD: str = "email.password"
    LOG_LEVEL: str = "logging.level"
    APP_ENV: str = "app.env"
    DB_URL: str = "database.url"

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass

class ConfigManager:
    """Manages configuration loading and access with environment overrides."""
    
    def __init__(self, config_dir: str = "compliance_poc/config"):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self.env_mapping = EnvConfig()
        
    def load_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Load and merge configuration files based on environment.
        
        Args:
            environment: Environment name (development/production)
            
        Returns:
            Merged configuration dictionary
            
        Raises:
            ConfigurationError: If config files cannot be loaded
        """
        # Determine environment
        environment = environment or os.getenv("APP_ENV", "development")
        
        try:
            # Load base config
            base_config = self._load_yaml_file("base.yaml")
            if not base_config:
                raise ConfigurationError("Base configuration is empty")
            
            # Load environment-specific config
            env_config = self._load_yaml_file(f"{environment}.yaml") or {}
            
            # Merge configurations
            self.config = self._deep_merge(base_config, env_config)
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            # Validate configuration
            self._validate_config()
            
            self.logger.info(f"Loaded configuration for environment: {environment}")
            return self.config
            
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "api.key")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self.config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
            
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Configuration dictionary
        """
        file_path = self.config_dir / filename
        if not file_path.exists():
            self.logger.warning(f"Configuration file not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {}
            
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if (
                key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Get all attributes from EnvConfig
        for env_var in vars(self.env_mapping):
            if env_var.startswith('_'):
                continue
                
            # Get the config path for this environment variable
            config_path = getattr(self.env_mapping, env_var)
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                # Update config value
                self._set_config_value(config_path, env_value)
                self.logger.debug(f"Override {config_path} from environment")
                
    def _set_config_value(self, path: str, value: Any) -> None:
        """Set a configuration value using dot notation path.
        
        Args:
            path: Configuration path in dot notation
            value: Value to set
        """
        current = self.config
        parts = path.split('.')
        
        # Navigate to the parent of the final key
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the final value
        current[parts[-1]] = value
        
    def _validate_config(self) -> None:
        """Validate the configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = [
            "app.name",
            "api.base_url",
            "paths.data_dir",
            "paths.logs_dir",
            "nlp.model",
            "logging.level",
            "logging.file"
        ]
        
        for path in required_sections:
            if not self.get(path):
                raise ConfigurationError(f"Required configuration missing: {path}")
                
        # Validate specific values
        log_level = self.get("logging.level", "").upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid log level: {log_level}")
            
        # Validate paths exist or can be created
        for path_key in ["paths.data_dir", "paths.logs_dir", "paths.output_dir"]:
            dir_path = Path(self.get(path_key, ""))
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ConfigurationError(f"Cannot create directory {dir_path}: {e}")
                    
    def create_env_file(self, template_path: Optional[str] = None) -> None:
        """Create a .env file template.
        
        Args:
            template_path: Path to write the .env template
        """
        template_path = template_path or ".env.template"
        env_vars = []
        
        # Add all environment variables from mapping
        for env_var in vars(self.env_mapping):
            if not env_var.startswith('_'):
                config_path = getattr(self.env_mapping, env_var)
                current_value = self.get(config_path, "")
                if isinstance(current_value, str) and not current_value.startswith("__"):
                    current_value = f"__{current_value}__"
                env_vars.append(f"{env_var}={current_value}")
                
        # Write template file
        try:
            with open(template_path, 'w') as f:
                f.write("# Environment variables for Regulatory Compliance Tool\n\n")
                f.write("\n".join(sorted(env_vars)))
            self.logger.info(f"Created environment template at {template_path}")
        except Exception as e:
            self.logger.error(f"Error creating environment template: {e}")
            
    def export_effective_config(self, output_path: Optional[str] = None) -> None:
        """Export the effective configuration for debugging.
        
        Args:
            output_path: Path to write the effective configuration
        """
        if not output_path:
            output_path = self.config_dir / "effective_config.yaml"
            
        try:
            # Remove sensitive values
            safe_config = self._redact_sensitive_values(self.config.copy())
            
            with open(output_path, 'w') as f:
                yaml.dump(safe_config, f, default_flow_style=False)
            self.logger.info(f"Exported effective configuration to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting effective configuration: {e}")
            
    def _redact_sensitive_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive values from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with sensitive values redacted
        """
        sensitive_keys = ['password', 'key', 'secret', 'token']
        result = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                result[key] = self._redact_sensitive_values(value)
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                result[key] = "**REDACTED**"
            else:
                result[key] = value
                
        return result
