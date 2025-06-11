"""Initialization utilities for the compliance system."""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

from compliance_poc.src.utils.config_manager import ConfigManager # EnvConfig not directly used here
from compliance_poc.src.utils.database import DatabaseManager
from compliance_poc.src.optimization.threshold_optimizer import ThresholdOptimizer

# Note: EnvConfig was removed from the import as it's not directly used as a type hint
# in this file's functions anymore. It's defined in config_manager.py.

def initialize_system(cm: ConfigManager) -> Tuple[DatabaseManager, ThresholdOptimizer]:
    """
    Initialize the compliance system using the provided ConfigManager.
    
    Args:
        cm: An instance of ConfigManager that has already loaded configuration.
        
    Returns:
        Tuple of (DatabaseManager, ThresholdOptimizer)
        
    Raises:
        RuntimeError: If initialization fails
    """
    logger = logging.getLogger("compliance-system")
    try:
        # Assuming main.py's setup_logging has already run logging.basicConfig().
        # Set the level for the root logger based on config.
        log_level_str = cm.get("logging.level", "INFO").upper()
        root_logger = logging.getLogger() # Get the root logger
        if hasattr(logging, log_level_str):
            root_logger.setLevel(getattr(logging, log_level_str))
            # This log might be suppressed if root_logger's level is higher than logger's level
            logger.info(f"Root logger level set to: {log_level_str} by initialize_system.")
        else:
            logger.warning(f"Invalid log level '{log_level_str}' in config. Root logger level unchanged by initialize_system.")
        
        logger.info(f"Initializing system components using ConfigManager for env: {cm.get('app.env', 'unknown')}")
        
        # Initialize database
        logger.info("Initializing database connection...")
        database_url = os.environ.get("DATABASE_URL") or cm.get("database.url")
        # DatabaseManager has a default connection string if database_url is None
        db_manager = DatabaseManager(database_url) if database_url else DatabaseManager()
        logger.info("DatabaseManager initialized.")
        
        # Initialize threshold optimizer with configuration and database
        logger.info("Initializing threshold optimizer...")
        # ThresholdOptimizer expects a config dictionary. cm.config is this dictionary.
        optimizer = ThresholdOptimizer(
            config=cm.config, 
            db_manager=db_manager
        )
        logger.info("ThresholdOptimizer initialized.")
        
        # Directory creation for cache and output are handled by their respective components or main setup.
        
        logger.info("System components initialization completed successfully.")
        return db_manager, optimizer
    except Exception as e:
        logger.exception("Failed to initialize system components in initialize_system")
        raise RuntimeError("System components initialization failed in initialize_system") from e

def cleanup_system(db_manager: Optional[DatabaseManager] = None) -> None:
    """
    Perform cleanup operations when shutting down the system.
    
    Args:
        db_manager: Database manager instance to cleanup
    """
    logger = logging.getLogger("compliance-system")
    
    try:
        if db_manager:
            # Close database connections
            logger.info("Closing database connections")
            if hasattr(db_manager, "engine"):
                db_manager.engine.dispose()
        
        logger.info("System cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during system cleanup: {e}")
        raise RuntimeError(f"System cleanup failed: {str(e)}") from e

def get_system_status(cm: ConfigManager, 
                     db_manager: DatabaseManager,
                     optimizer: ThresholdOptimizer) -> dict:
    """
    Get current status of the compliance system.
    
    Args:
        cm: ConfigManager instance with loaded configuration.
        db_manager: Database manager instance
        optimizer: Threshold optimizer instance
        
    Returns:
        Dictionary containing system status information
    """
    logger = logging.getLogger("compliance-system")
    
    try:
        # Check database connection
        db_status = "ok"
        try:
            # Attempt to create a session to check connectivity
            session = db_manager.Session()
            session.close()
        except Exception as e:
            db_status = f"error: {str(e)}"
            logger.warning(f"Database connection check failed: {e}")
            
        # Get optimization metrics
        optimization_metrics = optimizer.get_performance_metrics()
        
        # Check cache directory status
        # The CACHE_DIR is defined in main.py as Path("cache").
        # We'll use that fixed path for status checking here.
        # Alternatively, a dedicated config key for cache_dir could be used if it exists.
        cache_dir_path_str = cm.get("paths.cache_dir", "cache") # Default to "cache"
        cache_dir = Path(cache_dir_path_str)
        cache_status = "ok"
        if not cache_dir.exists():
            cache_status = f"directory missing ({cache_dir})"
        elif not os.access(cache_dir, os.W_OK):
            cache_status = f"not writable ({cache_dir})"
            
        db_url_display = (cm.get("database.url") or "default_sqlite").split("@")[-1]

        return {
            "environment": cm.get("app.env", "unknown"),
            "log_level": cm.get("logging.level", "UNKNOWN"),
            "demo_mode": cm.get("api.use_demo_data", False),
            "database": {
                "status": db_status,
                "url_suffix": db_url_display
            },
            "cache": {
                "status": cache_status,
                "directory": str(cache_dir)
            },
            "optimization": {
                "metrics": optimization_metrics,
                "adaptation_rate": optimizer.adaptation_rate
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
