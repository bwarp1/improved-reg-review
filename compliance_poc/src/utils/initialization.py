"""Initialization utilities for the compliance system."""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

from compliance_poc.src.utils.config_manager import ConfigManager, EnvironmentConfig
from compliance_poc.src.utils.database import DatabaseManager
from compliance_poc.src.optimization.threshold_optimizer import ThresholdOptimizer

def initialize_system(environment: str = "development") -> Tuple[EnvironmentConfig, DatabaseManager, ThresholdOptimizer]:
    """
    Initialize the compliance system with appropriate configuration and components.
    
    Args:
        environment: Name of the environment to load (e.g., "development", "production")
        
    Returns:
        Tuple of (config, database_manager, threshold_optimizer)
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger("compliance-system")
        
        # Load configuration
        logger.info(f"Initializing compliance system for environment: {environment}")
        config_manager = ConfigManager()
        config = config_manager.load_environment(environment)
        
        # Set log level from config
        logging.getLogger().setLevel(config.log_level)
        
        # Initialize database
        logger.info("Initializing database connection")
        database_url = os.environ.get("DATABASE_URL", config.database_url)
        db_manager = DatabaseManager(database_url)
        
        # Initialize optimizer with config and database
        logger.info("Initializing threshold optimizer")
        optimizer = ThresholdOptimizer(
            config=config.__dict__,  # Convert dataclass to dict
            db_manager=db_manager
        )
        
        # Create required directories
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(config.reporting, "output_dir"):
            Path(config.reporting["output_dir"]).mkdir(parents=True, exist_ok=True)
        
        logger.info("System initialization completed successfully")
        return config, db_manager, optimizer
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise RuntimeError(f"System initialization failed: {str(e)}") from e

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

def get_system_status(config: EnvironmentConfig, 
                     db_manager: DatabaseManager,
                     optimizer: ThresholdOptimizer) -> dict:
    """
    Get current status of the compliance system.
    
    Args:
        config: Current environment configuration
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
            db_manager.Session()
        except Exception as e:
            db_status = f"error: {str(e)}"
            
        # Get optimization metrics
        optimization_metrics = optimizer.get_performance_metrics()
        
        # Check cache directory
        cache_status = "ok"
        cache_dir = Path(config.cache_dir)
        if not cache_dir.exists():
            cache_status = "directory missing"
        elif not os.access(cache_dir, os.W_OK):
            cache_status = "not writable"
            
        return {
            "environment": config.name,
            "log_level": config.log_level,
            "demo_mode": config.use_demo_data,
            "database": {
                "status": db_status,
                "url": config.database_url.split("@")[-1]  # Hide credentials
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
