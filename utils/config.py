"""
Configuration utilities for the F1 Conversational AI project.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_env()
    
    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            filename: YAML configuration filename
            
        Returns:
            Configuration dictionary
        """
        if filename in self._configs:
            return self._configs[filename]
        
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self._configs[filename] = config
        return config
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        return os.getenv(key, default)
    
    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        """
        Get path from environment variable.
        
        Args:
            key: Environment variable name
            default: Default path if not found
            
        Returns:
            Path object
        """
        path_str = self.get_env(key, default)
        if path_str is None:
            raise ValueError(f"Path environment variable {key} not found")
        return Path(path_str)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.load_yaml("training_config.yaml")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.load_yaml("data_config.yaml")
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.load_yaml("evaluation_config.yaml")


# Global configuration instance
config = Config()
