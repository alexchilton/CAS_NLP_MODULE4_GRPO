"""
Configuration loading and validation utilities.

This module provides utilities to load YAML configuration files
and access them using dot notation for convenience.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigNamespace:
    """
    A class that allows dot notation access to dictionary values.

    Example:
        >>> config = ConfigNamespace({"training": {"batch_size": 32}})
        >>> config.training.batch_size
        32
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize ConfigNamespace from a dictionary.

        Args:
            config_dict: Dictionary to convert to namespace.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __repr__(self) -> str:
        """Return string representation of the namespace."""
        return f"ConfigNamespace({self.to_dict()})"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert namespace back to a dictionary.

        Returns:
            Dictionary representation of the namespace.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value with a default fallback.

        Args:
            key: The key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            The value or default.
        """
        return getattr(self, key, default)


def load_config(config_path: Union[str, Path]) -> ConfigNamespace:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        ConfigNamespace object with the loaded configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    # Convert to Path object for cross-platform compatibility
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate the config
    validate_config(config_dict)

    config = ConfigNamespace(config_dict)
    logger.info("Configuration loaded successfully")

    return config


def validate_config(config_dict: Dict[str, Any]) -> None:
    """
    Validate that required configuration fields exist.

    Args:
        config_dict: The configuration dictionary to validate.

    Raises:
        ValueError: If required fields are missing.
    """
    required_fields = {
        "training": ["batch_size", "learning_rate", "epochs"],
        "model": ["name"],
        "data": ["dataset_name"],
        "output": ["checkpoint_dir", "log_dir"],
    }

    missing_sections = []
    missing_fields = []

    for section, fields in required_fields.items():
        if section not in config_dict:
            missing_sections.append(section)
        else:
            for field in fields:
                if field not in config_dict[section]:
                    missing_fields.append(f"{section}.{field}")

    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    logger.info("Configuration validation passed")


def pretty_print_config(config: Union[ConfigNamespace, Dict[str, Any]], indent: int = 0) -> None:
    """
    Pretty print configuration for debugging.

    Args:
        config: Configuration to print (ConfigNamespace or dict).
        indent: Current indentation level (used for recursion).
    """
    if isinstance(config, ConfigNamespace):
        config = config.to_dict()

    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            pretty_print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def print_config_summary(config: ConfigNamespace) -> None:
    """
    Print a formatted summary of the configuration.

    Args:
        config: The configuration to summarize.
    """
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    pretty_print_config(config)
    print("=" * 60 + "\n")


def save_config(config: Union[ConfigNamespace, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration to save.
        output_path: Path where to save the configuration.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, ConfigNamespace):
        config = config.to_dict()

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Configuration saved to: {output_path}")
