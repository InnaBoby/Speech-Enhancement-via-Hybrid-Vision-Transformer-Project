"""
Configuration Management Utilities

This module provides utilities for loading, saving, and managing
configuration files in YAML format.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Configuration saved to {output_path}")


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries.

    The override_config values take precedence over base_config.

    Args:
        base_config: Base configuration
        override_config: Override configuration

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override value
            merged[key] = value

    return merged


def load_all_configs(config_dir: str = 'config') -> Dict[str, Any]:
    """
    Load all configuration files from directory and merge them.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        Merged configuration dictionary
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    # Define config files to load
    config_files = [
        'data_config.yaml',
        'model_config.yaml',
        'train_config.yaml',
    ]

    merged_config = {}

    for config_file in config_files:
        config_path = config_dir / config_file

        if config_path.exists():
            config = load_config(config_path)
            merged_config = merge_configs(merged_config, config)
        else:
            print(f"Warning: Configuration file not found: {config_path}")

    return merged_config


def get_config_value(
    config: Dict,
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get configuration value using dot-notation path.

    Example:
        get_config_value(config, 'model.transformer.num_heads', default=8)

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def set_config_value(
    config: Dict,
    key_path: str,
    value: Any
) -> Dict:
    """
    Set configuration value using dot-notation path.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set

    Returns:
        Modified configuration dictionary
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value

    return config


def validate_config(config: Dict) -> bool:
    """
    Validate configuration dictionary.

    Checks for required keys and valid values.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for required top-level keys
    required_keys = ['training']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate training config
    training_config = config.get('training', {})

    if 'num_epochs' in training_config:
        if training_config['num_epochs'] <= 0:
            raise ValueError("num_epochs must be positive")

    if 'batch_size' in training_config:
        if training_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")

    return True


def print_config(config: Dict, indent: int = 0) -> None:
    """
    Print configuration in a formatted way.

    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")
