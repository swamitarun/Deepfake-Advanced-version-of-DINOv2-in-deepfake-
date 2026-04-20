"""
Helper utilities — seed, device, logging, config loading.
"""

import os
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.getLogger(__name__).info(f"Random seed set to {seed}")


def get_device(preferred: str = 'cuda') -> torch.device:
    """
    Get the best available device.

    Args:
        preferred: Preferred device ('cuda' or 'cpu').

    Returns:
        torch.device object.
    """
    logger = logging.getLogger(__name__)

    if preferred == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_filename: str = 'training.log',
):
    """
    Configure logging to both console and file.

    Args:
        log_dir: Directory for log file. If None, logs only to console.
        log_level: Logging level.
        log_filename: Name of the log file.
    """
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_path}")


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logging.getLogger(__name__).info(f"Loaded config from {config_path}")
    return config
