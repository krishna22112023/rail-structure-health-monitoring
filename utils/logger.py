import sys
import pyprojroot
from pathlib import Path
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

import logging
from logging import config
import yaml
import os

logger = logging.getLogger(__name__)

def setup_logging(
    logging_config_path: str = "./config/log_config.yaml",
    default_level: int = logging.INFO,
    exclude_handlers: list = [],
    use_log_filename_prefix: bool = False,
    log_filename_prefix: str = "",
):
    """Load a specified custom configuration for logging.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to the logging YAML configuration file, by default "./conf/logging.yaml"
    default_level : int, optional
        Default logging level to use if the configuration file is not found,
        by default logging.INFO
    """
    logging_config_path = str(Path(root,logging_config_path))
    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        if use_log_filename_prefix:
            for handler in log_config["handlers"]:
                if "filename" in log_config["handlers"][handler]:
                    curr_log_filename = log_config["handlers"][handler]["filename"]
                    log_config["handlers"][handler]["filename"] = os.path.join(
                        log_filename_prefix, curr_log_filename
                    )

        logging_handlers = log_config["root"]["handlers"]
        log_config["root"]["handlers"] = [
            handler for handler in logging_handlers if handler not in exclude_handlers
        ]

        # TODO: handle prefix edit here

        config.dictConfig(log_config)
        logger.info("Successfully loaded custom logging configuration.")

    except FileNotFoundError as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")