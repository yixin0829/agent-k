import os
import sys
from datetime import datetime

from loguru import logger

# Create the logs directory if it doesn't exist
os.makedirs(".logs", exist_ok=True)

# Configure loguru
logger.remove()  # Remove the default handler

# Add handler for stderr (console)
logger.add(sys.stderr, level="DEBUG", colorize=True)

# Add handler for logging to a file with rotation and retention
logger.add(
    f".logs/{datetime.now().strftime('%Y-%m-%d')}.log",  # Log file path based on the current date and time
    level="DEBUG",
    rotation="10 MB",  # Rotate log file every 10 MB
    retention="7 days",  # Keep logs for 7 days
    compression="zip",  # Compress logs after rotation
)

# Export the configured logger
__all__ = ["logger"]
