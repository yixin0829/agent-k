import json
import re
from datetime import datetime
from typing import Any

import tiktoken

from src.config.logger import logger


def get_curr_ts():
    """Get the current timestamp in the format YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_json_code_block(content: str) -> dict[str, Any]:
    """Parse the JSON code block from the assistant response."""
    # Use regex to find the JSON code block
    json_code_block = re.search(r"<json>(.*?)</json>", content, re.DOTALL)
    if json_code_block:
        return json.loads(json_code_block.group(1))
    else:
        logger.error(f"Failed to parse JSON code block: {content}")
        return {}


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text: The text containing the XML.
        tag: The XML tag to extract content from.

    Returns:
        The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def count_tokens(text: str, encoder: str = "cl100k_base") -> int:
    """Count the number of tokens in a text using the specified encoder."""
    encoding = tiktoken.get_encoding(encoder)
    return len(encoding.encode(text))
