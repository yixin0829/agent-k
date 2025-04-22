import json
import random
import re
from datetime import datetime
from typing import Any

import httpx
import pandas as pd
from openai import OpenAI
from openai.types.beta import Assistant
from tqdm import tqdm

from agent_k.config.logger import logger
from agent_k.config.schemas import MinModHyperCols

CLIENT = OpenAI()


def download_file(url, path):
    client = httpx.Client(verify=False)
    with client.stream("GET", url) as r:
        size = int(r.headers.get("content-length", 0)) or None
        with (
            tqdm(total=size, unit="iB", unit_scale=True) as p_bar,
            open(path, "wb") as f,
        ):
            for data in r.iter_bytes():
                p_bar.update(len(data))
                f.write(data)


def sample_values_from_df(
    df: pd.DataFrame, column: str, n: int | tuple = 1
) -> str | list[str]:
    """Sample non-Unknown, non-null values from a column in a dataframe.

    Args:
        df: DataFrame to sample from
        column: Column name to sample from
        n: If int, sample exactly n values. If tuple of (min_n, max_n), sample random number of values in that range.

    Returns:
        If n=1, returns single sampled value as string.
        Otherwise returns list of sampled values.
    """
    unique_values = (
        df[~df[column].isin(["Unknown"]) & ~df[column].isna()][column].unique().tolist()
    )

    if isinstance(n, tuple):
        min_n, max_n = n
        n = random.randint(min_n, max_n)

    sampled = random.sample(unique_values, n)
    return sampled[0] if n == 1 else sampled


def load_list_to_df(data: list[list[str]], selected_cols: list[str]) -> pd.DataFrame:
    """Load a list of lists into a DataFrame with selected columns.

    Args:
        data: List of lists to load into DataFrame
        selected_columns: List of columns to select from the data

    Returns:
        DataFrame with selected columns and type conversion.
    """
    # TODO: address the ValueError during agent runtime (e.g. check columns in the generated SQL = columns in the question)
    try:
        df = pd.DataFrame(data, columns=selected_cols, dtype="object")
    except ValueError as e:
        logger.error(f"Error loading data to DataFrame: {e}")
        logger.error("Returning empty DataFrame as a fallback")
        logger.debug(f"Debugging info:\n{data=}\n{selected_cols=}")
        return pd.DataFrame()

    # Convert certain columns to float for easy merge
    for col in selected_cols:
        if col in [MinModHyperCols.TOTAL_GRADE, MinModHyperCols.TOTAL_TONNAGE]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_current_timestamp():
    """Get the current timestamp in the format YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def split_json_schema(
    schema: dict, field_lists: list[list[str]], include_defs: bool = True
) -> list[dict]:
    """
    Splits a JSON schema into multiple schemas based on provided field lists,
    optionally including only the definitions referenced in each schema.

    Args:
        schema (dict): The input JSON schema.
        field_lists (list of list of str): Each inner list contains fields for a separate schema.
        include_defs (bool): If True, include referenced definitions from the original schema's $defs.

    Returns:
        list of dict: A list of JSON schemas corresponding to each field list.
    """
    # Extract properties, required fields, and definitions from the original schema
    original_properties = schema.get("properties", {})
    original_required = set(schema.get("required", []))
    original_defs = schema.get("$defs", {})

    schemas = []
    for field_list in field_lists:
        new_schema = {"type": "object", "properties": {}, "required": []}
        # Build the new schema based on the field list
        for field in field_list:
            if field in original_properties:
                new_schema["properties"][field] = original_properties[field]
            if field in original_required:
                new_schema["required"].append(field)

        # Remove 'required' key if it's empty
        if not new_schema["required"]:
            new_schema.pop("required")

        # Optionally include only the definitions referenced in the new schema
        if include_defs and original_defs:
            new_defs = {}
            for prop in new_schema["properties"].values():
                if "$ref" in prop:
                    # Expecting ref format: "#/$defs/DefinitionName"
                    ref_parts = prop["$ref"].split("/")
                    if len(ref_parts) >= 3 and ref_parts[1] == "$defs":
                        def_name = ref_parts[2]
                        if def_name in original_defs:
                            new_defs[def_name] = original_defs[def_name]
            if new_defs:
                new_schema["$defs"] = new_defs

        schemas.append(new_schema)

    return schemas


def parse_json_code_block(content: str) -> dict[str, Any]:
    """Parse the JSON code block from the assistant response."""
    # Use regex to find the JSON code block
    json_code_block = re.search(r"<json>(.*?)</json>", content, re.DOTALL)
    if json_code_block:
        return json.loads(json_code_block.group(1))
    else:
        logger.error(f"Failed to parse JSON code block: {content}")
        return {}


def prompt_openai_assistant(assistant: Assistant, messages: list[dict]) -> str:
    thread = CLIENT.beta.threads.create(messages=messages)
    logger.info(f"Thread ID: {thread.id}")

    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.
    run = CLIENT.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == "completed":
        messages = list(CLIENT.beta.threads.messages.list(thread_id=thread.id))
    try:
        message_content = messages[0].content[0].text
    except IndexError as e:
        logger.exception(f"{e}.`messages`: {messages}")

    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(
            annotation.text, f"[{index}]"
        )
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = CLIENT.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    logger.debug(message_content.value)
    # logger.debug("\n".join(citations))

    return message_content.value
