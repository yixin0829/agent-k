# %% [markdown]
# ## Long-Context Batch Extraction
#

# %%
import json
import os
import time
from collections import defaultdict
from typing import Any

import litellm
import pandas as pd
import yaml
from pydantic import BaseModel

import src.config.experiment_config as config_experiment
import src.config.prompts as prompts
from src.config.logger import logger
from src.config.schemas import create_mineral_model_w_commodity
from src.utils.general import count_tokens, get_curr_ts
from src.utils.llm import (
    ContextLengthExceededError,
    create_markdown_retriever,
    detect_provider_and_remote_model,
    invoke_json,
)

# --------------------------------------------------------------------------------------
# Configuration Variables
# --------------------------------------------------------------------------------------
sample_size = config_experiment.BATCH_EXTRACTION_SAMPLE_SIZE
setup = config_experiment.BATCH_METHOD

# Model context limits (heuristic limits; provider errors remain the source of truth)
MODEL_MAX_TOKENS: dict[str, int] = {
    "gpt-oss-20b": 128000,
    "llama-3.3-70b": 128000,
}

# Numerical columns for unit conversion to million tonnes
NUMERICAL_COLUMNS = [
    "total_mineral_resource_tonnage",
    "total_mineral_reserve_tonnage",
    "total_mineral_resource_contained_metal",
    "total_mineral_reserve_contained_metal",
]

# File paths configuration
DATA_DIR = "data/processed/43-101_reports_refined"
GT_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth.csv"
LONG_CONTEXT_OUTPUT_DIR = "data/experiments/long_context_batch_extraction"
RAG_OUTPUT_DIR = "data/experiments/rag_batch_extraction"

# Note: batch_extract_func_map will be defined after the functions are declared

# --------------------------------------------------------------------------------------
# Provider-agnostic model invocation and structured-output helpers
# --------------------------------------------------------------------------------------


class ModelProviderError(Exception):
    """Raised when a model provider call fails in a non-recoverable way."""


class StructuredOutputParseError(Exception):
    """Raised when the model's output cannot be parsed into the expected schema."""


def _build_schema_guided_user_prompt(
    context: str | list[Any], pydantic_model: BaseModel
) -> str:
    """Create a schema-guided user prompt for providers without native structured output."""

    schema = pydantic_model.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    if isinstance(context, str):
        normalized_context = context
    else:
        normalized_context = ""
        for item in context:
            normalized_context += str(item) + "\n"

    return prompts.SCHEMA_GUIDED_USER_PROMPT.format(
        schema_str=schema_str, normalized_context=normalized_context
    )


def _supports_native_structured_output(model_name: str) -> bool:
    """Return True if we should leverage native structured outputs (OpenAI via litellm)."""

    provider, _ = detect_provider_and_remote_model(model_name)
    return provider == "openai"


def _context_maybe_too_long(model_name: str, input_tokens: int) -> bool:
    """Heuristic pre-check for overly long inputs when known."""

    for key, max_tokens in MODEL_MAX_TOKENS.items():
        if key.lower() in model_name.lower():
            return input_tokens > max_tokens - 2048  # leave headroom
    return False


# %%
def batch_extract_long_context(
    cdr_record_id: str,
    pydantic_model: BaseModel,
) -> tuple[str, int, int]:
    """
    Extract entities from a PDF file by feeding the entire context to the model.
    """

    # Read with the parsed Markdown file
    with open(
        os.path.join(DATA_DIR, f"{cdr_record_id}.md"),
        "r",
    ) as f:
        context = f.read()

    input_tokens = count_tokens(context)
    logger.info(f"[Batch Extraction] Context has {input_tokens} tokens")

    logger.info("[Batch Extraction] Creating structured response")

    model_name = config_experiment.BATCH_EXTRACTION_MODEL

    # Pre-check for context window where known. If too long, skip early.
    if _context_maybe_too_long(model_name, input_tokens):
        logger.warning(
            f"[Batch Extraction] Skipping {cdr_record_id}: input tokens {input_tokens} exceed likely context window for {model_name}."
        )
        raise ContextLengthExceededError(
            f"Input tokens {input_tokens} likely exceed context window for {model_name}."
        )

    if _supports_native_structured_output(model_name):
        input_payload: dict[str, Any] = {
            "model": model_name,
            "response_format": pydantic_model,
            "messages": [
                {
                    "role": "system",
                    "content": prompts.PDF_AGENT_SYSTEM_PROMPT_STRUCTURED,
                },
                {
                    "role": "user",
                    "content": prompts.PDF_AGENT_USER_PROMPT_STRUCTURED.format(
                        context=context
                    ),
                },
            ],
            "temperature": config_experiment.BATCH_EXTRACTION_TEMPERATURE,
        }

        try:
            response = litellm.completion(**input_payload)
            content = response.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "maximum context" in msg or "context length" in msg:
                logger.warning(
                    f"[Batch Extraction] Context too long for {model_name} on {cdr_record_id}: {e}"
                )
                raise ContextLengthExceededError(str(e))
            raise
    else:
        # Schema-guided textual prompting with post-parse
        user_prompt = _build_schema_guided_user_prompt(
            context=context, pydantic_model=pydantic_model
        )
        messages = [
            {
                "role": "system",
                "content": "You are a precise information extraction assistant that outputs strictly valid JSON only.",
            },
            {"role": "user", "content": user_prompt},
        ]
        # Prefer invoke_json to get validation and robust JSON extraction
        try:
            parsed = invoke_json(
                model_name=model_name,
                messages=messages,
                temperature=config_experiment.BATCH_EXTRACTION_TEMPERATURE,
                schema=pydantic_model,
            )
            content = parsed.model_dump_json()
        except ContextLengthExceededError:
            logger.warning(
                f"[Batch Extraction] Context too long for {model_name} on {cdr_record_id} (invoke)."
            )
            raise
        except Exception as e:
            raise StructuredOutputParseError(
                f"Failed to parse JSON from model output for {cdr_record_id}: {e}"
            )

    output_tokens = count_tokens(content)
    logger.info(f"[Batch Extraction] Response has {output_tokens} tokens")

    return content, input_tokens, output_tokens


# %% [markdown]
# ## RAG-Based Extraction
#
# Incorporate a retriever to compress the context to be more relevant.


# %%
def batch_extract_rag_based(
    cdr_record_id: str,
    pydantic_model: BaseModel,
) -> tuple[str, int, int]:
    """
    Use a retriever to compress the context to be more relevant before feeding it to the model to extract structured information.
    """

    # Create a Vector Store from markdown file
    markdown_path = os.path.join(DATA_DIR, f"{cdr_record_id}.md")
    logger.info(f"[Batch Extraction] Creating Vector Store for {cdr_record_id}")
    retriever = create_markdown_retriever(
        markdown_path=markdown_path,
        collection_name=cdr_record_id,
        k=config_experiment.MAX_NUM_RETRIEVED_DOCS,
    )

    # Retrieve context for each fields in the pydantic model
    context = []
    for field_name, field_info in pydantic_model.model_fields.items():
        field_type = field_info.annotation
        default_value = field_info.default if field_info.default else "N/A"
        description = field_info.description
        question = prompts.QUESTION_TEMPLATE.format(
            field=field_name,
            dtype=field_type,
            default=default_value,
            description=description,
        )
        context.extend(retriever.invoke(question))
    logger.info(f"[Batch Extraction] Retrieved chunks in context: {len(context)}")

    input_tokens = count_tokens(str(context))
    logger.info(f"[Batch Extraction] Context has {input_tokens} tokens")

    logger.info("[Batch Extraction] Creating structured response")

    model_name = config_experiment.BATCH_EXTRACTION_MODEL

    if _supports_native_structured_output(model_name):
        input_payload: dict[str, Any] = {
            "model": model_name,
            "response_format": pydantic_model,
            "messages": [
                {
                    "role": "system",
                    "content": prompts.PDF_AGENT_SYSTEM_PROMPT_STRUCTURED,
                },
                {
                    "role": "user",
                    "content": prompts.PDF_AGENT_USER_PROMPT_STRUCTURED.format(
                        context=context
                    ),
                },
            ],
            "temperature": config_experiment.BATCH_EXTRACTION_TEMPERATURE,
        }
        response = litellm.completion(**input_payload)
        content = response.choices[0].message.content
    else:
        user_prompt = _build_schema_guided_user_prompt(
            context=context, pydantic_model=pydantic_model
        )
        messages = [
            {
                "role": "system",
                "content": "You are a precise information extraction assistant that outputs strictly valid JSON only.",
            },
            {"role": "user", "content": user_prompt},
        ]
        try:
            parsed = invoke_json(
                model_name=model_name,
                messages=messages,
                temperature=config_experiment.BATCH_EXTRACTION_TEMPERATURE,
                schema=pydantic_model,
            )
            content = parsed.model_dump_json()
        except Exception as e:
            raise StructuredOutputParseError(
                f"Failed to parse JSON from model output for {cdr_record_id}: {e}"
            )
    output_tokens = count_tokens(content)
    logger.info(f"[Batch Extraction] Response has {output_tokens} tokens")

    return content, input_tokens, output_tokens


# %%
batch_extract_func_map = {
    config_experiment.BatchExtractionMethod.LONG_CONTEXT: batch_extract_long_context,
    config_experiment.BatchExtractionMethod.RAG_BASED: batch_extract_rag_based,
}


def run_experiment(
    gt_path: str,
    output_dir: str,
    setup: config_experiment.BatchExtractionMethod,
    sample_size: int | None = None,
):
    df_gt = pd.read_csv(gt_path)

    # For testing purpose. If None, extract from all PDF files
    if sample_size is not None:
        df_gt = df_gt.head(sample_size)

    # Log experiment hyperparameters
    logger.info(f"Batch extraction setup: {setup}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Total rows to process: {len(df_gt)}")
    logger.info(f"Output directory: {output_dir}")

    tokens = defaultdict(int)
    start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for output files
    timestamp = get_curr_ts()
    output_file_path = os.path.join(output_dir, f"{timestamp}_batch_extraction.csv")

    # Create empty DataFrame with headers and save to initialize CSV file
    # Use a sample to get the correct column structure
    sample_row = df_gt.iloc[0]
    sample_pydantic_model = create_mineral_model_w_commodity(
        sample_row["commodity_observed_name"]
    )
    sample_schema = sample_pydantic_model.model_json_schema()

    # Create column list based on the schema properties plus id and cdr_record_id
    columns = ["id", "cdr_record_id"] + list(sample_schema["properties"].keys())
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv(output_file_path, index=False)

    for index, row in df_gt.iterrows():
        logger.info(f"Processing row {index + 1} of {len(df_gt)}")
        id = row["id"]
        cdr_record_id = row["cdr_record_id"]
        commodity_observed_name = row["commodity_observed_name"]

        try:
            pydantic_model = create_mineral_model_w_commodity(commodity_observed_name)
            output, input_tokens, output_tokens = batch_extract_func_map[setup](
                cdr_record_id, pydantic_model
            )
            mineral_site_metadata = pydantic_model.model_validate_json(output)
            mineral_site_metadata = {
                "id": id,
                "cdr_record_id": cdr_record_id,
                **mineral_site_metadata.model_dump(),
            }

            # Convert to DataFrame and convert unit of numerical columns to million tonnes
            df_row = pd.DataFrame([mineral_site_metadata])
            for col in NUMERICAL_COLUMNS:
                if col in df_row.columns:
                    df_row[col] = pd.to_numeric(df_row[col], errors="coerce")
                    df_row[col] = df_row[col] / 1e6

            # Append to CSV file immediately
            df_row.to_csv(output_file_path, mode="a", header=False, index=False)

            # Track tokens
            tokens["input_tokens"] += input_tokens
            tokens["output_tokens"] += output_tokens
        except Exception as e:
            logger.exception(
                f"Error processing row {index + 1} of {len(df_gt)} (Skipping): {e}"
            )

    end_time = time.time()

    # Read the final CSV file to get the actual results
    df_pred = pd.read_csv(output_file_path)
    logger.info(
        f"Successfully extracted {len(df_pred)} reports. Results saved to {output_file_path}"
    )

    # Log experiment results
    logger.info(f"Experiment setup: {setup}")
    logger.info(f"Total time taken: {end_time - start_time} seconds")
    logger.info(f"Total input tokens: {tokens['input_tokens']}")
    logger.info(f"Total output tokens: {tokens['output_tokens']}")
    logger.info(
        f"Average input tokens per row: {tokens['input_tokens'] / len(df_gt):.2f}"
    )
    logger.info(
        f"Average output tokens per row: {tokens['output_tokens'] / len(df_gt):.2f}"
    )

    max_num_results = (
        config_experiment.MAX_NUM_RETRIEVED_DOCS
        if setup == config_experiment.BatchExtractionMethod.RAG_BASED
        else "N/A (long context)"
    )

    experiment_metadata = {
        "timestamp": timestamp,
        "setup": setup.value,
        "model": config_experiment.BATCH_EXTRACTION_MODEL,
        "temperature": config_experiment.BATCH_EXTRACTION_TEMPERATURE,
        "max_num_results": max_num_results,
        "sample_size": sample_size,
        "total_time_seconds": end_time - start_time,
        "total_input_tokens": tokens["input_tokens"],
        "total_output_tokens": tokens["output_tokens"],
        "average_input_tokens_per_row": tokens["input_tokens"] / len(df_gt),
        "average_output_tokens_per_row": tokens["output_tokens"] / len(df_gt),
        "num_rows_processed": len(df_pred),
        "total_rows": len(df_gt),
    }
    metadata_file_path = os.path.join(
        output_dir, f"{timestamp}_experiment_metadata.yaml"
    )
    with open(metadata_file_path, "w") as f:
        yaml.dump(experiment_metadata, f)


if __name__ == "__main__":
    if (
        config_experiment.BATCH_METHOD
        == config_experiment.BatchExtractionMethod.LONG_CONTEXT
    ):
        output_dir = LONG_CONTEXT_OUTPUT_DIR
    elif (
        config_experiment.BATCH_METHOD
        == config_experiment.BatchExtractionMethod.RAG_BASED
    ):
        output_dir = RAG_OUTPUT_DIR
    else:
        raise ValueError(
            f"Invalid batch extraction method: {config_experiment.BATCH_METHOD}"
        )

    run_experiment(
        gt_path=GT_PATH,
        output_dir=output_dir,
        setup=setup,
        sample_size=sample_size,
    )
