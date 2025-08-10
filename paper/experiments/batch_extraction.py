# %% [markdown]
# ## Long-Context Batch Extraction
#

# %%
import json
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import litellm
import pandas as pd
from google import genai
from openai import OpenAI
from pydantic import BaseModel

import agent_k.config.experiment_config as config_experiment
import agent_k.config.general as config_general
import agent_k.config.prompts_fast_n_slow as prompts_fast_n_slow
from agent_k.config.logger import logger
from agent_k.config.schemas import create_dynamic_mineral_model
from paper.experiments.utils import count_tokens, create_markdown_retriever

hf_router_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

google_genai_client = genai.Client()

litellm.drop_params = True  # Ignore temperature parameter if model doesn't support it


# --------------------------------------------------------------------------------------
# Provider-agnostic model invocation and structured-output helpers
# --------------------------------------------------------------------------------------


class ContextLengthExceededError(Exception):
    """Raised when the input context exceeds a model's maximum context length."""


class ModelProviderError(Exception):
    """Raised when a model provider call fails in a non-recoverable way."""


class StructuredOutputParseError(Exception):
    """Raised when the model's output cannot be parsed into the expected schema."""


def _detect_provider_and_remote_model(model_name: str) -> Tuple[str, str]:
    """Detect provider and map to the remote model identifier if needed.

    Returns a tuple of (provider, remote_model_name).
    Provider can be one of: "openai", "hf_router", "gemini".
    """

    name = model_name.strip()
    name = name.lower()

    # Gemini
    if name.startswith("gemini-"):
        return "gemini", name

    # HF router via OpenAI-compatible API
    if "gpt-oss-20b" in name:
        return "hf_router", "openai/gpt-oss-20b:groq"
    if "llama-3.3-70b" in name:
        return "hf_router", "meta-llama/Llama-3.3-70B-Instruct:groq"

    # Default to OpenAI (or OpenAI-compatible) through litellm
    return "openai", name


def _json_schema_from_model(pydantic_model: BaseModel) -> Dict[str, Any]:
    """Generate a JSON schema from a Pydantic v2 model class."""

    return pydantic_model.model_json_schema()


def _build_schema_guided_user_prompt(context: Any, pydantic_model: BaseModel) -> str:
    """Create a schema-guided user prompt for providers without native structured output."""

    schema = _json_schema_from_model(pydantic_model)
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    if isinstance(context, str):
        normalized_context = context
    else:
        normalized_context = ""
        for item in context:
            normalized_context += str(item) + "\n"

    return (
        "Extract the requested structured information from the provided context.\n\n"
        "Output rules:\n"
        "- Output MUST be valid JSON only, with no code fences, no extra text, no comments.\n"
        "- The JSON MUST conform to the following JSON Schema (Pydantic v2):\n"
        f"{schema_str}\n"
        "- Use the exact field names as specified.\n"
        "- Use 'N/A' for missing string values. Use 0 for missing numeric values. Use empty arrays where appropriate.\n"
        "- Do not include any explanatory text before or after the JSON.\n\n"
        "Context begins:\n"
        f"{normalized_context}\n"
        "Context ends.\n"
    )


def _extract_first_json(text: str) -> str:
    """Extract the first top-level JSON object or array from free-form text."""

    cleaned = text.strip()
    cleaned = re.sub(r"```(json)?", "", cleaned).strip()

    # Find first '{' or '['
    indices = [i for i in [cleaned.find("{"), cleaned.find("[")] if i != -1]
    start_idx = min(indices) if indices else 0
    candidate = cleaned[start_idx:]

    def _balanced_slice(s: str) -> Optional[str]:
        if not s:
            return None
        stack: List[str] = []
        open_to_close = {"{": "}", "[": "]"}
        openers = set(open_to_close.keys())
        closers = set(open_to_close.values())
        started = False
        for i, ch in enumerate(s):
            if ch in openers:
                stack.append(open_to_close[ch])
                started = True
            elif ch in closers and stack:
                expected = stack.pop()
                if ch != expected:
                    # mismatched, keep scanning
                    pass
                if not stack and started:
                    return s[: i + 1]
        return None

    sliced = _balanced_slice(candidate)
    return sliced.strip() if sliced else cleaned


def _invoke_model_messages(model_name: str, messages: List[Dict[str, str]]) -> str:
    """Invoke a chat model across multiple providers and return assistant text."""

    provider, remote = _detect_provider_and_remote_model(model_name)

    if provider == "openai":
        response = litellm.completion(model=remote, messages=messages)
        return response.choices[0].message.content

    if provider == "hf_router":
        try:
            result = hf_router_client.chat.completions.create(
                model=remote,
                messages=messages,
                temperature=config_experiment.BATCH_EXTRACTION_TEMPERATURE,
            )
            return result.choices[0].message.content or ""
        except Exception as e:
            msg = str(e).lower()
            if (
                "maximum context" in msg
                or "context length" in msg
                or "too many tokens" in msg
            ):
                raise ContextLengthExceededError(str(e))
            raise ModelProviderError(f"HF Router call failed: {e}")

    if provider == "gemini":
        try:
            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            user_parts = [m["content"] for m in messages if m["role"] == "user"]
            prompt = "\n\n".join(system_parts + user_parts)
            response = google_genai_client.models.generate_content(
                model=remote,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=config_experiment.BATCH_EXTRACTION_TEMPERATURE
                ),
            )
            return getattr(response, "text", "")
        except Exception as e:
            msg = str(e).lower()
            if (
                "maximum context" in msg
                or "context length" in msg
                or "too many tokens" in msg
            ):
                raise ContextLengthExceededError(str(e))
            raise ModelProviderError(f"Gemini call failed: {e}")

    raise ModelProviderError(f"Unknown provider for model: {model_name}")


def _supports_native_structured_output(model_name: str) -> bool:
    """Return True if we should leverage native structured outputs (OpenAI via litellm)."""

    provider, _ = _detect_provider_and_remote_model(model_name)
    return provider == "openai"


_MODEL_MAX_TOKENS: Dict[str, int] = {
    # Heuristic limits; provider errors remain the source of truth
    "gpt-oss-20b": 128000,
    "llama-3.3-70b": 128000,
    "gemini-2.5-flash": 2000000,
}


def _context_maybe_too_long(model_name: str, input_tokens: int) -> bool:
    """Heuristic pre-check for overly long inputs when known."""

    for key, max_tokens in _MODEL_MAX_TOKENS.items():
        if key.lower() in model_name.lower():
            return input_tokens > max_tokens - 2048  # leave headroom
    return False


# %%
def batch_extract_long_context(
    cdr_record_id: str,
    pydantic_model: BaseModel,
) -> Tuple[str, int, int]:
    """
    Extract entities from a PDF file by feeding the entire context to the model.
    """

    # Read with the parsed Markdown file
    with open(
        os.path.join("paper/data/processed/43-101-refined", f"{cdr_record_id}.md"), "r"
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
        input_payload: Dict[str, Any] = {
            "model": model_name,
            "response_format": pydantic_model,
            "messages": [
                {
                    "role": "system",
                    "content": prompts_fast_n_slow.PDF_AGENT_SYSTEM_PROMPT_STRUCTURED,
                },
                {
                    "role": "user",
                    "content": prompts_fast_n_slow.PDF_AGENT_USER_PROMPT_STRUCTURED.format(
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
            if (
                "maximum context" in msg
                or "context length" in msg
                or "too many tokens" in msg
            ):
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
        try:
            content = _invoke_model_messages(model_name=model_name, messages=messages)
        except ContextLengthExceededError:
            logger.warning(
                f"[Batch Extraction] Context too long for {model_name} on {cdr_record_id} (invoke)."
            )
            raise

        try:
            content = _extract_first_json(content)
            json.loads(content)
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
) -> Tuple[str, int, int]:
    """
    Use a retriever to compress the context to be more relevant before feeding it to the model to extract structured information.
    """

    # Create a Vector Store from markdown file
    markdown_path = os.path.join(
        "paper/data/processed/43-101-refined", f"{cdr_record_id}.md"
    )
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
        question = prompts_fast_n_slow.QUESTION_TEMPLATE.format(
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
        input_payload: Dict[str, Any] = {
            "model": model_name,
            "response_format": pydantic_model,
            "messages": [
                {
                    "role": "system",
                    "content": prompts_fast_n_slow.PDF_AGENT_SYSTEM_PROMPT_STRUCTURED,
                },
                {
                    "role": "user",
                    "content": prompts_fast_n_slow.PDF_AGENT_USER_PROMPT_STRUCTURED.format(
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
        content = _invoke_model_messages(model_name=model_name, messages=messages)
        try:
            content = _extract_first_json(content)
            json.loads(content)
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
    gt_path: str, output_dir: str, setup: config_experiment.BatchExtractionMethod
):
    df_gt = pd.read_csv(gt_path)

    rows = []
    tokens = defaultdict(int)
    start_time = time.time()
    for index, row in df_gt.iterrows():
        # if index > 1:
        #     break
        logger.info(f"Processing row {index + 1} of {len(df_gt)}")
        id = row["id"]
        cdr_record_id = row["cdr_record_id"]
        commodity_observed_name = row["commodity_observed_name"]

        try:
            pydantic_model = create_dynamic_mineral_model(commodity_observed_name)
            output, input_tokens, output_tokens = batch_extract_func_map[setup](
                cdr_record_id, pydantic_model
            )
            mineral_site_metadata = pydantic_model.model_validate_json(output)
            mineral_site_metadata = {
                "id": id,
                "cdr_record_id": cdr_record_id,
                **mineral_site_metadata.model_dump(),
            }
            rows.append(mineral_site_metadata)

            # Track tokens
            tokens["input_tokens"] += input_tokens
            tokens["output_tokens"] += output_tokens
        except Exception as e:
            logger.exception(
                f"Error processing row {index + 1} of {len(df_gt)} (Skipping): {e}"
            )

    end_time = time.time()

    df_pred = pd.DataFrame(rows)
    logger.info(
        f"Successfully extracted {len(df_pred)} reports. Write to experiment results directory."
    )
    # Generate timestamp once for both files
    os.makedirs(output_dir, exist_ok=True)

    # Convert unit from tonnes to Mt
    numerical_columns = [
        "total_mineral_resource_tonnage",
        "total_mineral_reserve_tonnage",
        "total_mineral_resource_contained_metal",
        "total_mineral_reserve_contained_metal",
    ]
    for column in numerical_columns:
        df_pred[column] = df_pred[column].astype(float) / 1e6

    # Save results to CSV
    timestamp = config_general.get_curr_ts()
    df_pred.to_csv(f"{output_dir}/{timestamp}_batch_extraction.csv", index=False)

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

    # Save experiment metadata to yaml file
    import yaml

    max_num_results = (
        config_experiment.MAX_NUM_RETRIEVED_DOCS
        if setup == config_experiment.BatchExtractionMethod.RAG_BASED
        else "N/A (long context)"
    )

    experiment_metadata = {
        "timestamp": timestamp,
        "setup": setup,
        "model": config_experiment.BATCH_EXTRACTION_MODEL,
        "temperature": config_experiment.BATCH_EXTRACTION_TEMPERATURE,
        "max_num_results": max_num_results,
        "total_time_seconds": end_time - start_time,
        "total_input_tokens": tokens["input_tokens"],
        "total_output_tokens": tokens["output_tokens"],
        "average_input_tokens_per_row": tokens["input_tokens"] / len(df_gt),
        "average_output_tokens_per_row": tokens["output_tokens"] / len(df_gt),
        "num_rows_processed": len(df_pred),
        "total_rows": len(df_gt),
    }
    with open(f"{output_dir}/{timestamp}_experiment_metadata.yaml", "w") as f:
        yaml.dump(experiment_metadata, f)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # Note: Hyperparameters (configured manually in config/experiment_config.py)
    # ----------------------------------------------------------------------------------
    # Models: [
    #     "gpt-4o-mini", "gpt-4.1", "o4-mini", "Llama-3.3-70B-Instruct",
    #     "gpt-oss-20b", "gemini-2.5-flash"
    # ]
    # Setups: ["long_context", "rag_based"]
    # Temperature: [0.1]
    # Max retrieved results: [5] --> only applicable to rag_based

    # output_dir = "paper/data/experiments/long_context_batch_extraction"
    output_dir = "paper/data/experiments/rag_batch_extraction"
    gt_path = "paper/data/processed/ground_truth/inferlink_ground_truth.csv"

    run_experiment(
        gt_path=gt_path, output_dir=output_dir, setup=config_experiment.BATCH_METHOD
    )
