import os
from typing import Optional, Type

import chromadb
import litellm
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
from pydantic import BaseModel

import src.config.experiment_config as config_experiment
from src.config.logger import logger
from src.utils.general import count_tokens


def create_markdown_retriever(
    markdown_path: str,
    collection_name: str,
    headers_to_split_on: Optional[list[tuple[str, str]]] = None,
    embedding_model: str = "text-embedding-3-small",
    k: int = config_experiment.NUM_RETRIEVED_DOCS,
    max_tokens_per_batch: int = 250000,
) -> VectorStoreRetriever:
    """
    Creates a Chroma retriever from a markdown document.

    Args:
        markdown_path: Path to the markdown file
        collection_name: Name for the Chroma collection
        headers_to_split_on: list of tuples containing markdown header levels and their names
        embedding_model: Name of the OpenAI embedding model to use

    Returns:
        Chroma retriever object
    """
    # Set default headers if none provided
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    # Read markdown file
    try:
        with open(markdown_path, "r", encoding="utf-8") as file:
            markdown_document = file.read()
    except Exception as e:
        logger.error(f"Error reading markdown file: {e}")
        raise

    # MD splitter split document
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    doc_splits = markdown_splitter.split_text(markdown_document)

    # Char-level splits to further control the number of tokens per split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens_per_batch // 4,
        chunk_overlap=100,
    )
    for i, doc in enumerate(doc_splits):
        doc_tokens = count_tokens(str(doc))
        if doc_tokens > max_tokens_per_batch:
            logger.warning(f"Split is larger than max_tokens_per_batch: {doc_tokens}")
            splits = text_splitter.split_documents([doc])
            doc_splits.pop(i)
            doc_splits.extend(splits)

    # Log splitting information
    try:
        markdown_document_tokens = count_tokens(markdown_document)
        doc_splits_len = len(doc_splits)
        avg_tokens = markdown_document_tokens / doc_splits_len

        logger.info(f"Number of tokens: {markdown_document_tokens}")
        logger.info(f"Number of splits: {doc_splits_len}")
        logger.info(f"Average tokens per split: {avg_tokens:.0f}")
    except Exception as e:
        logger.warning(f"Could not log token statistics: {e}")

    # Create vectorstore and retriever
    try:
        # Initialize the Chroma client
        client = chromadb.Client()

        # Hash the collection name to determine which of 10 collections to use
        collection_hash = hash(collection_name)
        collection_index = abs(collection_hash) % 10  # Get value between 0-9
        hashed_collection_name = f"rag-chroma_shard_{collection_index}"
        logger.info(f"Hashed collection name: {hashed_collection_name}")

        # Delete existing in-memory collection if it exists
        try:
            client.delete_collection(hashed_collection_name)
            logger.info(f"Deleted existing collection: {hashed_collection_name}")
        except Exception:
            pass

        # Create a new vectorstore with persist_directory=None to keep it in-memory
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = Chroma(
            client=client,
            collection_name=hashed_collection_name,
            embedding_function=embeddings,
            persist_directory=None,  # Keep in-memory to avoid persistence
        )

        pointer, batch, batch_token_count = 0, [], 0

        while pointer < len(doc_splits):
            batch.append(doc_splits[pointer])
            batch_tokens = count_tokens(str(doc_splits[pointer]))
            batch_token_count += batch_tokens
            if batch_token_count > max_tokens_per_batch:
                # Remove the last appended doc that caused overflow
                batch.pop()
                logger.info(
                    f"Batch token count: {batch_token_count - batch_tokens}. Add batch to vectorstore and reset batch."
                )
                vectorstore.add_documents(documents=batch)
                batch, batch_token_count = [], 0
                # Add the current doc to start the new batch
                batch.append(doc_splits[pointer])
                batch_tokens = count_tokens(str(doc_splits[pointer]))
                batch_token_count = batch_tokens
            pointer += 1
        vectorstore.add_documents(documents=batch)

        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


# --------------------------------------------------------------------------------------
# Provider-agnostic model invocation helpers (shared across experiments)
# --------------------------------------------------------------------------------------


class ModelProviderError(Exception):
    """Raised when a model provider call fails in a non-recoverable way."""


class ContextLengthExceededError(Exception):
    """Raised when the input context exceeds a model's maximum context length."""


# Initialize provider clients
hugging_face_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN"),
)

# Avoid passing unsupported params to some providers
litellm.drop_params = True


def detect_provider_and_remote_model(model_name: str) -> tuple[str, str]:
    """Detect provider and map to the remote model identifier if needed.

    Returns a tuple of (provider, remote_model_name).
    Provider can be one of: "openai", "hugging_face".
    """

    model_name_lower = model_name.strip().lower()

    # HF router via OpenAI-compatible API
    if (
        "gpt-oss-20b" in model_name_lower
        or "qwen3" in model_name_lower
        or "llama" in model_name_lower
        or "deepseek" in model_name_lower
        or "qwen" in model_name_lower
    ):
        return "hugging_face", model_name

    # Default to OpenAI (or OpenAI-compatible) through litellm
    return "openai", model_name


def invoke_model_messages(
    model_name: str, messages: list[dict[str, str]], temperature: float
) -> str:
    """Invoke a chat model across multiple providers and return assistant text.

    Args:
        model_name: Friendly model name.
        messages: Array of chat messages with roles {system,user,assistant}.
        temperature: Decoding temperature.

    Returns:
        Assistant message content as a string.
    """

    provider, remote = detect_provider_and_remote_model(model_name)

    if provider == "openai":
        response = litellm.completion(
            model=remote, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content

    if provider == "hugging_face":
        try:
            result = hugging_face_client.chat.completions.create(
                model=remote,
                messages=messages,
                temperature=temperature,
            )
            return result.choices[0].message.content or ""
        except Exception as e:
            msg = str(e).lower()
            if "maximum context" in msg or "context length" in msg:
                raise ContextLengthExceededError(str(e))
            raise ModelProviderError(f"Hugging Face call failed: {e}")

    raise ModelProviderError(f"Unknown provider for model: {model_name}")


def invoke_json(
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    schema: Type[BaseModel],
) -> BaseModel:
    """Invoke a chat model and parse the response into a Pydantic model.

    This helper calls invoke_model_messages and then attempts to parse the first
    top-level JSON object or array from the response into the provided schema.

    Args:
        model_name: Friendly model name to route to a provider.
        messages: Chat messages in {role, content} format.
        temperature: Decoding temperature.
        schema: Pydantic model class to validate against.
        strict: If True, raise an exception on parse/validation failure. If False,
            return a best-effort fallback by directly attempting to json.loads the
            entire string and validate; if that fails, raise.

    Returns:
        Parsed Pydantic model instance.
    """

    import json as _json

    raw = invoke_model_messages(
        model_name=model_name, messages=messages, temperature=temperature
    )
    logger.debug(f"Raw response:\n{raw}")

    def _extract_first_json_segment(text: str) -> str:
        cleaned = text.strip()
        # Remove common code fences if present
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        last_close = cleaned.rfind("}")
        if last_close == -1:
            raise ValueError("No closing '}' found in text for JSON extraction.")
        # Scan backwards for the first '{' before last_close
        first_open = cleaned.rfind("{", 0, last_close + 1)
        if first_open == -1:
            raise ValueError("No opening '{' found in text for JSON extraction.")
        json_str = cleaned[first_open : last_close + 1]
        # Remove all indentation and newlines
        compact_json = "".join(line.strip() for line in json_str.splitlines())
        logger.debug(f"Compact JSON:\n{compact_json}")
        return compact_json

    try:
        payload = _extract_first_json_segment(raw)
        data = _json.loads(payload)
        return schema.model_validate(data)
    except Exception:
        raise ValueError(
            f"Failed to parse model response into the expected schema.\nSchema: {schema}\nData: {data}"
        )
