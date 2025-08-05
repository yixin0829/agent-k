from typing import Optional

import chromadb
import tiktoken
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

import agent_k.config.experiment_config as config_experiment
from agent_k.config.logger import logger


def count_tokens(text: str, encoder: str = "cl100k_base") -> int:
    """Count the number of tokens in a text using the specified encoder."""
    encoding = tiktoken.get_encoding(encoder)
    return len(encoding.encode(text))


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
        headers_to_split_on: List of tuples containing markdown header levels and their names
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
