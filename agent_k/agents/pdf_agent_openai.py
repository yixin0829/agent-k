import builtins
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI
from pydantic import Field, create_model

import agent_k.config.general as config_general
import agent_k.config.prompts as config_prompts
from agent_k.config.logger import logger
from agent_k.config.schemas import (
    DataSource,
    MinModHyperCols,
    RelevantEntities,
    RelevantEntitiesPredefined,
)
from agent_k.setup.load_43_101 import list_43_101_reports

client = OpenAI()


def str_to_type(type_name: str):
    """Convert a string like 'int' to its corresponding Python type."""
    return getattr(builtins, type_name)


def parse_json_code_block(assistant_response: str) -> dict[str, Any]:
    """Parse the JSON code block from the assistant response."""
    try:
        json_code_block = assistant_response.split("```json")[1].split("```")[0]
        return json.loads(json_code_block)
    except Exception as e:
        logger.error(f"Failed to parse JSON code block: {e}")
        logger.error(f"Assistant response: \n```\n{assistant_response}\n```")
        return {}


def extract_relevant_entities(question: str) -> RelevantEntities:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": config_prompts.ENTITY_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": config_prompts.ENTITY_EXTRACTION_USER_PROMPT.format(
                    question=question
                ),
            },
        ],
        response_format=RelevantEntities,
    )

    relevant_entities = completion.choices[0].message.parsed
    return relevant_entities


def extract_from_pdf(
    pdf_path: str,
    relevant_entities: RelevantEntities | dict,
) -> dict[str, Any]:
    """
    Extract entities from a PDF file one by one using OpenAI Assistant. Aggregate the entities into a dictionary.
    Note: If relevant_entities is a string, it is used as the JSON schema directly.
    """
    if isinstance(relevant_entities, dict):
        relevant_entities_json_schema = relevant_entities
    elif isinstance(relevant_entities, RelevantEntities):
        # Construct a Pydantic model from the relevant entities. The key is the entity name, and the value is the entity description. The default value is None.
        DynamicRelevantEntities = create_model(
            "DynamicRelevantEntities",
            **{
                entity.entity_name: (
                    str_to_type(entity.entity_data_type),
                    Field(default="Not Found", description=entity.entity_description),
                )
                for entity in relevant_entities.entities
            },
        )
        relevant_entities_json_schema = DynamicRelevantEntities.model_json_schema()
    else:
        raise ValueError(f"Invalid relevant entities type: {type(relevant_entities)}")

    logger.debug(f"Relevant entities JSON schema: {relevant_entities_json_schema}")

    logger.info("Creating OpenAI assistant")
    assistant = client.beta.assistants.create(
        name="MinMod Assistant",
        instructions=config_prompts.PDF_AGENT_SYSTEM_PROMPT,
        tools=[
            {"type": "file_search"},
            {"type": "code_interpreter"},
        ],
        model="gpt-4o-mini",
    )

    logger.info("Creating thread with user message and file")
    filename_to_id = list_43_101_reports()
    filename = pdf_path.split("/")[-1]
    file_id = filename_to_id[filename]
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": config_prompts.PDF_AGENT_USER_PROMPT.format(
                    relevant_entities_json_schema=relevant_entities_json_schema,
                ),
                "attachments": [
                    {
                        "file_id": file_id,
                        "tools": [{"type": "file_search"}],
                    }
                ],
            },
        ],
    )

    logger.info("Starting assistant run")

    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    messages = list(
        client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
    )

    try:
        message_content = messages[0].content[0].text
    except IndexError:
        logger.error(
            f"No message content found for thread {thread.id} and run {run.id}. {messages=}"
        )
        logger.error("Returning empty dictionary")
        return {}

    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(
            annotation.text, f"[{index}]"
        )
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    print(message_content.value)
    print("\n".join(citations))

    return parse_json_code_block(message_content.value)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def resolve_entities(entities: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve the entities from the PDF agent extraction by comparing embeddings.
    Uses cached embeddings for entity candidates to improve performance.
    """
    logger.info("Resolving entities")
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )
    # Select 43-101 and MRDS data sources
    df_hyper = df_hyper[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isin(
            [DataSource.MRDATA_USGS_GOV_MRDS.value, DataSource.API_CDR_LAND.value]
        )
    ]
    entities_to_resolve = [
        MinModHyperCols.MINERAL_SITE_NAME.value,
        MinModHyperCols.STATE_OR_PROVINCE.value,
        MinModHyperCols.COUNTRY.value,
        MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
        MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
    ]

    # Get unique candidates and their embeddings for each entity type
    entity_candidates = defaultdict(list)
    entity_embeddings = defaultdict(list)
    entity_embeddings_file = os.path.join(
        config_general.PDF_AGENT_CACHE_DIR, "entity_embeddings.pkl"
    )
    entity_candidates_file = os.path.join(
        config_general.PDF_AGENT_CACHE_DIR, "entity_candidates.pkl"
    )
    if os.path.exists(entity_embeddings_file) and os.path.exists(
        entity_candidates_file
    ):
        logger.info("Loading entity embeddings and candidates from cache")
        with open(entity_embeddings_file, "rb") as f:
            entity_embeddings = pickle.load(f)
        with open(entity_candidates_file, "rb") as f:
            entity_candidates = pickle.load(f)
    else:
        logger.info("Computing entity embeddings and candidates")
        for entity in entities_to_resolve:
            unique_values = (
                df_hyper[
                    ~df_hyper[entity].isin(["Unknown"]) & ~df_hyper[entity].isna()
                ][entity]
                .unique()
                .tolist()
            )
            entity_candidates[entity] = unique_values

            # Pre-compute embeddings for all candidates
            unique_values = [str(v) for v in unique_values]
            BATCH_SIZE = 2048
            for i in range(0, len(unique_values), BATCH_SIZE):
                batch = unique_values[i : i + BATCH_SIZE]
                candidate_embeddings = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small",
                ).data
                entity_embeddings[entity].extend(
                    [c.embedding for c in candidate_embeddings]
                )

        # Save entity embeddings and candidates
        logger.info("Saving entity embeddings and candidates to cache")
        with open(entity_embeddings_file, "wb") as f:
            pickle.dump(entity_embeddings, f)
        with open(entity_candidates_file, "wb") as f:
            pickle.dump(entity_candidates, f)

    # Resolve entities using cached embeddings
    resolved_entities = {}
    for entity_name, entity_value in entities.items():
        if entity_name not in entities_to_resolve or not entity_value:
            resolved_entities[entity_name] = entity_value
            continue

        # Get embedding for the entity value
        value_embedding = (
            client.embeddings.create(
                input=str(entity_value),
                model="text-embedding-3-small",
            )
            .data[0]
            .embedding
        )

        # Calculate similarities using cached candidate embeddings
        similarities = [
            cosine_similarity(value_embedding, candidate_emb)
            for candidate_emb in entity_embeddings[entity_name]
        ]

        # Find most similar candidate
        max_sim_idx = np.argmax(similarities)
        max_sim = similarities[max_sim_idx]

        # Only use candidate if similarity is above threshold
        logger.info(
            f"Resolved {entity_name} from '{entity_value}' to '{entity_candidates[entity_name][max_sim_idx]}' with similarity {max_sim}"
        )
        resolved_entities[entity_name + "_resolved"] = entity_candidates[entity_name][
            max_sim_idx
        ]
        resolved_entities[entity_name + "_similarity"] = max_sim

    return resolved_entities


def extract_from_all_pdfs(
    mineral_report_dir: str = config_general.CDR_REPORTS_DIR, full_eval: bool = False
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files and return as a DataFrame
    """
    pdf_paths = []
    for i, pdf_path in enumerate(os.listdir(mineral_report_dir)):
        if i > 2 and not full_eval:
            break
        pdf_paths.append(os.path.join(mineral_report_dir, pdf_path))

    data_rows = []
    for i, pdf_path in enumerate(pdf_paths):
        logger.info(f"{i+1}/{len(pdf_paths)}: Extracting entities from {pdf_path}")
        entities = extract_from_pdf(
            pdf_path, RelevantEntitiesPredefined.model_json_schema()
        )
        if entities:
            resolved_entities = resolve_entities(entities)
            entities.update(resolved_entities)
            entities.update({"cdr_record_id": pdf_path.split("/")[-1].split(".")[0]})
            data_rows.append(entities)
        else:
            logger.error(f"Failed to extract entities from {pdf_path}")
            continue

    df = pd.DataFrame(data_rows)

    # Replace the "Not Found" values with None
    df = df.replace("Not Found", None)

    if not os.path.exists(config_general.PDF_AGENT_CACHE_DIR):
        logger.info(f"Creating directory {config_general.PDF_AGENT_CACHE_DIR}")
        os.makedirs(config_general.PDF_AGENT_CACHE_DIR)

    logger.info(f"Saving extraction results to {config_general.PDF_AGENT_CACHE_DIR}")
    df.to_csv(
        os.path.join(
            config_general.PDF_AGENT_CACHE_DIR,
            f"pdf_agent_extraction_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
        ),
        index=False,
    )

    return df


if __name__ == "__main__":
    # Example: Single extraction
    questions = [
        "What are all the mineral sites with a deposit type of U-M intrusion nickel-copper-PGE? Report mineral site name, state or province, country, top 1 deposit type, total grade and total tonnage."
    ]
    pdf_paths = [
        "/home/yixin0829/minmod/agent-k/data/raw/all_sources/43-101/02a000a83e76360bec8f3fce2ff46cc8099f950cc1f757f8a16592062c49b3a5c5.pdf"
    ]
    for question in questions:
        relevant_entities = extract_relevant_entities(question)
        for pdf_path in pdf_paths:
            entities = extract_from_pdf(
                pdf_path,
                relevant_entities,
            )

    # Example: Batch extraction
    # extract_from_all_pdfs(full_eval=False)
