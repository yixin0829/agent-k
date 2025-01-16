import builtins
import json
from typing import Any

from openai import OpenAI
from pydantic import Field, create_model

import agent_k.config.prompts as config_prompts
from agent_k.config.logger import logger
from agent_k.config.schemas import RelevantEntities
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


if __name__ == "__main__":
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
