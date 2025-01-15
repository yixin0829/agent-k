from typing import Any

from openai import OpenAI

import agent_k.config.prompts as config_prompts
from agent_k.config.schemas import RelevantEntities


def extract_relevant_entities(question: str) -> RelevantEntities:
    client = OpenAI()
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


def extract_entities_from_pdf(
    pdf_path: str, question: str, relevant_entities: RelevantEntities
) -> dict[str, Any]:
    """
    Extract entities from a PDF file one by one using OpenAI Assistant. Aggregate the entities into a dictionary.
    """
    client = OpenAI()
    assistant = client.beta.assistants.create(
        name="MinMod Assistant",
        instructions=config_prompts.PDF_AGENT_SYSTEM_PROMPT,
        tools=[
            {"type": "file_search", "file_ids": [pdf_path]},
            {"type": "code_interpreter"},
        ],
        model="gpt-4o-mini",
    )
    thread = client.beta.threads.create()
    run = client.beta.threads.runs.create_and_poll(
        instructions="Extract the total tonnage for all mineral resources and reserves with the original unit used in the report.",
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        print(messages)
    else:
        print(run.status)

    return {"total_tonnage": messages}


if __name__ == "__main__":
    print(
        extract_relevant_entities(
            "What are all the mineral sites with a deposit type of U-M intrusion nickel-copper-PGE? Report record value, state or province, country, top 1 deposit type and total grade."
        )
    )
