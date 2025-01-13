import asyncio
from typing import Annotated, Literal

from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.agents.openai import OpenAIAssistantAgent

from agent_k.config.general import OPENAI_ASSISTANT_CLIENT, OPENAI_ASSISTANT_MODEL
from agent_k.config.logger import logger


def addition(
    x: Annotated[float, "The first number to add"],
    y: Annotated[float, "The second number to add"],
) -> float:
    """
    Add two numbers and return the result.
    """
    return x + y


def convert_tonnage_to_mt(
    tonnage: Annotated[float, "The tonnage to convert"],
    unit: Annotated[Literal["tonne", "t", "kg", "kilogram"], "The unit of the tonnage"],
) -> float:
    """
    Convert different units of tonnage to million tonnes.
    """
    if unit in ["tonne", "t"]:
        return tonnage / 1e6
    elif unit in ["kg", "kilogram"]:
        return tonnage / 1e9
    else:
        raise ValueError(f"Unsupported unit: {unit}")


async def example():
    cancellation_token = CancellationToken()

    # Create an assistant with code interpreter
    assistant = OpenAIAssistantAgent(
        name="pdf_assistant",
        description="Helps with extracting information from PDF files",
        client=OPENAI_ASSISTANT_CLIENT,
        model=OPENAI_ASSISTANT_MODEL,
        instructions="You are a helpful PDF assistant that can extract information from PDF files. Return the information in a JSON format.\nFirst, identify the main mineral site this NI 43-101 report is about. Then, extract the tonnage data for all mineral resources and reserves with the original unit used in the report. Finally, aggregate the tonnage data to calculate the total tonnage (in million tonnes) for the site using addition and convert_tonnage_to_mt tools. Once you have the total tonnage, return the information in a JSON format with the following keys: 'status', 'site_name', 'total_tonnage', 'unit'. For example, {'status': 'TERMINATE', 'site_name': 'Site A', 'total_tonnage': 1000, 'unit': 'million tonnes'}. Do not include any other information in the JSON.",
        tools=["code_interpreter", "file_search", addition, convert_tonnage_to_mt],
    )

    text_termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [assistant],
        max_turns=5,
        termination_condition=text_termination,
    )

    logger.info("Upload files for the assistant to use")
    await assistant.on_upload_for_file_search(
        "/home/yixin0829/minmod/agent-k/data/raw/all_sources/43-101/02a000a83e76360bec8f3fce2ff46cc8099f950cc1f757f8a16592062c49b3a5c5.pdf",
        cancellation_token,
    )

    logger.info("Reset the team for a new task")
    try:
        await team.reset()  # Reset the team for a new task.
    except RuntimeError:
        pass
    logger.info("Run the team")

    await Console(team.run_stream(task="What's the total tonnage of the mineral site?"))

    # Clean up resources
    await assistant.delete_uploaded_files(cancellation_token)
    await assistant.delete_assistant(cancellation_token)


asyncio.run(example())
