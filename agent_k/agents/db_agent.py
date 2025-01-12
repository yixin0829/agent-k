import asyncio
from typing import Annotated, Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat, Swarm
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

import agent_k.config.general as config_general
import agent_k.config.prompts as config_prompts
from agent_k.utils.db_utils import DuckDBWrapper


async def list_tables(
    reflection: Annotated[str, "Think about why you need to list all tables."],
) -> list[str]:
    """
    List all tables in the database.
    """
    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        return f"Tables in the database: {db.list_tables()}"


async def list_columns(
    reflection: Annotated[
        str, "Think about why you need to list columns for this given table."
    ],
    table: Annotated[str, "The table to list columns from"],
) -> list[str]:
    """
    List all columns in a given table.
    """
    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        columns = db.list_columns(table)
        return f"Columns in the table {table}: {columns}"


async def list_column_unique_values(
    reflection: Annotated[
        str, "Think about why you need to list unique values for this given column."
    ],
    column: Annotated[str, "The column to list unique values from"],
    table: Annotated[str, "The table where the column is located"],
) -> list[str]:
    """
    List all unique values in a given column.
    """
    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        unique_values = db.list_column_unique_values(column, table)
        return f"Unique values in the column {column} of the table {table}: {unique_values}"


async def list_columns_with_details(
    reflection: Annotated[
        str, "Think about why you need to list columns with more details."
    ],
    table: Annotated[str, "The table to list columns with details from"],
) -> str:
    """
    Get the detailed description of columns in a given table.
    """
    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        details = db.list_columns_with_details(table)
        return f"Details of the table {table}:\n{details}"


async def run_query(
    reflection: Annotated[
        str, "Final check if the query is correct based on the previous tool calls."
    ],
    query: Annotated[str, "The SQL query to run"],
) -> dict[str, Any]:
    """
    Run a SQL query and return the result in a dictionary format (execution_status, message, and jsonified dataframe).
    """
    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        execution_status, message, df = db.run_query(query)
        return {
            "execution_status": execution_status,
            "message": message,
            "data": df.to_json(orient="values") if df is not None else None,
        }


def construct_db_agent() -> AssistantAgent:
    """
    Construct and configure a database agent.
    """
    db_agent = AssistantAgent(
        name="db_agent",
        description="A database agent that can gather schema information using tools, generate SQL queries, and run SQL queries.",
        model_client=config_general.OPENAI_MODEL_CLIENT,
        system_message=config_prompts.DB_AGENT_SYSTEM_PROMPT_V2,
        tools=[
            list_tables,
            list_columns,
            list_column_unique_values,
            list_columns_with_details,
            run_query,
        ],
    )

    return db_agent


def construct_db_agent_team():
    db_agent = construct_db_agent()
    text_termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat(
        [db_agent], max_turns=10, termination_condition=text_termination
    )
    return team


def construct_swarm_team():
    sql_agent = AssistantAgent(
        name="sql_agent",
        description="A SQL agent that can gather database schema information and generate SQL queries.",
        model_client=config_general.OPENAI_MODEL_CLIENT,
        handoffs=["critic_agent"],
        system_message=config_prompts.SQL_AGENT_SYSTEM_PROMPT,
        tools=[
            list_tables,
            list_columns,
            list_column_unique_values,
            list_columns_with_details,
        ],
    )
    critic_agent = AssistantAgent(
        name="critic_agent",
        description="A critic agent that can evaluate the SQL query result.",
        model_client=config_general.OPENAI_MODEL_CLIENT,
        handoffs=["sql_agent"],
        system_message=config_prompts.CRITIC_AGENT_SYSTEM_PROMPT,
        tools=[
            run_query,
        ],
    )
    text_termination = TextMentionTermination("APPROVE")
    team = Swarm(
        [sql_agent, critic_agent],
        max_turns=10,
        termination_condition=text_termination,
    )

    return team


async def demo_run_single_response() -> None:
    """
    This calls the agent with one user message to get one response only
    """
    db_agent: AssistantAgent = construct_db_agent()

    response = await db_agent.on_messages(
        [
            TextMessage(
                content="What are all the mineral sites located in Sofala Province, Vietnam? Report record value and total grade.",
                source="user",
            )
        ],
        cancellation_token=CancellationToken(),
    )
    print(response.inner_messages)
    print("=" * 100)
    print(response.chat_message)


async def demo_run_one_agent_team(observe_method: str = "console") -> None:
    """
    This runs the agent with a team of one agent to talk to itself
    """
    db_agent: AssistantAgent = construct_db_agent()
    text_termination = TextMentionTermination("TERMINATE")
    agent_team = RoundRobinGroupChat(
        [db_agent], max_turns=10, termination_condition=text_termination
    )
    task = "What are all the mineral sites located in Tasmania, Australia? Report mineral site name and state or province."

    try:
        await agent_team.reset()  # Reset the team for a new task.
    except RuntimeError:
        pass

    if observe_method == "console":
        await Console(agent_team.run_stream(task=task))
    elif observe_method == "async":
        async for message in agent_team.run_stream(task=task):
            if isinstance(message, TaskResult):
                print("Stop Reason:", message.stop_reason)
            else:
                print(message)


if __name__ == "__main__":
    asyncio.run(demo_run_one_agent_team())
