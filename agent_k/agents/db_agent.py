import json
from dataclasses import dataclass
from typing import Annotated, Any, Callable

from autogen import ConversableAgent, UserProxyAgent, register_function
from loguru import logger

import agent_k.config.general as config_general
import agent_k.config.prompts as config_prompts
from agent_k.utils.db_utils import PostgresDB


@dataclass
class Tool:
    function: Callable
    desc: str


def list_tables(
    reflection: Annotated[str, "Think about why you need to list all tables."],
) -> list[str]:
    with PostgresDB() as db:
        return db.list_tables()


def list_columns(
    reflection: Annotated[
        str, "Think about why you need to list columns for this given table."
    ],
    table: Annotated[str, "The table to list columns from"],
) -> list[str]:
    with PostgresDB() as db:
        return db.list_columns(table)


def list_column_unique_values(
    reflection: Annotated[
        str, "Think about why you need to list unique values for this given column."
    ],
    column: Annotated[str, "The column to list unique values from"],
    table: Annotated[str, "The table where the column is located"],
) -> list[str]:
    with PostgresDB() as db:
        return db.list_column_unique_values(column, table)


def list_columns_with_details(
    reflection: Annotated[
        str, "Think about why you need to list columns with more details."
    ],
    table: Annotated[str, "The table to list columns with details from"],
) -> str:
    with PostgresDB() as db:
        return db.list_columns_with_details(table)


def run_query(
    reflection: Annotated[
        str, "Final check if the query is correct based on the previous tool calls."
    ],
    query: Annotated[str, "The SQL query to run"],
) -> dict[str, Any]:
    with PostgresDB() as db:
        execution_status, message, df = db.run_query(query)
        return {
            "execution_status": execution_status,
            "message": message,
            "data": df.to_json(orient="values") if df is not None else None,
        }


def check_termination(msg: dict):
    """
    Terminate when the agent get a successful sql query execution status.
    """
    if "tool_responses" not in msg:
        # If it's not a tool response, it's not the termination condition
        return False

    # This is the return value of the tool in string format
    json_str = msg["tool_responses"][0]["content"]
    if "execution_status" not in json_str:
        # If it's not a tool response from run_query, it's not the termination condition
        return False

    obj = json.loads(json_str)
    # logger.debug(f"Termination check obj['execution_status']: {obj['execution_status']}")
    return obj["execution_status"]


def construct_db_agent() -> tuple[ConversableAgent, UserProxyAgent]:
    """Construct and configure a database agent and user proxy agent.

    This function creates and configures two agents:
    1. A database agent that can execute SQL queries and gather schema information
    2. A user proxy agent that executes the tools on behalf of the DB agent

    The DB agent is configured with:
    - GPT-4 language model
    - System prompt for SQL query generation
    - Tool registration for database operations
    - Automatic termination on successful query execution

    The user proxy agent is configured with:
    - Never ask for human input
    - Automatic termination on successful query execution when the DB agent says 'TERMINATE'

    Returns:
        Tuple[ConversableAgent, UserProxyAgent]: The configured DB agent and user proxy agent
    """
    db_agent = ConversableAgent(
        "db_agent",
        llm_config={"config_list": config_general.AUTOGEN_CONFIG_LIST},
        system_message=config_prompts.DB_AGENT_SYSTEM_PROMPT_V2,
        is_termination_msg=check_termination,
        human_input_mode="NEVER",
    )

    user_proxy = UserProxyAgent(
        "user_proxy",
        llm_config=None,
        is_termination_msg=lambda msg: msg.get("content") is not None
        and "terminate" in msg["content"].lower(),
        human_input_mode="NEVER",
    )

    # Register tools in both db_agent (caller) and user_proxy (executor)
    tool_registry = {
        "list_tables": Tool(
            function=list_tables, desc="List all tables in the database."
        ),
        "list_columns": Tool(
            function=list_columns, desc="List all columns in a given table."
        ),
        "list_column_unique_values": Tool(
            function=list_column_unique_values,
            desc="List all unique values in a given column.",
        ),
        "list_columns_with_details": Tool(
            function=list_columns_with_details,
            desc="List all columns with more details in a given table. Call this tool when list_columns is called and the column definition is not clear enough to generate a good SQL query.",
        ),
        "run_query": Tool(
            function=run_query,
            desc="Run a SQL query and return the result (execution status, message, and jsonified dataframe).",
        ),
    }

    for _tool_name, tool in tool_registry.items():
        register_function(
            tool.function, caller=db_agent, executor=user_proxy, description=tool.desc
        )

    # Check if the tools are registered (JSON schema format)
    logger.debug(db_agent.llm_config["tools"])

    return db_agent, user_proxy
