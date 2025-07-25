# %%
import json
import operator
import os
import re
from operator import add
from time import time
from typing import Annotated, Any, Literal

import litellm
import pandas as pd
import yaml
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

import agent_k.config.experiment_config as config_experiment
import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_PROMPT,
    QUESTION_TEMPLATE,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)
from agent_k.config.schemas import (
    InferlinkEvalColumns,
    MineralSiteMetadata,
)
from agent_k.notebooks.agentic_rag_v5 import create_markdown_retriever
from agent_k.setup.load_43_101 import list_43_101_reports
from agent_k.utils.general import (
    extract_xml,
    get_curr_ts,
    parse_json_code_block,
    split_json_schema,
)
from paper.experiments.agentic_rag_v6 import graph_builder_v6
from paper.experiments.agentic_rag_v7 import graph_builder_v7
from paper.experiments.self_rag_v2 import self_rag_graph_builder

# Global variables
CLIENT = OpenAI()
litellm.drop_params = True  # Ignore temperature parameter if model doesn't support it
filename_to_id_map = list_43_101_reports()
retry_count = 0


class State(TypedDict):
    markdown_path: str  # 43-101 report record ID
    json_schema: dict  # Predefined JSON schema (Assumed it's available)
    method: Literal["F&S SELF RAG", "F&S AGENTIC RAG"]
    retriever: Any  # Retriever for self-RAG

    # Populated by LLMs
    fast_schema: dict
    slow_schema: dict
    fast_extraction_agent_result: dict[str, Any]
    slow_extraction_agent_result_map: Annotated[list, operator.add]
    slow_extraction_agent_result: dict[str, Any]
    slow_extraction_validation: Literal["YES", "NO"]
    feedback: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    final_extraction_result: MineralSiteMetadata


class ComplexEntityState(TypedDict):
    method: Literal["F&S SELF RAG", "F&S AGENTIC RAG"]
    markdown_path: str
    entity_name: str
    description: str
    default_value: Any
    dtype: Literal["string", "number", "boolean", "array", "object"]
    retriever: Any


def schema_decompose(state: State):
    simple_entities = []
    complex_entities = [
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    ]

    # logger.debug(f"Simple entities: {simple_entities}")
    # logger.debug(f"Complex entities: {complex_entities}")

    fast_schema, slow_schema = split_json_schema(
        state["json_schema"], [simple_entities, complex_entities]
    )

    return {
        "fast_schema": fast_schema,
        "slow_schema": slow_schema,
    }


def fast_and_slow_route(state: State):
    logger.info("Routing to the appropriate extraction agent")
    next_nodes = []

    if state["fast_schema"]["properties"]:
        next_nodes.append("fast_extraction_agent")

    if state["slow_schema"]["properties"]:
        # Map reduce extraction of complex entities to parallel nodes
        for entity_name, entity_schema in state["slow_schema"]["properties"].items():
            match state["method"]:
                case "F&S AGENTIC RAG":
                    map_reduce_node = "map_slow_extraction_agent"
                case "F&S SELF RAG":
                    map_reduce_node = "map_slow_extraction_agent"
                case _:
                    raise ValueError(f"Unknown method: {state['method']}")
            next_nodes.append(
                Send(
                    map_reduce_node,
                    {
                        "method": state["method"],
                        "markdown_path": state["markdown_path"],
                        "entity_name": entity_name,
                        "default_value": entity_schema.get("default"),
                        "description": entity_schema.get("description", None),
                        "dtype": entity_schema.get("type", None),
                        "retriever": state["retriever"],
                    },
                )
            )

    return next_nodes


def slow_extraction_output_parser(
    content: str,
    entity_name: str,
    dtype: Literal["string", "number", "boolean", "array", "object"],
) -> Any:
    # Corner cases
    if "not found" in content.lower():
        return "Not Found"

    try:
        parsed_output = extract_xml(content, "answer")
    except IndexError as e:
        raise Exception(
            f"Error parsing <answer> XML tags for {entity_name}\nContent: {content}"
        ) from e

    if dtype == "number" and parsed_output != "Not Found":
        # Preprocess the output to remove any non-numeric characters except for the decimal point
        parsed_output = re.sub(r"[^0-9.]", "", parsed_output)
        parsed_output = float(parsed_output)

    return parsed_output


def map_slow_extraction_agent(state: ComplexEntityState):
    question = QUESTION_TEMPLATE.format(
        field=state["entity_name"],
        dtype=state["dtype"],
        default=state["default_value"],
        description=state["description"],
    )

    def before_sleep_callback(retry_state):
        global retry_count
        retry_count += 1
        logger.warning(
            f"[{retry_state.attempt_number}/{config_experiment.SLOW_WORKFLOW_RETRY_LIMIT}] Retrying... {retry_state.outcome.exception()}"
        )

    # Use tenacity to retry the graph invocation and output parsing with exponential backoff
    @retry(
        stop=stop_after_attempt(config_experiment.SLOW_WORKFLOW_RETRY_LIMIT),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_callback,
    )
    def invoke_rag_and_parse_with_retry(
        question, retriever, config, entity_name, dtype
    ):
        # Compile a new graph for each slow entity extraction to avoid mixed memory
        match state["method"]:
            case "F&S SELF RAG":
                rag_graph = self_rag_graph_builder.compile()
                graph_inputs = {
                    "question": question,
                    "generation": "N/A",
                    "retriever": retriever,
                    "hallucination_grade": "N/A",
                    "hallucination_grader_reasoning": "N/A",
                    "answer_grade": "N/A",
                    "answer_grader_reasoning": "N/A",
                }
            case "F&S AGENTIC RAG":
                rag_graph = graph_builder_v6.compile()
                graph_inputs = {
                    "markdown_path": state["markdown_path"],
                    "question": question,
                    "retriever": retriever,
                }
            case _:
                raise ValueError(f'Unknown state["method"]: {state["method"]}')
        value = rag_graph.invoke(graph_inputs, config=config)
        content = value["generation"]
        parsed_output = slow_extraction_output_parser(content, entity_name, dtype)
        return content, parsed_output

    try:
        content, parsed_output = invoke_rag_and_parse_with_retry(
            question,
            state["retriever"],
            config={"recursion_limit": config_experiment.RECURSION_LIMIT},
            entity_name=state["entity_name"],
            dtype=state["dtype"],
        )
    except Exception as e:
        logger.error(
            f"All retries failed for {state['entity_name']}: {e}. Returning -1."
        )
        content = f"Failed to extract {state['entity_name']} after {config_experiment.SLOW_WORKFLOW_RETRY_LIMIT} attempts. Returning -1."
        parsed_output = -1

    return {
        "messages": [{"role": "assistant", "content": content}],
        "slow_extraction_agent_result_map": [{state["entity_name"]: parsed_output}],
    }


def slow_extraction_optimizer(state: State):
    logger.info(
        "Correcting the existing slow extraction results based on the feedback and previous extraction messages"
    )

    messages = [
        {
            "role": "system",
            "content": OPTIMIZER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": OPTIMIZER_USER_PROMPT.format(
                extraction_results=state["slow_extraction_agent_result"],
                feedback=state["feedback"],
                messages=state["messages"],
                json_schema=json.dumps(state["json_schema"]),
            ),
        },
    ]

    response = litellm.completion(
        model=config_experiment.SLOW_EXTRACT_OPTIMIZER_MODEL,
        temperature=config_experiment.SLOW_EXTRACT_OPTIMIZER_TEMPERATURE,
        messages=messages,
    )
    content = response.choices[0].message.content
    parsed_json = parse_json_code_block(content)

    return {
        "slow_extraction_agent_result": parsed_json,
        "messages": [{"role": "assistant", "content": content}],
    }


def reduce_slow_extraction_results(state: State):
    """
    Reduces the results from multiple map-reduce operations into a single dictionary.
    """
    merged_result = {}
    for d in state["slow_extraction_agent_result_map"]:
        for k, v in d.items():
            merged_result[k] = v

    return {"slow_extraction_agent_result": merged_result}


def validate_extraction_result(state: State):
    """
    Globally `Validate the slow extracted entities
    """
    logger.info("Validating slow extraction result")

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VALIDATOR_USER_PROMPT.format(
                extraction_results=state["slow_extraction_agent_result"],
                messages=state["messages"],
                json_schema=json.dumps(state["json_schema"]),
            ),
        },
    ]

    response = litellm.completion(
        model=config_experiment.SLOW_EXTRACT_VALIDATION_MODEL,
        temperature=config_experiment.SLOW_EXTRACT_VALIDATION_TEMPERATURE,
        messages=messages,
    )
    content = response.choices[0].message.content
    parsed_feedback = extract_xml(content, "feedback")
    parsed_output = extract_xml(content, "answer")

    return {
        "slow_extraction_validation": parsed_output,
        "feedback": [{"role": "assistant", "content": parsed_feedback}],
    }


def validate_extraction_result_route(state: State):
    logger.info("Validating extraction result route")

    if state["slow_extraction_validation"].lower().strip() == "yes":
        return "extraction_synthesis"
    else:
        return "slow_extraction_optimizer"


def slow_extraction_end(state: State):
    return {"slow_extraction_end": True}


def extraction_synthesis(state: State):
    # synthesize the fast and slow extraction results into a single JSON object
    logger.info("Synthesizing extraction results")

    final_extraction_result = {**state["slow_extraction_agent_result"]}

    return {"final_extraction_result": final_extraction_result}


def build_dpe_w_map_reduce_self_rag_graph():
    """
    DPE with map reduce self rag graph.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("schema_decompose", schema_decompose)
    graph_builder.add_node(
        "map_slow_extraction_agent",
        map_slow_extraction_agent,
    )
    graph_builder.add_node(
        "reduce_slow_extraction_results", reduce_slow_extraction_results
    )
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)
    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        fast_and_slow_route,
        ["map_slow_extraction_agent"],
    )
    graph_builder.add_edge(
        "map_slow_extraction_agent", "reduce_slow_extraction_results"
    )
    graph_builder.add_edge(
        "reduce_slow_extraction_results",
        "extraction_synthesis",
    )
    graph_builder.add_edge("extraction_synthesis", END)

    graph = graph_builder.compile()

    return graph


display(
    Image(
        build_dpe_w_map_reduce_self_rag_graph()
        .get_graph()
        .draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


def build_dpe_w_map_reduce_agentic_rag_graph():
    """
    DPE with map reduce agentic rag graph.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("schema_decompose", schema_decompose)
    graph_builder.add_node(
        "map_slow_extraction_agent",
        map_slow_extraction_agent,
    )
    graph_builder.add_node("slow_extraction_optimizer", slow_extraction_optimizer)
    graph_builder.add_node("validate_extraction_result", validate_extraction_result)
    graph_builder.add_node(
        "reduce_slow_extraction_results", reduce_slow_extraction_results
    )
    # graph_builder.add_node("slow_extraction_end", slow_extraction_end)
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)
    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        fast_and_slow_route,
        ["map_slow_extraction_agent"],
    )
    graph_builder.add_edge(
        "map_slow_extraction_agent", "reduce_slow_extraction_results"
    )
    # ----------------------------------------------------------------------------------
    # Global slow validator
    # ----------------------------------------------------------------------------------
    graph_builder.add_edge(
        "reduce_slow_extraction_results", "validate_extraction_result"
    )
    graph_builder.add_conditional_edges(
        "validate_extraction_result",
        validate_extraction_result_route,
        ["slow_extraction_optimizer", "extraction_synthesis"],
    )
    graph_builder.add_edge("slow_extraction_optimizer", "validate_extraction_result")
    # graph_builder.add_edge("slow_extraction_end", "extraction_synthesis")
    # ----------------------------------------------------------------------------------
    # W/O global slow validator
    # ----------------------------------------------------------------------------------
    # graph_builder.add_edge(
    #     "reduce_slow_extraction_results", "slow_extraction_end"
    # )
    # graph_builder.add_edge(
    #     ["fast_extraction_agent", "slow_extraction_end"],
    #     "extraction_synthesis",
    # )

    graph_builder.add_edge("extraction_synthesis", END)
    graph = graph_builder.compile()

    return graph


display(
    Image(
        build_dpe_w_map_reduce_agentic_rag_graph()
        .get_graph()
        .draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


# %%
class StateSequential(TypedDict):
    markdown_path: str  # 43-101 report record ID
    json_schema: dict  # Predefined JSON schema (Assumed it's available)
    method: Literal["F&S SELF RAG", "F&S AGENTIC RAG"]
    retriever: Any  # Retriever for self-RAG

    # Populated by LLMs
    previous_answers: Annotated[list[dict], add]  # Used for global hallucination check
    slow_extraction_agent_result: dict[str, Any]
    slow_extraction_validation: Literal["YES", "NO"]
    feedback: Annotated[list, add_messages]
    messages: Annotated[list, add_messages]
    final_extraction_result: MineralSiteMetadata


def extract_mineral_resource_tonnage(state: StateSequential):
    question = QUESTION_TEMPLATE.format(
        field=InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        dtype="float",
        default=0,
        description=state["json_schema"]["properties"][
            InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value
        ]["description"],
    )

    graph = graph_builder_v7.compile()
    value = graph.invoke(
        {
            "question": question,
            "retriever": state["retriever"],
            "previous_answers": state["previous_answers"],
        },
    )

    parsed_output = slow_extraction_output_parser(
        value["generation"],
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        "number",
    )

    return {
        "messages": [{"role": "assistant", "content": value["generation"]}],
        "previous_answers": [
            {"role": "assistant", "question": question, "answer": value["generation"]}
        ],
        "slow_extraction_agent_result_map": [
            {InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value: parsed_output}
        ],
    }


def extract_mineral_reserve_tonnage(state: StateSequential):
    question = QUESTION_TEMPLATE.format(
        field=InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        dtype="float",
        default=0,
        description=state["json_schema"]["properties"][
            InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value
        ]["description"],
    )

    graph = graph_builder_v7.compile()
    value = graph.invoke(
        {
            "question": question,
            "retriever": state["retriever"],
            "previous_answers": state["previous_answers"],
        },
    )

    parsed_output = slow_extraction_output_parser(
        value["generation"],
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        "number",
    )

    return {
        "messages": [{"role": "assistant", "content": value["generation"]}],
        "previous_answers": [
            {"role": "assistant", "question": question, "answer": value["generation"]}
        ],
        "slow_extraction_agent_result_map": [
            {InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value: parsed_output}
        ],
    }


def extract_mineral_resource_contained_metal(state: StateSequential):
    question = QUESTION_TEMPLATE.format(
        field=InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        dtype="float",
        default=0,
        description=state["json_schema"]["properties"][
            InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value
        ]["description"],
    )

    graph = graph_builder_v7.compile()
    value = graph.invoke(
        {
            "question": question,
            "retriever": state["retriever"],
            "previous_answers": state["previous_answers"],
        },
    )

    parsed_output = slow_extraction_output_parser(
        value["generation"],
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        "number",
    )

    return {
        "messages": [{"role": "assistant", "content": value["generation"]}],
        "previous_answers": [
            {"role": "assistant", "question": question, "answer": value["generation"]}
        ],
        "slow_extraction_agent_result_map": [
            {
                InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value: parsed_output
            }
        ],
    }


def extract_mineral_reserve_contained_metal(state: StateSequential):
    question = QUESTION_TEMPLATE.format(
        field=InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
        dtype="float",
        default=0,
        description=state["json_schema"]["properties"][
            InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value
        ]["description"],
    )

    graph = graph_builder_v7.compile()
    value = graph.invoke(
        {
            "question": question,
            "retriever": state["retriever"],
            "previous_answers": state["previous_answers"],
        },
    )

    parsed_output = slow_extraction_output_parser(
        value["generation"],
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
        "number",
    )

    return {
        "messages": [{"role": "assistant", "content": value["generation"]}],
        "previous_answers": [
            {"role": "assistant", "question": question, "answer": value["generation"]}
        ],
        "slow_extraction_agent_result_map": [
            {
                InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value: parsed_output
            }
        ],
    }


def build_agentic_rag_v7_graph():
    graph_builder = StateGraph(StateSequential)
    graph_builder.add_node(
        "extract_mineral_resource_tonnage", extract_mineral_resource_tonnage
    )
    graph_builder.add_node(
        "extract_mineral_reserve_tonnage", extract_mineral_reserve_tonnage
    )
    graph_builder.add_node(
        "extract_mineral_resource_contained_metal",
        extract_mineral_resource_contained_metal,
    )
    graph_builder.add_node(
        "extract_mineral_reserve_contained_metal",
        extract_mineral_reserve_contained_metal,
    )
    graph_builder.add_node(
        "reduce_slow_extraction_results", reduce_slow_extraction_results
    )
    graph_builder.add_node("validate_extraction_result", validate_extraction_result)
    graph_builder.add_node("slow_extraction_optimizer", slow_extraction_optimizer)
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)

    graph_builder.add_edge(START, "extract_mineral_resource_tonnage")
    graph_builder.add_edge(
        "extract_mineral_resource_tonnage", "extract_mineral_reserve_tonnage"
    )
    graph_builder.add_edge(
        "extract_mineral_reserve_tonnage", "extract_mineral_resource_contained_metal"
    )
    graph_builder.add_edge(
        "extract_mineral_resource_contained_metal",
        "extract_mineral_reserve_contained_metal",
    )
    graph_builder.add_edge(
        "extract_mineral_reserve_contained_metal", "reduce_slow_extraction_results"
    )
    graph_builder.add_edge(
        "reduce_slow_extraction_results", "validate_extraction_result"
    )
    graph_builder.add_conditional_edges(
        "validate_extraction_result",
        validate_extraction_result_route,
        ["slow_extraction_optimizer", "extraction_synthesis"],
    )
    graph_builder.add_edge("slow_extraction_optimizer", "validate_extraction_result")
    graph_builder.add_edge("extraction_synthesis", END)

    graph = graph_builder.compile()

    return graph


display(
    Image(
        build_agentic_rag_v7_graph()
        .get_graph()
        .draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)


# %%
def extract_from_pdf(
    pdf_path: str,
    json_schema: dict,
    method: Literal["F&S SELF RAG", "F&S AGENTIC RAG", "F&S AGENTIC RAG V7"],
) -> dict:
    """
    Extract information from a PDF file using different extraction methods.

    Args:
        pdf_path (str): Path to the PDF file
        json_schema (dict): JSON schema defining the extraction fields
        method (str): Extraction method to use. One of "F&S", "DPE", or "DPE MAP_REDUCE"
        recursion_limit (int): Maximum number of recursions for validation loop

    Returns:
        dict: Extracted information as a dictionary from MineralSiteMetadata Pydantic model
    """

    # Initialize the retriever with the markdown file path
    markdown_filename = pdf_path.split("/")[-1].replace(".pdf", ".md")

    markdown_path = os.path.join(
        "paper/data/processed/43-101-refined", markdown_filename
    )
    if config_experiment.RETRIEVAL_METHOD == config_experiment.RetrievalMethod.RAG:
        retriever = create_markdown_retriever(
            markdown_path, collection_name=markdown_path
        )
    elif (
        config_experiment.RETRIEVAL_METHOD
        == config_experiment.RetrievalMethod.LONG_CONTEXT
    ):
        retriever = None  # Will use gpt-4.1-mini as a retriever in the sub-graph

    match method:
        case "F&S SELF RAG":
            graph = build_dpe_w_map_reduce_self_rag_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
                {"recursion_limit": config_experiment.RECURSION_LIMIT},
            )
        case "F&S AGENTIC RAG":
            graph = build_dpe_w_map_reduce_agentic_rag_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
                {"recursion_limit": config_experiment.RECURSION_LIMIT},
            )
        case "F&S AGENTIC RAG V7":
            graph = build_agentic_rag_v7_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
                {"recursion_limit": config_experiment.RECURSION_LIMIT},
            )
        case _:
            raise ValueError(f"Unknown method: {method}")

    result = result["final_extraction_result"]
    return result


def extract_from_inferlink_pdfs(
    sample_size: int | None,
    method: Literal["F&S", "DPE MAP_REDUCE", "F&S SELF RAG", "F&S AGENTIC RAG"],
    eval_type: Literal["FULL", "VAL", "TEST"],
    output_dir: str,
    output_filename: str,
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame.
    Results are saved incrementally to a CSV file to prevent data loss.
    """
    if eval_type == "VAL":
        inferlink_ground_truth = pd.read_csv(
            "paper/data/processed/ground_truth/inferlink_ground_truth_val.csv"
        )
    elif eval_type == "TEST":
        inferlink_ground_truth = pd.read_csv(
            "paper/data/processed/ground_truth/inferlink_ground_truth_test.csv"
        )
    elif eval_type == "FULL":
        inferlink_ground_truth = pd.read_csv(
            "paper/data/processed/ground_truth/inferlink_ground_truth.csv"
        )

    # Log experiment hyperparameters
    logger.info(f"Method: {method}")
    logger.info(f"Eval type: {eval_type}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output filename: {output_filename}")

    ids = inferlink_ground_truth[InferlinkEvalColumns.ID.value].tolist()
    cdr_record_ids = inferlink_ground_truth[
        InferlinkEvalColumns.CDR_RECORD_ID.value
    ].tolist()
    main_commodities = inferlink_ground_truth[
        InferlinkEvalColumns.COMMODITY_OBSERVED_NAME.value
    ].tolist()

    # For testing purpose. If None, extract from all PDF files
    if sample_size:
        ids = ids[:sample_size]
        cdr_record_ids = cdr_record_ids[:sample_size]
        main_commodities = main_commodities[:sample_size]

    logger.info(f"Extracting entities from {len(cdr_record_ids)} PDF files")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_filename)

    # Create empty DataFrame with IDs and extracted columns
    schema = MineralSiteMetadata.model_json_schema()
    empty_df = pd.DataFrame(
        columns=pd.Series(
            [
                InferlinkEvalColumns.ID.value,
                InferlinkEvalColumns.CDR_RECORD_ID.value,
                InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
                InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
                InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
                InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
            ]
        )
    )
    empty_df.to_csv(output_file_path, index=False)

    # Define numerical columns for conversion
    numerical_columns = [
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    ]

    for i, (id, cdr_record_id, main_commodity) in enumerate(
        zip(ids, cdr_record_ids, main_commodities, strict=False)
    ):
        logger.info(
            f"{i + 1}/{len(ids)}: Extracting entities from {cdr_record_id} with main commodity {main_commodity}"
        )

        try:
            # Replace <main_commodity> with the actual main commodity in the structured data schema
            schema_str = json.dumps(schema)
            schema_str = schema_str.replace("<main_commodity>", main_commodity)
            current_schema = json.loads(schema_str)

            # Extract entities from the PDF file
            entities = extract_from_pdf(
                os.path.join(config_general.CDR_REPORTS_DIR, f"{cdr_record_id}.pdf"),
                current_schema,
                method=method,
            )

            if entities:
                # Add record ID to entities
                entities = {**{"id": id, "cdr_record_id": cdr_record_id}, **entities}

                # Convert to DataFrame and convert unit of numerical columns to million tonnes
                df_row = pd.DataFrame([entities])
                for col in numerical_columns:
                    if col in df_row.columns:
                        df_row[col] = pd.to_numeric(df_row[col], errors="coerce")
                        df_row[col] = df_row[col] / 1e6

                # Append to CSV file
                df_row.to_csv(output_file_path, mode="a", header=False, index=False)

        except Exception as e:
            logger.exception(f"Failed to extract from {cdr_record_id}: {e}")
            continue

    # Read the final CSV file
    final_df = pd.read_csv(output_file_path)
    return final_df


if __name__ == "__main__":
    # Log metadata about the extraction (total time, number of PDFs, number of entities extracted)
    start_time = time()

    # ----------------------------------------------------------------------------------
    # Configs
    # ----------------------------------------------------------------------------------
    sample_size = config_experiment.PDF_EXTRACTION_SAMPLE_SIZE
    method = config_experiment.PDF_EXTRACTION_METHOD.value
    eval_type = config_experiment.PDF_EXTRACTION_EVAL_TYPE

    output_dir = "paper/data/experiments"
    final_df = extract_from_inferlink_pdfs(
        sample_size=sample_size,
        method=method,
        eval_type=eval_type,
        output_dir=output_dir,
        output_filename=f"{method.lower().replace(' ', '_')}_{get_curr_ts()}.csv",
    )

    # Write experiment configs and metadata to a yaml file
    output_file_path = os.path.join(
        output_dir,
        f"{method.lower().replace(' ', '_')}_{get_curr_ts()}.csv",
    )
    config_file = output_file_path.replace(".csv", "_config.yaml")

    # Create a dictionary to hold all YAML data
    yaml_data = {
        "experiment_configurations": {
            "code_agent_react_configs": {
                "python_agent_model": config_experiment.PYTHON_AGENT_MODEL,
                "python_agent_temperature": config_experiment.PYTHON_AGENT_TEMPERATURE,
            },
            "agentic_rag_configs": {
                "num_retrieved_docs": config_experiment.NUM_RETRIEVED_DOCS,
                "grade_retrieval_model": config_experiment.GRADE_RETRIEVAL_MODEL,
                "grade_retrieval_temperature": config_experiment.GRADE_RETRIEVAL_TEMPERATURE,
                "grade_hallucination_model": config_experiment.GRADE_HALLUCINATION_MODEL,
                "grade_hallucination_temperature": config_experiment.GRADE_HALLUCINATION_TEMPERATURE,
                "question_rewriter_model": config_experiment.QUESTION_REWRITER_MODEL,
                "question_rewriter_temperature": config_experiment.QUESTION_REWRITER_TEMPERATURE,
                "react_code_agent_recursion_limit": config_experiment.REACT_CODE_AGENT_RECURSION_LIMIT,
            },
            "self_rag_configs": {
                "grade_retrieval_model": config_experiment.SELF_RAG_GRADE_RETRIEVAL_MODEL,
                "grade_retrieval_temperature": config_experiment.SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE,
                "generation_model": config_experiment.SELF_RAG_GENERATION_MODEL,
                "generation_temperature": config_experiment.SELF_RAG_GENERATION_TEMPERATURE,
                "grade_hallucination_model": config_experiment.SELF_RAG_GRADE_HALLUCINATION_MODEL,
                "grade_hallucination_temperature": config_experiment.SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE,
                "grade_answer_model": config_experiment.SELF_RAG_GRADE_ANSWER_MODEL,
                "grade_answer_temperature": config_experiment.SELF_RAG_GRADE_ANSWER_TEMPERATURE,
                "question_rewriter_model": config_experiment.SELF_RAG_QUESTION_REWRITER_MODEL,
                "question_rewriter_temperature": config_experiment.SELF_RAG_QUESTION_REWRITER_TEMPERATURE,
            },
            "pdf_agent_configs": {
                "slow_extract_validation_model": config_experiment.SLOW_EXTRACT_VALIDATION_MODEL,
                "slow_extract_validation_temperature": config_experiment.SLOW_EXTRACT_VALIDATION_TEMPERATURE,
                "slow_extract_optimizer_model": config_experiment.SLOW_EXTRACT_OPTIMIZER_MODEL,
                "slow_extract_optimizer_temperature": config_experiment.SLOW_EXTRACT_OPTIMIZER_TEMPERATURE,
                "recursion_limit": config_experiment.RECURSION_LIMIT,
                "self_rag_retry_limit": config_experiment.SLOW_WORKFLOW_RETRY_LIMIT,
            },
            "experiment_parameters": {
                "method": method,
                "evaluation_type": eval_type,
                "sample_size": "100%" if sample_size is None else sample_size,
            },
            "experiment_metadata": {
                "code_agent_retry_count": retry_count,
                "average_code_agent_retry_count_per_pdf": round(
                    retry_count / len(final_df), 2
                ),
                "number_of_pdfs_processed": len(final_df),
                "number_of_entities_extracted": len(final_df.index),
                "total_time_taken": time() - start_time,
                "average_time_per_pdf": round((time() - start_time) / len(final_df), 2),
                "output_file_path": output_file_path,
                "config_file_path": config_file,
            },
        }
    }

    # Write the dictionary to YAML file
    with open(config_file, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
