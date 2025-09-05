# %%
import json
import operator
import os
import re
from typing import Annotated, Any, Literal

import litellm
import pandas as pd
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

import src.config.experiment_config as config_experiment
from src.config.logger import logger
from src.config.prompts import (
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_PROMPT,
    QUESTION_TEMPLATE,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)
from src.config.schemas import (
    MineralEvalDfColumns,
    MineralSiteMetadata,
)
from src.experiments.agent_k import graph_builder as agent_k_graph_builder
from src.experiments.self_rag import graph_builder as self_rag_graph_builder
from src.experiments.tat_llm import build_graph as build_tat_llm_graph
from src.utils.general import (
    extract_xml,
    get_curr_ts,
    parse_json_code_block,
)
from src.utils.llm import (
    create_markdown_retriever,
    invoke_model_messages,
)

# --------------------------------------------------------------------------------------
# Configuration Variables
# --------------------------------------------------------------------------------------

# Global client and settings
CLIENT = OpenAI()
litellm.drop_params = True  # Ignore temperature parameter if model doesn't support it
retry_count = 0

# File paths configuration
PROCESSED_REPORTS_DIR = "data/processed/43-101_reports_refined"
RAW_REPORTS_DIR = "data/raw/43-101"
GT_FULL_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth.csv"
GT_TEST_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth_test.csv"
GT_DEV_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth_dev.csv"

# Output directories configuration
TAT_LLM_OUTPUT_DIR = "data/experiments/tat_llm"
AGENT_K_OUTPUT_DIR = "data/experiments/agent_k"
SELF_RAG_OUTPUT_DIR = "data/experiments/self_rag"

# Numerical columns for unit conversion to million tonnes
NUMERICAL_COLUMNS = [
    MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
    MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
    MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
    MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
]

# Ground truth file paths mapping
GT_PATH_MAP = {
    "TEST": GT_TEST_PATH,
    "DEV": GT_DEV_PATH,
    "FULL": GT_FULL_PATH,
}

# Output directory mapping
OUTPUT_DIR_MAP = {
    "TAT-LLM": TAT_LLM_OUTPUT_DIR,
    "AGENT-K": AGENT_K_OUTPUT_DIR,
    "SELF-RAG": SELF_RAG_OUTPUT_DIR,
}


class State(TypedDict):
    markdown_path: str  # Refined NI 43-101 report path
    json_schema: dict  # User-defined JSON schema
    method: Literal["SELF-RAG", "AGENT-K", "TAT-LLM"]
    retriever: Any

    # Populated by LLMs
    extraction_results: Annotated[list, operator.add]  # Map-reduce results
    extraction_json: dict[str, Any]  # Final extraction results JSON

    global_validation: Literal["YES", "NO"]  # Latest Global validation result
    feedback: Annotated[list, add_messages]  # Feedback from global validation agent
    messages: Annotated[list, add_messages]  # Messages as short-term memory


class ComplexPropertyState(TypedDict):
    method: Literal["SELF-RAG", "AGENT-K", "TAT-LLM"]
    markdown_path: str
    property_name: str
    description: str
    default_value: Any
    dtype: Literal["string", "number", "boolean", "array", "object"]
    retriever: Any


def direct_extract_route(state: State):
    """Route directly to complex property extraction in parallel."""
    logger.info("Routing directly to complex property extraction")
    next_nodes = []

    # Extract all complex properties in parallel using MapReduce
    for property_name, property_schema in state["json_schema"]["properties"].items():
        if state["method"] in {"AGENT-K", "SELF-RAG", "TAT-LLM"}:
            map_reduce_node = "map_extraction_agent"
        next_nodes.append(
            Send(
                map_reduce_node,
                {
                    "method": state["method"],
                    "markdown_path": state["markdown_path"],
                    "property_name": property_name,
                    "default_value": property_schema.get("default"),
                    "description": property_schema.get("description", None),
                    "dtype": property_schema.get("type", None),
                    "retriever": state["retriever"],
                },
            )
        )

    return next_nodes


def extraction_output_parser(
    content: str,
    property_name: str,
    dtype: Literal["string", "number", "boolean", "array", "object"],
) -> Any:
    # Corner cases
    if "not found" in content.lower():
        return "Not Found"

    # For TAT-LLM, the content is a direct numerical value as string from execution
    # For SELF-RAG and AGENT-K, extract from XML tags
    # Try to extract from XML tags first (for SELF-RAG and AGENT-K)
    parsed_output = extract_xml(content, "answer")

    # If no XML tags found (empty string returned), assume it's TAT-LLM's direct output
    if not parsed_output:
        parsed_output = content.strip()

    if dtype == "number" and parsed_output != "Not Found":
        # For direct numerical strings (TAT-LLM), try to parse as-is first
        try:
            # First try to parse the string directly as a float
            parsed_output = float(parsed_output)
        except (ValueError, TypeError):
            # If that fails, remove non-numeric characters and try again
            cleaned = re.sub(r"[^0-9.-]", "", str(parsed_output))
            try:
                parsed_output = float(cleaned)
            except ValueError:
                logger.warning(
                    f"Failed to parse as number: {parsed_output} for {property_name}"
                )
                return -1

    return parsed_output


def map_extraction_agent(state: ComplexPropertyState):
    question = QUESTION_TEMPLATE.format(
        field=state["property_name"],
        dtype=state["dtype"],
        default=state["default_value"],
        description=state["description"],
    )

    def before_sleep_callback(retry_state):
        global retry_count
        retry_count += 1
        logger.warning(
            f"[{retry_state.attempt_number}/{config_experiment.EXTRACTION_MAX_RETRIES}] Retrying... {retry_state.outcome.exception()}"
        )

    # Use tenacity to retry the graph invocation and output parsing with exponential backoff
    @retry(
        stop=stop_after_attempt(config_experiment.EXTRACTION_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_callback,
    )
    def invoke_extraction_with_retry(question, retriever, property_name, dtype):
        # Compile a new graph for each slow property extraction to avoid mixed memory
        match state["method"]:
            case "SELF-RAG":
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
            case "AGENT-K":
                rag_graph = agent_k_graph_builder.compile()
                graph_inputs = {
                    "markdown_path": state["markdown_path"],
                    "property_name": property_name,
                    "question": question,
                    "retriever": retriever,
                }
            case "TAT-LLM":
                rag_graph = build_tat_llm_graph()
                # Get documents from retriever
                documents = retriever.invoke(question)
                graph_inputs = {
                    "question": question,
                    "documents": documents,
                }
        value = rag_graph.invoke(graph_inputs)
        content = str(value["generation"])  # Ensure content is always a string
        parsed_output = extraction_output_parser(content, property_name, dtype)
        return content, parsed_output

    try:
        content, parsed_output = invoke_extraction_with_retry(
            question,
            state["retriever"],
            property_name=state["property_name"],
            dtype=state["dtype"],
        )
    except Exception as e:
        logger.error(
            f"All retries failed for {state['property_name']}: {e}. Returning -1."
        )
        content = f"Failed to extract {state['property_name']} after {config_experiment.EXTRACTION_MAX_RETRIES} attempts. Returning -1."
        parsed_output = -1

    return {
        "messages": [{"role": "assistant", "content": content}],
        "extraction_results": [{state["property_name"]: parsed_output}],
    }


def global_validation_agent_optimizer(state: State):
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
                extraction_results=state["extraction_json"],
                feedback=state["feedback"],
                messages=state["messages"],
                json_schema=json.dumps(state["json_schema"]),
            ),
        },
    ]

    content = invoke_model_messages(
        model_name=config_experiment.SLOW_EXTRACT_OPTIMIZER_MODEL,
        messages=messages,
        temperature=config_experiment.SLOW_EXTRACT_OPTIMIZER_TEMPERATURE,
    )
    parsed_json = parse_json_code_block(content)

    return {
        "extraction_json": parsed_json,
        "messages": [{"role": "assistant", "content": content}],
    }


def reduce_extraction_results(state: State):
    """
    Reduces the results from multiple map-reduce operations into a single dictionary.
    """
    merged_result = {}
    for d in state["extraction_results"]:
        for k, v in d.items():
            merged_result[k] = v

    return {"extraction_json": merged_result}


def global_validation_agent(state: State):
    """
    Globally `Validate the slow extracted entities
    """
    logger.info("Validating slow extraction result")

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VALIDATOR_USER_PROMPT.format(
                extraction_json=state["extraction_json"],
                messages=state["messages"],
                json_schema=json.dumps(state["json_schema"]),
            ),
        },
    ]

    content = invoke_model_messages(
        model_name=config_experiment.SLOW_EXTRACT_VALIDATION_MODEL,
        messages=messages,
        temperature=config_experiment.SLOW_EXTRACT_VALIDATION_TEMPERATURE,
    )
    parsed_feedback = extract_xml(content, "feedback")
    parsed_output = extract_xml(content, "answer")

    return {
        "global_validation": parsed_output,
        "feedback": [{"role": "assistant", "content": parsed_feedback}],
    }


def validate_extraction_result_route(state: State):
    logger.info("Validating extraction result route")

    if state["global_validation"].lower().strip() == "yes":
        return END
    else:
        return "global_validation_agent_optimizer"


def build_dpe_w_map_reduce_self_rag_graph():
    """
    DPE with map reduce self rag graph.
    """
    self_rag_graph_builder = StateGraph(State)
    self_rag_graph_builder.add_node(
        "map_extraction_agent",
        map_extraction_agent,
    )
    self_rag_graph_builder.add_node(
        "reduce_extraction_results", reduce_extraction_results
    )
    self_rag_graph_builder.add_conditional_edges(
        START,
        direct_extract_route,
        ["map_extraction_agent"],
    )
    self_rag_graph_builder.add_edge("map_extraction_agent", "reduce_extraction_results")
    self_rag_graph_builder.add_edge(
        "reduce_extraction_results",
        END,
    )

    graph = self_rag_graph_builder.compile()

    return graph


def build_dpe_w_map_reduce_agent_k_graph():
    """
    DPE with map reduce agent-k graph.
    """
    agent_k_graph_builder = StateGraph(State)
    agent_k_graph_builder.add_node(
        "map_extraction_agent",
        map_extraction_agent,
    )
    agent_k_graph_builder.add_node(
        "global_validation_agent_optimizer", global_validation_agent_optimizer
    )
    agent_k_graph_builder.add_node(
        "validate_extraction_result", global_validation_agent
    )
    agent_k_graph_builder.add_node(
        "reduce_extraction_results", reduce_extraction_results
    )
    agent_k_graph_builder.add_conditional_edges(
        START,
        direct_extract_route,
        ["map_extraction_agent"],
    )
    agent_k_graph_builder.add_edge("map_extraction_agent", "reduce_extraction_results")
    agent_k_graph_builder.add_edge(
        "reduce_extraction_results", "validate_extraction_result"
    )
    agent_k_graph_builder.add_conditional_edges(
        "validate_extraction_result",
        validate_extraction_result_route,
        ["global_validation_agent_optimizer", END],
    )
    agent_k_graph_builder.add_edge(
        "global_validation_agent_optimizer", "validate_extraction_result"
    )
    graph = agent_k_graph_builder.compile()

    return graph


def build_dpe_w_map_reduce_tat_llm_graph():
    """
    DPE with map reduce TAT-LLM graph.
    TAT-LLM uses a simpler flow without validation/optimization loop.
    """
    tat_llm_graph_builder = StateGraph(State)
    tat_llm_graph_builder.add_node(
        "map_extraction_agent",
        map_extraction_agent,
    )
    tat_llm_graph_builder.add_node(
        "reduce_extraction_results", reduce_extraction_results
    )

    # Define edges
    tat_llm_graph_builder.add_conditional_edges(
        START,
        direct_extract_route,
        ["map_extraction_agent"],
    )
    tat_llm_graph_builder.add_edge("map_extraction_agent", "reduce_extraction_results")
    tat_llm_graph_builder.add_edge(
        "reduce_extraction_results",
        END,
    )

    graph = tat_llm_graph_builder.compile()
    return graph


# %%
def extract_from_pdf(
    pdf_path: str,
    json_schema: dict,
    method: Literal["SELF-RAG", "AGENT-K", "TAT-LLM"],
) -> dict:
    """
    Extract information from a PDF file using different extraction methods.

    Args:
        pdf_path (str): Path to the PDF file
        json_schema (dict): JSON schema defining the extraction fields
        method (str): Extraction method to use. One of "SELF-RAG", "AGENT-K", or "TAT-LLM"

    Returns:
        dict: Extracted information as a dictionary from MineralSiteMetadata Pydantic model
    """

    # Initialize the retriever with the markdown file path
    markdown_filename = pdf_path.split("/")[-1].replace(".pdf", ".md")

    markdown_path = os.path.join(PROCESSED_REPORTS_DIR, markdown_filename)
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
        case "SELF-RAG":
            graph = build_dpe_w_map_reduce_self_rag_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
            )
        case "AGENT-K":
            graph = build_dpe_w_map_reduce_agent_k_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
            )
        case "TAT-LLM":
            graph = build_dpe_w_map_reduce_tat_llm_graph()
            result = graph.invoke(
                {
                    "markdown_path": markdown_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
            )
        case _:
            raise ValueError(f"Unknown method: {method}")

    result = result["extraction_json"]
    return result


def extract_from_pdfs(
    sample_size: int | None,
    method: Literal["SELF-RAG", "AGENT-K", "TAT-LLM"],
    eval_type: Literal["FULL", "TEST", "DEV"],
    output_dir: str,
    output_filename: str,
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame.
    Results are saved incrementally to a CSV file to prevent data loss.
    """
    mineral_ground_truth = pd.read_csv(GT_PATH_MAP[eval_type])

    # Log experiment hyperparameters
    logger.info(f"Method: {method}")
    logger.info(f"Eval type: {eval_type}")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output filename: {output_filename}")

    ids = mineral_ground_truth[MineralEvalDfColumns.ID.value].tolist()
    cdr_record_ids = mineral_ground_truth[
        MineralEvalDfColumns.CDR_RECORD_ID.value
    ].tolist()
    main_commodities = mineral_ground_truth[
        MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value
    ].tolist()

    # For testing purpose. If None, extract from all PDF files
    if sample_size is not None:
        ids = ids[:sample_size]
        cdr_record_ids = cdr_record_ids[:sample_size]
        main_commodities = main_commodities[:sample_size]

    logger.info(f"Extracting entities from {len(cdr_record_ids)} PDF files")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, output_filename)

    # Create empty DataFrame with IDs and extracted columns
    empty_df = pd.DataFrame(
        columns=pd.Series(
            [
                MineralEvalDfColumns.ID.value,
                MineralEvalDfColumns.CDR_RECORD_ID.value,
                MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
                MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
                MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
                MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
            ]
        )
    )
    empty_df.to_csv(output_file_path, index=False)

    for i, (id, cdr_record_id, main_commodity) in enumerate(
        zip(ids, cdr_record_ids, main_commodities, strict=False)
    ):
        logger.info(
            f"{i + 1}/{len(ids)}: Extracting entities from {cdr_record_id} with main commodity {main_commodity}"
        )

        try:
            # Replace <main_commodity> with the actual main commodity in the structured data schema
            schema = MineralSiteMetadata.model_json_schema()
            schema_str = json.dumps(schema)
            schema_str = schema_str.replace("<main_commodity>", main_commodity)
            current_schema = json.loads(schema_str)

            # Extract entities from the PDF file
            entities = extract_from_pdf(
                os.path.join(RAW_REPORTS_DIR, f"{cdr_record_id}.pdf"),
                current_schema,
                method=method,
            )

            if entities:
                # Add record ID to entities
                entities = {**{"id": id, "cdr_record_id": cdr_record_id}, **entities}

                # Convert to DataFrame and convert unit of numerical columns to million tonnes
                df_row = pd.DataFrame([entities])
                for col in NUMERICAL_COLUMNS:
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
    # ----------------------------------------------------------------------------------
    # Configs
    # ----------------------------------------------------------------------------------
    sample_size = config_experiment.PDF_EXTRACTION_SAMPLE_SIZE
    method = config_experiment.PDF_EXTRACTION_METHOD.value
    eval_type = config_experiment.PDF_EXTRACTION_EVAL_TYPE

    output_dir = OUTPUT_DIR_MAP.get(method)
    if output_dir is None:
        raise ValueError(f"Unknown method: {method}")
    os.makedirs(output_dir, exist_ok=True)

    final_df = extract_from_pdfs(
        sample_size=sample_size,
        method=method,
        eval_type=eval_type,
        output_dir=output_dir,
        output_filename=f"{method.lower().replace(' ', '_')}_{get_curr_ts()}.csv",
    )
