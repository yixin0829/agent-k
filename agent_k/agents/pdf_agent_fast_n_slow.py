import json
import operator
import os
import re
from time import time
from typing import Annotated, Any, Literal, Optional

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.pregel import RetryPolicy
from langgraph.types import Send
from litellm import completion
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    DEEP_EXTRACT_USER_PROMPT,
    OPTIMIZER_SYSTEM_PROMPT,
    OPTIMIZER_USER_PROMPT,
    PDF_AGENT_USER_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    VALIDATOR_USER_PROMPT,
)
from agent_k.config.schemas import (
    InferlinkEvalColumns,
    MineralSiteMetadata,
    MinModHyperCols,
)
from agent_k.notebooks.self_rag_v5 import (
    QUESTION_TEMPLATE,
    create_markdown_retriever,
    self_rag_graph_builder,
)
from agent_k.setup.load_43_101 import list_43_101_reports
from agent_k.utils.general import (
    get_current_timestamp,
    parse_json_code_block,
    prompt_openai_assistant,
    split_json_schema,
)

################################# Configs #################################
CLIENT = OpenAI()
SLOW_EXTRACT_VALIDATION_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.1
SLOW_EXTRACT_OPTIMIZER_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.1
RETRY_POLICY = RetryPolicy(max_attempts=3)
RECURSION_LIMIT = 12  # Self-RAG recursion limit
SELF_RAG_RETRY_LIMIT = 5
################################# Configs #################################


# Global variables
filename_to_id_map = list_43_101_reports()
retry_count = 0


def batch_extract(
    pdf_path: str,
    json_schema: dict,
) -> str:
    """
    Extract entities from a PDF file using OpenAI Assistant.
    """

    json_schema_str = json.dumps(json_schema)

    assistant = CLIENT.beta.assistants.retrieve("asst_dbMIxMYwSocPIKpZ3KLnadWB")

    filename = pdf_path.split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": PDF_AGENT_USER_PROMPT.format(
                json_schema=json_schema_str,
            ),
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [
                        {"type": "file_search"},
                        {"type": "code_interpreter"},
                    ],
                }
            ],
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


def deep_extract(pdf_path: str, field, default, description, dtype):
    """
    Extract ONE entity from a PDF file using OpenAI Assistant.
    """
    logger.info(f"Extracting {field} from {pdf_path}")
    # Use the same assistant for all deep extraction
    assistant = CLIENT.beta.assistants.retrieve("asst_50sbd2mNoNhaPecIKU34vXUP")

    # Get the OpenAI file ID
    filename = pdf_path.split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": DEEP_EXTRACT_USER_PROMPT.format(
                field=field,
                description=description,
                dtype=dtype,
                default=default,
            ),
            "attachments": [
                {
                    "file_id": file_id,
                    "tools": [
                        {"type": "file_search"},
                        {"type": "code_interpreter"},
                    ],
                }
            ],
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


class State(TypedDict):
    pdf_path: str  # 43-101 report record ID
    json_schema: dict  # Predefined JSON schema (Assumed it's available)
    method: Literal["F&S", "DPE", "DPE MAP_REDUCE", "DPE MAP_REDUCE SELF RAG"]
    retriever: Any  # Retriever for self-RAG

    # Populated by LLMs
    simple_entities: list[str]
    complex_entities: list[str]
    fast_schema: dict
    slow_schema: dict
    fast_extraction_agent_result: dict[str, Any]
    slow_extraction_agent_result_map_reduce: Annotated[list, operator.add]
    slow_extraction_agent_result: dict[str, Any]
    slow_extraction_validation: Literal["YES", "NO"]
    feedback: Annotated[list, add_messages]

    messages: Annotated[list, add_messages]
    final_extraction_result: MineralSiteMetadata


class ComplexEntityState(TypedDict):
    pdf_path: str
    entity_name: str
    description: str
    default_value: Any
    dtype: Literal["string", "number", "boolean", "array", "object"]
    retriever: Any


def schema_decompose(state: State):
    # TODO: Comment to save cost. Uncomment this once in production
    # response = completion(
    #     model=MODEL,
    #     messages=[
    #         {"role": "system", "content": SCHEMA_DECOMPOSE_SYS_PROMPT},
    #         {
    #             "role": "user",
    #             "content": DECOMPOSE_USER_PROMPT_TEMPLATE.format(
    #                 json_schema=json.dumps(state["json_schema"])
    #             ),
    #         },
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    # )

    # # Parse the <answer> XML tags
    # content = response.choices[0].message.content
    # simple_entities = []
    # complex_entities = []
    # for line in content.split("\n"):
    #     if line.startswith("1. Simple entities:"):
    #         simple_entities = line.split(":")[1].strip()
    #         simple_entities = literal_eval(simple_entities)
    #     elif line.startswith("2. Complex entities:"):
    #         complex_entities = line.split(":")[1].strip()
    #         complex_entities = literal_eval(complex_entities)
    # logger.debug(f"Response: {response.choices[0].message.content}")

    # TODO: Remove this once in production
    simple_entities = ["mineral_site_name", "country", "state_or_province"]
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
        "simple_entities": simple_entities,
        "complex_entities": complex_entities,
        "fast_schema": fast_schema,
        "slow_schema": slow_schema,
    }


def fast_and_slow_route(state: State):
    logger.info("Routing to the appropriate extraction agent")

    next_nodes = []

    if state["fast_schema"]["properties"]:
        next_nodes.append("fast_extraction_agent")

    if state["slow_schema"]["properties"]:
        if state["method"] == "DPE":
            next_nodes.append("slow_extraction_agent_dpe")
        elif state["method"] in ["DPE MAP_REDUCE", "DPE MAP_REDUCE SELF RAG"]:
            # Map reduce extraction of complex entities to parallel nodes
            for entity_name, entity_schema in state["slow_schema"][
                "properties"
            ].items():
                match state["method"]:
                    case "DPE MAP_REDUCE":
                        map_reduce_node = "slow_extraction_agent_map_reduce"
                    case "DPE MAP_REDUCE SELF RAG":
                        map_reduce_node = "slow_extraction_agent_map_reduce_self_rag"
                    case _:
                        raise ValueError(f"Unknown method: {state['method']}")
                next_nodes.append(
                    Send(
                        map_reduce_node,
                        {
                            "pdf_path": state["pdf_path"],
                            "retriever": state["retriever"],
                            "entity_name": entity_name,
                            "default_value": entity_schema.get("default", None),
                            "description": entity_schema.get("description", None),
                            "dtype": entity_schema.get("type", None),
                        },
                    )
                )
        else:
            # Default to batch extraction
            next_nodes.append("slow_extraction_agent")

    if not next_nodes:
        raise ValueError(
            "No next nodes found. Possibly because both simple and complex entities are empty."
        )

    return next_nodes


def fast_extraction_agent(state: State):
    logger.info("Batch extracting simple entities from the 43-101 report")

    fast_schema = state["fast_schema"]
    logger.debug(f"Fast schema: {fast_schema}")

    if fast_schema["properties"] == {}:
        raise ValueError(
            "No simple entities to extract. Returning empty dict as result."
        )

    content = batch_extract(state["pdf_path"], fast_schema)
    parsed_json = parse_json_code_block(content)

    return {"fast_extraction_agent_result": parsed_json}


def slow_extraction_agent(state: State):
    logger.info("Batch extracting complex entities from the 43-101 report")

    slow_schema = state["slow_schema"]
    logger.debug(f"Slow schema: {slow_schema}")

    if slow_schema["properties"] == {}:
        raise ValueError(
            "No complex entities to extract. Returning empty dict as result."
        )

    content = batch_extract(state["pdf_path"], slow_schema)
    parsed_json = parse_json_code_block(content)

    return {"slow_extraction_agent_result": parsed_json}


def slow_extraction_output_parser(
    content: str,
    entity_name: str,
    dtype: Literal["string", "number", "boolean", "array", "object"],
):
    # Corner cases
    if "not found" in content.lower():
        return "Not Found"

    try:
        parsed_output = content.split("<answer>")[1].split("</answer>")[0].strip()
    except IndexError as e:
        raise Exception(
            f"Error parsing <answer> XML tags for {entity_name}\nContent: {content}"
        ) from e

    if dtype == "number" and parsed_output != "Not Found":
        # Preprocess the output to remove any non-numeric characters except for the decimal point
        parsed_output = re.sub(r"[^0-9.]", "", parsed_output)
        parsed_output = float(parsed_output)

    return parsed_output


def slow_extraction_agent_dpe(state: State):
    """Map reduce extraction of complex entities from the 43-101 report"""
    logger.info("Deep extraction of complex entities from the 43-101 report")

    slow_schema = state["slow_schema"]
    logger.debug(f"Slow schema: {slow_schema}")

    extraction_results = {}
    messages = []
    logger.info("Extracting entities from the 43-101 report")
    for entity_name, entity_schema in slow_schema["properties"].items():
        default_value = entity_schema.get("default", None)
        description = entity_schema.get("description", None)
        dtype = entity_schema.get("type", None)

        content = deep_extract(
            state["pdf_path"], entity_name, default_value, description, dtype
        )
        parsed_output = slow_extraction_output_parser(content, entity_name, dtype)

        extraction_results[entity_name] = parsed_output
        messages.append({"role": "assistant", "content": content})

    return {
        "slow_extraction_agent_result": extraction_results,
        "messages": messages,
    }


def slow_extraction_agent_map_reduce(state: ComplexEntityState):
    pdf_path = state["pdf_path"]
    entity_name = state["entity_name"]
    default_value = state["default_value"]
    description = state["description"]
    dtype = state["dtype"]

    content = deep_extract(pdf_path, entity_name, default_value, description, dtype)
    parsed_output = slow_extraction_output_parser(content, entity_name, dtype)

    return {
        "messages": [{"role": "assistant", "content": content}],
        "slow_extraction_agent_result_map_reduce": [{entity_name: parsed_output}],
    }


def slow_extraction_agent_map_reduce_self_rag(state: ComplexEntityState):
    entity_name = state["entity_name"]
    default_value = state["default_value"]
    description = state["description"]
    dtype = state["dtype"]
    retriever = state["retriever"]

    question = QUESTION_TEMPLATE.format(
        field=entity_name,
        dtype=dtype,
        default=default_value,
        description=description,
    )
    graph_inputs = {
        "question": question,
        "generation": "N/A",
        "retriever": retriever,
        "hallucination_grade": "N/A",
    }

    def before_sleep_callback(retry_state):
        global retry_count
        retry_count += 1
        logger.warning(
            f"[{retry_state.attempt_number}/{SELF_RAG_RETRY_LIMIT}] Retrying... {retry_state.outcome.exception()}"
        )

    # Use tenacity to retry the graph invocation and output parsing with exponential backoff
    @retry(
        stop=stop_after_attempt(SELF_RAG_RETRY_LIMIT),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_callback,
    )
    def invoke_self_rag_and_parse_with_retry(graph_inputs, config, entity_name, dtype):
        # Compile a new graph for each slow entity extraction to avoid mixed memory
        self_rag_graph = self_rag_graph_builder.compile()
        value = self_rag_graph.invoke(graph_inputs, config=config)
        content = value["generation"]
        parsed_output = slow_extraction_output_parser(content, entity_name, dtype)
        return content, parsed_output

    try:
        content, parsed_output = invoke_self_rag_and_parse_with_retry(
            graph_inputs,
            config={"recursion_limit": RECURSION_LIMIT},
            entity_name=entity_name,
            dtype=dtype,
        )
    except Exception as e:
        logger.error(
            f"All retries failed for {entity_name}: {e}. Returning default value."
        )
        content = (
            f"Failed to extract {entity_name} after {SELF_RAG_RETRY_LIMIT} attempts."
        )
        parsed_output = default_value

    return {
        "messages": [{"role": "assistant", "content": content}],
        "slow_extraction_agent_result_map_reduce": [{entity_name: parsed_output}],
    }


def slow_extraction_optimizer(state: State):
    slow_schema = state["slow_schema"]

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
                json_schema=json.dumps(slow_schema),
            ),
        },
    ]

    response = completion(
        model=SLOW_EXTRACT_OPTIMIZER_MODEL,
        temperature=SLOW_EXTRACT_OPTIMIZER_TEMPERATURE,
        messages=messages,
    )
    content = response.choices[0].message.content
    parsed_json = parse_json_code_block(content)

    return {
        "slow_extraction_agent_result": parsed_json,
        "messages": [{"role": "assistant", "content": content}],
    }


def merge_map_reduce_results(state: State):
    """
    Merges the results from multiple map-reduce operations into a single dictionary.
    """
    merged_result = {}
    for d in state["slow_extraction_agent_result_map_reduce"]:
        for k, v in d.items():
            merged_result[k] = v

    return {"slow_extraction_agent_result": merged_result}


def validate_extraction_result(state: State):
    # Validate the extraction result
    logger.info("Validating slow extraction result")
    slow_schema = state["slow_schema"]

    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VALIDATOR_USER_PROMPT.format(
                extraction_results=state["slow_extraction_agent_result"],
                messages=state["messages"],
                json_schema=json.dumps(slow_schema),
            ),
        },
    ]

    response = completion(
        model=SLOW_EXTRACT_VALIDATION_MODEL,
        temperature=SLOW_EXTRACT_VALIDATION_TEMPERATURE,
        messages=messages,
    )
    logger.debug(f"Response: {response.choices[0].message.content}")
    content = response.choices[0].message.content
    parsed_feedback = content.split("<feedback>")[1].split("</feedback>")[0]
    parsed_output = content.split("<answer>")[1].split("</answer>")[0]

    return {
        "slow_extraction_validation": parsed_output,
        "feedback": [{"role": "assistant", "content": parsed_feedback}],
    }


def validate_extraction_result_route(state: State):
    logger.info("Validating extraction result route")
    next_nodes = []
    if state["slow_extraction_validation"].lower().strip() == "no":
        next_nodes.append("slow_extraction_optimizer")
    elif state["slow_extraction_validation"].lower().strip() == "yes":
        next_nodes.append("slow_extraction_finish")
    else:
        # Go back to the slow extraction agent by default
        next_nodes.append("slow_extraction_optimizer")

    return next_nodes


def slow_extraction_finish(state: State):
    return {"slow_extraction_finish": True}


def extraction_synthesis(state: State):
    # synthesize the fast and slow extraction results into a single JSON object
    logger.info("Synthesizing extraction results")
    final_extraction_result = {
        **state["fast_extraction_agent_result"],
        **state["slow_extraction_agent_result"],
    }
    return {"final_extraction_result": MineralSiteMetadata(**final_extraction_result)}


def build_batch_extraction_graph():
    """
    Fast and slow batch extraction graph.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("schema_decompose", schema_decompose)
    graph_builder.add_node("fast_extraction_agent", fast_extraction_agent)
    graph_builder.add_node("slow_extraction_agent", slow_extraction_agent)
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)
    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        fast_and_slow_route,
        {
            "fast_extraction_agent": "fast_extraction_agent",
            "slow_extraction_agent": "slow_extraction_agent",
        },
    )
    graph_builder.add_edge(
        ["fast_extraction_agent", "slow_extraction_agent"], "extraction_synthesis"
    )
    graph_builder.add_edge("extraction_synthesis", END)
    graph = graph_builder.compile()

    return graph


def build_dpe_w_map_reduce_graph():
    """
    DPE with map reduce graph.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("schema_decompose", schema_decompose)
    graph_builder.add_node("fast_extraction_agent", fast_extraction_agent)
    graph_builder.add_node(
        "slow_extraction_agent_map_reduce",
        slow_extraction_agent_map_reduce,
        retry=RETRY_POLICY,
    )
    graph_builder.add_node("slow_extraction_optimizer", slow_extraction_optimizer)
    graph_builder.add_node("validate_extraction_result", validate_extraction_result)
    graph_builder.add_node("slow_extraction_finish", slow_extraction_finish)
    graph_builder.add_node("merge_map_reduce_results", merge_map_reduce_results)
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)
    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        fast_and_slow_route,
        ["fast_extraction_agent", "slow_extraction_agent_map_reduce"],
    )
    graph_builder.add_edge(
        "slow_extraction_agent_map_reduce", "merge_map_reduce_results"
    )
    graph_builder.add_edge("merge_map_reduce_results", "validate_extraction_result")
    graph_builder.add_conditional_edges(
        "validate_extraction_result",
        validate_extraction_result_route,
        ["slow_extraction_optimizer", "slow_extraction_finish"],
    )
    graph_builder.add_edge("slow_extraction_optimizer", "validate_extraction_result")
    graph_builder.add_edge(
        ["fast_extraction_agent", "slow_extraction_finish"], "extraction_synthesis"
    )
    graph_builder.add_edge("extraction_synthesis", END)
    # Compile the graph
    graph = graph_builder.compile()

    return graph


def build_dpe_w_map_reduce_self_rag_graph():
    """
    DPE with map reduce self rag graph.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("schema_decompose", schema_decompose)
    graph_builder.add_node("fast_extraction_agent", fast_extraction_agent)
    graph_builder.add_node(
        "slow_extraction_agent_map_reduce_self_rag",
        slow_extraction_agent_map_reduce_self_rag,
        retry=RETRY_POLICY,
    )
    graph_builder.add_node("slow_extraction_optimizer", slow_extraction_optimizer)
    graph_builder.add_node("validate_extraction_result", validate_extraction_result)
    graph_builder.add_node("slow_extraction_finish", slow_extraction_finish)
    graph_builder.add_node("merge_map_reduce_results", merge_map_reduce_results)
    graph_builder.add_node("extraction_synthesis", extraction_synthesis)
    graph_builder.add_edge(START, "schema_decompose")
    graph_builder.add_conditional_edges(
        "schema_decompose",
        fast_and_slow_route,
        ["fast_extraction_agent", "slow_extraction_agent_map_reduce_self_rag"],
    )
    graph_builder.add_edge(
        "slow_extraction_agent_map_reduce_self_rag", "merge_map_reduce_results"
    )
    graph_builder.add_edge("merge_map_reduce_results", "validate_extraction_result")
    graph_builder.add_conditional_edges(
        "validate_extraction_result",
        validate_extraction_result_route,
        ["slow_extraction_optimizer", "slow_extraction_finish"],
    )
    graph_builder.add_edge("slow_extraction_optimizer", "validate_extraction_result")
    graph_builder.add_edge(
        ["fast_extraction_agent", "slow_extraction_finish"], "extraction_synthesis"
    )
    graph_builder.add_edge("extraction_synthesis", END)
    # Compile the graph
    graph = graph_builder.compile()

    return graph


def extract_from_pdf(
    pdf_path: str,
    json_schema: dict,
    method: Literal["F&S", "DPE MAP_REDUCE", "DPE MAP_REDUCE SELF RAG"],
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

    match method:
        case "F&S":
            graph = build_batch_extraction_graph()
            result = graph.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                },
                {"recursion_limit": RECURSION_LIMIT},
            )
        case "DPE MAP_REDUCE":
            graph = build_dpe_w_map_reduce_graph()
            result = graph.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                },
                {"recursion_limit": RECURSION_LIMIT},
            )
        case "DPE MAP_REDUCE SELF RAG":
            # Initialize the retriever with the markdown file path
            markdown_filename = pdf_path.split("/")[-1].replace(".pdf", ".md")
            markdown_path = os.path.join(
                "data/processed/43-101-refined", markdown_filename
            )
            retriever = create_markdown_retriever(
                markdown_path, collection_name=markdown_path
            )

            graph = build_dpe_w_map_reduce_self_rag_graph()
            result = graph.invoke(
                {
                    "pdf_path": pdf_path,
                    "json_schema": json_schema,
                    "method": method,
                    "retriever": retriever,
                },
                {"recursion_limit": RECURSION_LIMIT},
            )
        case _:
            raise ValueError(f"Unknown method: {method}")

    result = result["final_extraction_result"].model_dump(mode="json")
    return result


def extract_from_inferlink_pdfs(
    sample_size: int,
    method: Literal["F&S", "DPE MAP_REDUCE", "DPE MAP_REDUCE SELF RAG"],
    eval_type: Literal["FULL", "VAL", "TEST"],
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame.
    Results are saved incrementally to a CSV file to prevent data loss.
    """
    if eval_type == "VAL":
        inferlink_ground_truth = pd.read_csv(
            "data/processed/ground_truth/inferlink_ground_truth_val.csv"
        )
    elif eval_type == "TEST":
        inferlink_ground_truth = pd.read_csv(
            "data/processed/ground_truth/inferlink_ground_truth_test.csv"
        )
    elif eval_type == "FULL":
        inferlink_ground_truth = pd.read_csv(
            "data/processed/ground_truth/inferlink_ground_truth.csv"
        )

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
    output_dir = os.path.join(config_general.PDF_AGENT_CACHE_DIR, "inferlink")
    os.makedirs(output_dir, exist_ok=True)

    # Create output evaluation file path
    output_file = os.path.join(
        output_dir,
        f"{method.lower().replace(' ', '_')}_extraction_results_{get_current_timestamp()}.csv",
    )

    # Create empty DataFrame with headers
    schema = MineralSiteMetadata.model_json_schema()
    empty_df = pd.DataFrame(
        columns=[
            InferlinkEvalColumns.ID.value,
            InferlinkEvalColumns.CDR_RECORD_ID.value,
            *list(schema["properties"].keys()),
        ]
    )
    empty_df.to_csv(output_file, index=False)

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
                df_row.to_csv(output_file, mode="a", header=False, index=False)

        except Exception as e:
            logger.error(f"Failed to extract from {cdr_record_id}: {e}")
            continue

    # Read the final CSV file and return as DataFrame
    final_df = pd.read_csv(output_file)

    return final_df


def extract_from_all_pdfs(
    mineral_report_dir: str = config_general.CDR_REPORTS_DIR,
    sample_size: Optional[int] = None,
    manually_checked_pdf_paths: Optional[list[str]] = None,
    method: Literal[
        "F&S", "DPE MAP_REDUCE", "DPE MAP_REDUCE SELF RAG"
    ] = "DPE MAP_REDUCE",
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame
    """
    # Load PDF paths used for evaluation
    pdf_paths = []
    for _i, pdf_path in enumerate(os.listdir(mineral_report_dir)):
        pdf_paths.append(os.path.join(mineral_report_dir, pdf_path))

    if sample_size:
        pdf_paths = pdf_paths[:sample_size]

    if manually_checked_pdf_paths:
        pdf_paths = [path for path in pdf_paths if path in manually_checked_pdf_paths]

    # Load ground truth data
    ground_truth_path = os.path.join(
        config_general.GROUND_TRUTH_DIR,
        "minmod_hyper_response_enriched_nickel_subset_43_101_gt.csv",
    )
    df_hyper_43_101_subset = pd.read_csv(ground_truth_path)
    hyper_record_ids = df_hyper_43_101_subset[MinModHyperCols.RECORD_VALUE.value].values
    logger.info(
        f"Hyper dataframe (subset 43-101) filtered to {len(df_hyper_43_101_subset)} rows"
    )

    data_rows = []
    schema = MineralSiteMetadata.model_json_schema()
    for i, path in enumerate(pdf_paths):
        # Skip if the PDF file is not in the subset of ground truth
        cdr_record_id = path.split("/")[-1].split(".")[0]
        if cdr_record_id not in hyper_record_ids:
            logger.warning(
                f"{i + 1}/{len(pdf_paths)}: Skipping {path} because it is not in the ground truth"
            )
            continue

        logger.info(f"{i + 1}/{len(pdf_paths)}: Extracting entities from {path}")
        try:
            entities = extract_from_pdf(path, schema, method=method)
        except Exception as e:
            logger.error(f"Failed to extract from {path}: {e}")

        try:
            if entities:
                entities = entities.model_dump(mode="json")
                entities.update({"cdr_record_id": cdr_record_id})
                data_rows.append(entities)
        except Exception as e:
            logger.error(f"Failed to process entities for {path}: {e}")

    # Filter out empty results
    data_rows = [row for row in data_rows if row]

    df = pd.DataFrame(data_rows)

    if not os.path.exists(config_general.PDF_AGENT_CACHE_DIR):
        logger.info(f"Creating directory {config_general.PDF_AGENT_CACHE_DIR}")
        os.makedirs(config_general.PDF_AGENT_CACHE_DIR)

    logger.info(f"Saving extraction results to {config_general.PDF_AGENT_CACHE_DIR}")
    df.to_csv(
        os.path.join(
            config_general.PDF_AGENT_CACHE_DIR,
            f"pdf_agent_extraction_{get_current_timestamp()}.csv",
        ),
        index=False,
    )

    return df


if __name__ == "__main__":
    # Log metadata about the extraction (total time, number of PDFs, number of entities extracted)
    start_time = time()

    ################################# Configs #################################
    sample_size = None
    method = "DPE MAP_REDUCE SELF RAG"
    eval_type = "TEST"
    ################################# Configs #################################

    df = extract_from_inferlink_pdfs(
        sample_size=sample_size,
        method=method,
        eval_type=eval_type,
    )

    logger.info("Extracting entities from all PDF files")
    # Experiment hyperparameters
    logger.info(f"Code agent retry limit: {SELF_RAG_RETRY_LIMIT}")
    logger.info(f"Sample size: {'100%' if sample_size is None else sample_size}%")
    logger.info(f"Method: {method}")
    logger.info(f"Evaluation type: {eval_type}")

    # Metadata
    logger.info(f"Code agent retry count: {retry_count}")
    logger.info(f"Average code agent retry count per PDF: {retry_count / len(df):.2f}")
    logger.info(f"Total time taken: {time() - start_time:.2f} seconds")
    logger.info(f"Average time per PDF: {(time() - start_time) / len(df):.2f} seconds")
    logger.info(f"Number of PDFs: {len(df)}")
    logger.info(f"Number of entities extracted: {len(df.index)}")
