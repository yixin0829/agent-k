import json
import operator
import os
from time import time
from typing import Annotated, Any, Literal, Optional

import pandas as pd
from IPython.display import Image, display
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
from litellm import completion
from openai import OpenAI
from openai.types.beta import Assistant
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from typing_extensions import TypedDict

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    DEEP_EXTRACT_USER_PROMPT,
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
from agent_k.setup.load_43_101 import list_43_101_reports
from agent_k.utils.general import get_current_timestamp

# Configs
CLIENT = OpenAI()
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
TOP_P = 0.5

# Global variables
filename_to_id_map = list_43_101_reports()


def split_json_schema(
    schema: dict, field_lists: list[list[str]], include_defs: bool = True
) -> list[dict]:
    """
    Splits a JSON schema into multiple schemas based on provided field lists,
    optionally including only the definitions referenced in each schema.

    Args:
        schema (dict): The input JSON schema.
        field_lists (list of list of str): Each inner list contains fields for a separate schema.
        include_defs (bool): If True, include referenced definitions from the original schema's $defs.

    Returns:
        list of dict: A list of JSON schemas corresponding to each field list.
    """
    # Extract properties, required fields, and definitions from the original schema
    original_properties = schema.get("properties", {})
    original_required = set(schema.get("required", []))
    original_defs = schema.get("$defs", {})

    schemas = []
    for field_list in field_lists:
        new_schema = {"type": "object", "properties": {}, "required": []}
        # Build the new schema based on the field list
        for field in field_list:
            if field in original_properties:
                new_schema["properties"][field] = original_properties[field]
            if field in original_required:
                new_schema["required"].append(field)

        # Remove 'required' key if it's empty
        if not new_schema["required"]:
            new_schema.pop("required")

        # Optionally include only the definitions referenced in the new schema
        if include_defs and original_defs:
            new_defs = {}
            for prop in new_schema["properties"].values():
                if "$ref" in prop:
                    # Expecting ref format: "#/$defs/DefinitionName"
                    ref_parts = prop["$ref"].split("/")
                    if len(ref_parts) >= 3 and ref_parts[1] == "$defs":
                        def_name = ref_parts[2]
                        if def_name in original_defs:
                            new_defs[def_name] = original_defs[def_name]
            if new_defs:
                new_schema["$defs"] = new_defs

        schemas.append(new_schema)

    return schemas


def parse_json_code_block(content: str) -> dict[str, Any]:
    """Parse the JSON code block from the assistant response."""
    try:
        json_code_block = content.split("<json>")[1].split("</json>")[0]
        return json.loads(json_code_block)
    except Exception as e:
        logger.error(f"Failed to parse JSON code block: {e}")
        logger.error(f"LLM response: {content}")
        return {}


def prompt_openai_assistant(assistant: Assistant, messages: list[dict]) -> str:
    thread = CLIENT.beta.threads.create(messages=messages)

    # Use the create and poll SDK helper to create a run and poll the status of
    # the run until it's in a terminal state.
    run = CLIENT.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    if run.status == "completed":
        messages = list(CLIENT.beta.threads.messages.list(thread_id=thread.id))
    try:
        message_content = messages[0].content[0].text
    except IndexError as e:
        logger.exception(f"{e}.`messages`: {messages}")

    annotations = message_content.annotations
    citations = []
    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(
            annotation.text, f"[{index}]"
        )
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = CLIENT.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    logger.debug(message_content.value)
    # logger.debug("\n".join(citations))

    return message_content.value


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


def viz_graph(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    pdf_path: str  # 43-101 report record ID
    json_schema: dict  # Predefined JSON schema (Assumed it's available)
    method: Literal["F&S", "DPE", "DPE MAP_REDUCE"]

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

    # # Parse the <output> XML tags
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

    logger.debug(f"Simple entities: {simple_entities}")
    logger.debug(f"Complex entities: {complex_entities}")

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
    logger.debug(f"Fast schema: {state['fast_schema']}")
    logger.debug(f"Slow schema: {state['slow_schema']}")

    next_nodes = []

    if state["fast_schema"]["properties"]:
        next_nodes.append("fast_extraction_agent")
    if state["slow_schema"]["properties"]:
        if state["method"] == "DPE":
            next_nodes.append("slow_extraction_agent_dpe")
        elif state["method"] == "DPE MAP_REDUCE":
            # Map reduce extraction of complex entities
            for entity_name, entity_schema in state["slow_schema"][
                "properties"
            ].items():
                next_nodes.append(
                    Send(
                        "slow_extraction_agent_map_reduce",
                        {
                            "pdf_path": state["pdf_path"],
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
        parsed_output = content.split("<output>")[1].split("</output>")[0].strip()
    except IndexError:
        logger.exception(
            f"Error parsing <output> XML tags for {entity_name}\nContent: {content}"
        )

    if dtype == "number" and parsed_output != "Not Found":
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


def slow_extraction_optimizer(state: State):
    slow_schema = state["slow_schema"]

    logger.info(
        "Correcting the existing slow extraction results based on the feedback and previous extraction messages"
    )
    assistant = CLIENT.beta.assistants.retrieve("asst_D2MTLBHWmh8sgYw2d90C4JyP")

    # Get the OpenAI file ID
    filename = state["pdf_path"].split("/")[-1]
    file_id = filename_to_id_map[filename]

    messages = [
        {
            "role": "user",
            "content": OPTIMIZER_USER_PROMPT.format(
                extraction_results=state["slow_extraction_agent_result"],
                feedback=state["feedback"],
                messages=state["messages"],
                json_schema=json.dumps(slow_schema),
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
    parsed_json = parse_json_code_block(content)

    return {
        "slow_extraction_agent_result": parsed_json,
        "messages": [{"role": "assistant", "content": content}],
    }


def merge_map_reduce_results(state: State):
    merged_result = {}
    for d in state["slow_extraction_agent_result_map_reduce"]:
        for k, v in d.items():
            merged_result[k] = v

    return {"slow_extraction_agent_result": merged_result}


def validate_extraction_result(state: State):
    # Validate the extraction result
    logger.info("Validating slow extraction result")
    slow_schema = state["slow_schema"]

    # logger.debug(f"state messages: {state['messages']}")

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VALIDATOR_USER_PROMPT.format(
                    extraction_results=state["slow_extraction_agent_result"],
                    messages=state["messages"],
                    json_schema=json.dumps(slow_schema),
                ),
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    logger.debug(f"Response: {response.choices[0].message.content}")
    content = response.choices[0].message.content
    parsed_feedback = content.split("<feedback>")[1].split("</feedback>")[0]
    parsed_output = content.split("<output>")[1].split("</output>")[0]

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
        "slow_extraction_agent_map_reduce", slow_extraction_agent_map_reduce
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


def extract_from_pdf(
    pdf_path: str,
    json_schema: dict,
    method: Literal["F&S", "DPE MAP_REDUCE"] = "DPE MAP_REDUCE",
    recursion_limit: int = 12,
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

    def _loguru_before_sleep_callback(retry_state):
        """Custom callback for Loguru logging before sleep in tenacity retries"""
        logger.warning(
            f"Retrying {retry_state.fn.__name__} in {retry_state.next_action.sleep} seconds "
            f"after {retry_state.attempt_number} attempt(s). Error: {retry_state.outcome.exception()}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception_type((Exception)),
        reraise=True,
        before_sleep=_loguru_before_sleep_callback,
    )
    def _extract_with_retry():
        try:
            match method:
                case "F&S":
                    graph = build_batch_extraction_graph()
                    result = graph.invoke(
                        {
                            "pdf_path": pdf_path,
                            "json_schema": json_schema,
                            "method": method,
                        }
                    )
                case "DPE MAP_REDUCE":
                    graph = build_dpe_w_map_reduce_graph()
                    result = graph.invoke(
                        {
                            "pdf_path": pdf_path,
                            "json_schema": json_schema,
                            "method": method,
                        },
                        {"recursion_limit": recursion_limit},
                    )
                case _:
                    raise ValueError(f"Unknown method: {method}")

            result = result["final_extraction_result"].model_dump(mode="json")
            return result

        except GraphRecursionError as e:
            logger.error(f"Recursion Error: {e}")
            raise
        except Exception as e:
            logger.warning(f"Extraction failed for {pdf_path}: {str(e)}. Retrying...")
            raise

    return _extract_with_retry()


def extract_from_inferlink_pdfs(
    sample_size: int = None,
    method: Literal["F&S", "DPE MAP_REDUCE"] = "DPE MAP_REDUCE",
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame
    """
    inferlink_ground_truth_filtered_path = pd.read_csv(
        "data/processed/inferlink_ground_truth_filtered.csv"
    )
    cdr_record_ids = inferlink_ground_truth_filtered_path["cdr_record_id"].tolist()
    main_commodity = inferlink_ground_truth_filtered_path["main_commodity"].tolist()

    if sample_size:
        cdr_record_ids = cdr_record_ids[:sample_size]
        main_commodity = main_commodity[:sample_size]

    logger.info(f"Extracting entities from {len(cdr_record_ids)} PDF files")

    data_rows = []
    for i, (cdr_record_id, mc) in enumerate(
        zip(cdr_record_ids, main_commodity, strict=False)
    ):
        logger.info(
            f"{i + 1}/{len(cdr_record_ids)}: Extracting entities from {cdr_record_id} with main commodity {mc}"
        )

        # replace <main_commodity> with the main commodity
        schema = MineralSiteMetadata.model_json_schema()
        schema_str = json.dumps(schema)
        schema_str = schema_str.replace("<main_commodity>", mc)
        schema = json.loads(schema_str)
        path = os.path.join(config_general.CDR_REPORTS_DIR, f"{cdr_record_id}.pdf")
        try:
            entities = extract_from_pdf(path, schema, method=method)
            entities = {**{"cdr_record_id": cdr_record_id}, **entities}
            data_rows.append(entities)
        except Exception as e:
            logger.error(f"Failed to extract from {path}: {e}")

    df = pd.DataFrame(data_rows)

    df.to_csv(
        os.path.join(
            config_general.PDF_AGENT_CACHE_DIR,
            "inferlink",
            f"{method.lower().replace(' ', '_')}_extraction_results_{get_current_timestamp()}.csv",
        ),
        index=False,
    )
    return df


def extract_from_all_pdfs(
    mineral_report_dir: str = config_general.CDR_REPORTS_DIR,
    sample_size: Optional[int] = None,
    manually_checked_pdf_paths: Optional[list[str]] = None,
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
            entities = extract_from_pdf(path, schema, method="DPE MAP_REDUCE")
            if entities:
                entities = entities.model_dump(mode="json")
                entities.update({"cdr_record_id": cdr_record_id})
                data_rows.append(entities)
        except Exception as e:
            logger.error(f"Failed to extract from {path}: {e}")

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

    sample_size = None
    df = extract_from_inferlink_pdfs(sample_size=sample_size, method="DPE MAP_REDUCE")

    logger.info("Extracting entities from all PDF files")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Total time taken: {time() - start_time:.2f} seconds")
    logger.info(f"Average time per PDF: {(time() - start_time) / len(df):.2f} seconds")
    logger.info(f"Number of PDFs: {len(df)}")
    logger.info(f"Number of entities extracted: {len(df.index)}")
