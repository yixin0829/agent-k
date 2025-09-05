# %%
import logging
import os
import re
import time
from collections import defaultdict
from operator import add
from typing import Annotated, Any

import pandas as pd
import yaml
from langgraph.graph import END, START, StateGraph
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict

import src.config.experiment_config as config_experiment
import src.config.prompts as prompts
from src.config.logger import logger
from src.config.schemas import (
    TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION,
    TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION,
    TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION,
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from src.utils.code_interpreter import PythonExecTool
from src.utils.general import get_curr_ts
from src.utils.llm import create_markdown_retriever, invoke_model_messages

# --------------------------------------------------------------------------------------
# Configuration Variables
# --------------------------------------------------------------------------------------

# Model configuration
MODEL = config_experiment.TAT_LLM_MODEL
TEMPERATURE = config_experiment.TAT_LLM_TEMPERATURE
MAX_RETRIES = config_experiment.EXTRACTION_MAX_RETRIES

# File paths configuration
PROCESSED_REPORTS_DIR = "data/processed/43-101_reports_refined"
GT_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth.csv"
OUTPUT_DIR = "data/experiments/tat_llm"
COLLECTION_NAME = "rag-chroma"

# Demo configuration
DEMO_SAMPLE_SIZE = 3

# Complex properties configuration
COMPLEX_PROPERTIES = [
    (
        "total_mineral_resource_tonnage",
        "float",
        0,
        TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    ),
    (
        "total_mineral_reserve_tonnage",
        "float",
        0,
        TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION,
    ),
    (
        "total_mineral_resource_contained_metal",
        "float",
        0,
        TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION,
    ),
    (
        "total_mineral_reserve_contained_metal",
        "float",
        0,
        TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION,
    ),
]

# CSV output columns
OUTPUT_COLUMNS = [
    "id",
    "cdr_record_id",
    "commodity_observed_name",
    "total_mineral_resource_tonnage",
    "total_mineral_reserve_tonnage",
    "total_mineral_resource_contained_metal",
    "total_mineral_reserve_contained_metal",
]


# %%
class GraphState(TypedDict):
    question: str
    documents: list[str]  # Changed from context: str to align with SELF-RAG and AGENT-K
    messages: Annotated[list[str], add]
    generation: float


def extract(state: GraphState):
    logger.info("--EXTRACTING RELEVANT VALUES FROM THE CONTEXT--")

    # Join documents list into a single context string
    context = ""
    for document in state["documents"]:
        context += str(document) + "\n\n"

    content = invoke_model_messages(
        model_name=MODEL,
        messages=[
            *state["messages"],
            {
                "role": "user",
                "content": prompts.EXTRACT_PROMPT.format(
                    question=state["question"], context=context
                ),
            },
        ],
        temperature=TEMPERATURE,
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": prompts.EXTRACT_PROMPT.format(
                    question=state["question"], context=context
                ),
            },
            {"role": "assistant", "content": content},
        ]
    }


def program_reasoner(state: GraphState):
    logger.info("--GENERATING PYTHON PROGRAM TO ANSWER THE QUESTION--")

    user_prompt = prompts.PROGRAM_REASONER_USER_PROMPT.format(
        question=state["question"]
    )
    content = invoke_model_messages(
        model_name=MODEL,
        messages=[
            *state["messages"],
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=TEMPERATURE,
    )
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ]
    }


def execute(state: GraphState):
    logger.info("--EXECUTING THE PYTHON PROGRAM--")

    msg_w_code = state["messages"][-1]["content"]
    output = PythonExecTool().run_code_block(msg_w_code)

    logger.info("--EXECUTION OUTPUT--")
    logger.info(output)

    return {"generation": output}


def build_graph():
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("extract", extract)
    graph_builder.add_node("program_reasoner", program_reasoner)
    graph_builder.add_node("execute", execute)

    graph_builder.add_edge(START, "extract")
    graph_builder.add_edge("extract", "program_reasoner")
    graph_builder.add_edge("program_reasoner", "execute")
    graph_builder.add_edge("execute", END)

    graph = graph_builder.compile()
    return graph


# %%
def invoke_graph_with_retries(
    graph_inputs: dict[str, Any], max_retries: int = 5
) -> dict[str, Any]:
    """Invoke the compiled graph with retry logic using tenacity.

    Args:
        graph_inputs: Input dictionary passed to the graph.
        max_retries: Maximum number of attempts before giving up.

    Returns:
        The graph result dictionary on success.

    Raises:
        Exception: If invocation fails for all retry attempts.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential_jitter(initial=0.5, max=4.0),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _invoke_graph(graph_inputs: dict[str, Any]) -> dict[str, Any]:
        """Invoke the graph a single time; retries handled by decorator."""
        graph = build_graph()
        return graph.invoke(graph_inputs)

    try:
        return _invoke_graph(graph_inputs)
    except RetryError as err:  # Exhausted retries
        raise Exception("Graph invocation failed after retries") from err


# %% [markdown]
# # Run Experiments


def run_experiment(
    gt_path: str,
    output_dir: str,
    sample_size: int | None = None,
    max_retries: int = MAX_RETRIES,
) -> None:
    """Run the extraction experiment over a ground-truth CSV.

    For each row and each complex property, retrieve context and invoke the
    reasoning graph. The graph invocation is retried based on max_retries per property
    if exceptions occur; on persistent failure, a sentinel value of -1 is
    recorded and processing continues with the next property.

    Args:
        gt_path: Path to the ground-truth CSV file.
        output_dir: Directory where incremental CSV outputs are written.
        sample_size: Optional limit on number of rows to process for testing.
        max_retries: Maximum number of retry attempts for failed graph invocations.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_gt = pd.read_csv(gt_path)

    # For testing purpose. If None, extract from all PDF files
    if sample_size is not None:
        df_gt = df_gt.head(sample_size)

    # Log experiment hyperparameters
    logger.info("TAT-LLM extraction experiment")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Total rows to process: {len(df_gt)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max retries per property: {max_retries}")

    tokens = defaultdict(int)
    start_time = time.time()
    start_timestamp = get_curr_ts()

    # Create empty DataFrame with headers and save to initialize CSV file
    empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    output_file_path = os.path.join(output_dir, f"{start_timestamp}_ere_extraction.csv")
    empty_df.to_csv(output_file_path, index=False)
    for index, row in df_gt.iterrows():
        logger.info(f"Processing row {index + 1} of {len(df_gt)}")
        id = row["id"]
        cdr_record_id = row["cdr_record_id"]
        commodity_observed_name = row["commodity_observed_name"]
        row_template = {
            "id": id,
            "cdr_record_id": cdr_record_id,
            "commodity_observed_name": commodity_observed_name,
            "total_mineral_resource_tonnage": -1,
            "total_mineral_reserve_tonnage": -1,
            "total_mineral_resource_contained_metal": -1,
            "total_mineral_reserve_contained_metal": -1,
        }

        retriever = create_markdown_retriever(
            f"{PROCESSED_REPORTS_DIR}/{cdr_record_id}.md",
            collection_name=COLLECTION_NAME,
        )
        for (
            property_name,
            property_dtype,
            property_default,
            property_description,
        ) in COMPLEX_PROPERTIES:
            question = prompts.QUESTION_TEMPLATE.format(
                field=property_name,
                dtype=property_dtype,
                default=property_default,
                description=property_description.replace(
                    "<main_commodity>", commodity_observed_name
                ),
            )
            documents = retriever.invoke(question)
            graph_inputs = {
                "question": question,
                "documents": documents,  # Changed from context to documents for consistency
            }
            # Compile graph and invoke with retries
            try:
                result = invoke_graph_with_retries(
                    graph_inputs, max_retries=max_retries
                )
            except Exception as err:
                logger.exception(
                    f"Failed to invoke graph for property '{property_name}' after retries: {err}"
                )
                row_template[property_name] = -1
                continue

            # Parse the integer or float number from the answer using regex. Make decimal point optional.
            match = re.search(r"(\d+(\.\d*)?)", result["generation"])
            if match is None:
                logger.error(
                    f"No float number found in the answer: {result['generation']}"
                )
                row_template[property_name] = -1
            else:
                logger.info(
                    f"Found float number in the answer: {match.group(1)}. Convert to Mt."
                )
                row_template[property_name] = float(match.group(1)) / 1e6

        # Convert to DataFrame and append to CSV file immediately
        df_row = pd.DataFrame([row_template])
        df_row.to_csv(output_file_path, mode="a", header=False, index=False)

        logger.info(f"Appended row {index + 1} to CSV file in {output_dir}.")

    end_time = time.time()

    # Read the final CSV file to get the actual results and log completion
    df_final = pd.read_csv(output_file_path)
    logger.info(
        f"TAT-LLM extraction experiment completed. Successfully processed {len(df_final)} reports. Results saved to {output_file_path}"
    )

    # Log experiment results
    logger.info("TAT-LLM extraction experiment")
    logger.info(f"Total time taken: {end_time - start_time} seconds")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Temperature: {TEMPERATURE}")

    # Create experiment metadata
    experiment_metadata = {
        "timestamp": start_timestamp,
        "experiment_type": "tat_llm",
        "model": MODEL,
        "temperature": TEMPERATURE,
        "sample_size": sample_size,
        "max_retries": max_retries,
        "total_time_seconds": end_time - start_time,
        "num_rows_processed": len(df_final),
        "total_rows": len(df_gt),
        "complex_properties": [prop[0] for prop in COMPLEX_PROPERTIES],
        "output_file": output_file_path,
    }

    # Save experiment metadata
    metadata_file_path = os.path.join(
        output_dir, f"{start_timestamp}_experiment_metadata.yaml"
    )
    with open(metadata_file_path, "w") as f:
        yaml.dump(experiment_metadata, f)

    logger.info(f"Experiment metadata saved to {metadata_file_path}")


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # Configs
    # ----------------------------------------------------------------------------------
    # None = full dataset. Otherwise, use first K samples
    sample_size = DEMO_SAMPLE_SIZE
    output_dir = OUTPUT_DIR
    gt_path = GT_PATH

    run_experiment(
        gt_path=gt_path,
        output_dir=output_dir,
        sample_size=sample_size,
    )
