# %%
import os
import re
from collections import defaultdict
from operator import add
from typing import Annotated

import litellm
import pandas as pd
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    QUESTION_TEMPLATE,
)
from agent_k.config.schemas import (
    TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION,
    TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION,
    TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION,
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from agent_k.notebooks.agentic_rag_v5 import create_markdown_retriever
from agent_k.tools.python_code_interpreter import PythonExecTool

# %% [markdown]
# # Configs

# %%
# MODEL = "gpt-4o-mini-2024-07-18"
MODEL = "o4-mini"
TEMPERATURE = 0.1

STEP_BACK_PROMPT = """Given a question that requires numerical reasoning, step back and think about the domain knowledge, math formula and relevant numerical value names required to solve the question. Generate a few examples that covers differnt corner cases without calculating the answer to help yourself understand the question better.

# Question
{question}"""

EXTRACT_PROMPT = """Given a question that requires numerical reasoning and context from documents, extract the actual numerical values and their units that are required to solve the question based on your previous reasoning of the question.

All extracted numerical values must be present in the original context. If no relevant values are found, return "No relevant values found".

# Question
{question}

# Context
{context}"""

PROGRAM_REASONER_USER_PROMPT = """You are a helper assistant that generate a python program to solve the question based on the given question and the context. You should follow the guidelines below:
1. The generated python program should be executable with correct syntax.
2. The final answer should be assigned to the variable `ans`.
3. The `ans` variable should be a float number.
4. Enclose the python code in ```python and ``` code block.

Based on the your previous intermediate reasoning, generate a python program to answer the given question now.

# Question
{question}"""

# %% [markdown]
# # Graph


# %%
class GraphState(TypedDict):
    question: str
    context: str
    messages: Annotated[list[str], add]
    answer: float


def step_back(state: GraphState):
    logger.info("--STEP BACK PROMPTING--")

    response = litellm.completion(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": STEP_BACK_PROMPT.format(question=state["question"]),
            }
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return {
        "messages": [
            {
                "role": "user",
                "content": STEP_BACK_PROMPT.format(question=state["question"]),
            },
            {"role": "assistant", "content": content},
        ]
    }


def extract(state: GraphState):
    logger.info("--EXTRACTING RELEVANT VALUES FROM THE CONTEXT--")

    response = litellm.completion(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            *state["messages"],
            {
                "role": "user",
                "content": EXTRACT_PROMPT.format(
                    question=state["question"], context=state["context"]
                ),
            },
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return {
        "messages": [
            {
                "role": "user",
                "content": EXTRACT_PROMPT.format(
                    question=state["question"], context=state["context"]
                ),
            },
            {"role": "assistant", "content": content},
        ]
    }


def program_reasoner(state: GraphState):
    logger.info("--GENERATING PYTHON PROGRAM TO ANSWER THE QUESTION--")

    user_prompt = PROGRAM_REASONER_USER_PROMPT.format(question=state["question"])
    response = litellm.completion(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            *state["messages"],
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    content = response["choices"][0]["message"]["content"]
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

    return {"answer": output}


def build_graph(viz: bool = False):
    graph_builder = StateGraph(GraphState)
    # graph_builder.add_node("step_back", step_back)
    graph_builder.add_node("extract", extract)
    graph_builder.add_node("program_reasoner", program_reasoner)
    graph_builder.add_node("execute", execute)

    # graph_builder.add_edge(START, "step_back")
    # graph_builder.add_edge("step_back", "extract")
    graph_builder.add_edge(START, "extract")
    graph_builder.add_edge("extract", "program_reasoner")
    graph_builder.add_edge("program_reasoner", "execute")
    graph_builder.add_edge("execute", END)

    if viz:
        display(
            Image(
                graph_builder.compile()
                .get_graph()
                .draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

    graph = graph_builder.compile()
    return graph


# %%
if __name__ == "__main__":
    question = QUESTION_TEMPLATE.format(
        field="total_mineral_resource_tonnage",
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )

    retriever = create_markdown_retriever(
        "paper/data/processed/43-101-refined/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md",
        collection_name="rag-chroma",
    )

    documents = retriever.invoke(question)

    graph_inputs = {
        "question": question,
        "context": documents,
    }

    # Compile graph and invoke
    graph = build_graph()
    result = graph.invoke(graph_inputs, config={"recursion_limit": 12})

    for message in result["messages"]:
        logger.debug(message)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(result["answer"])

# %% [markdown]
# # Run Experiments

# %%
complex_properties = [
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


def run_experiment(gt_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df_gt = pd.read_csv(gt_path)

    rows = []
    tokens = defaultdict(int)
    start_timestamp = config_general.get_curr_ts()
    for index, row in df_gt.iterrows():
        # if index > 0:
        #     break

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
            f"paper/data/processed/43-101-refined/{cdr_record_id}.md",
            collection_name="rag-chroma",
        )
        for (
            property_name,
            property_dtype,
            property_default,
            property_description,
        ) in complex_properties:
            question = QUESTION_TEMPLATE.format(
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
                "context": documents,
            }
            # Compile graph and invoke
            graph = build_graph()
            result = graph.invoke(graph_inputs, config={"recursion_limit": 12})

            # Parse the integer or float number from the answer using regex
            match = re.search(r"(\d+\.\d+)", result["answer"])
            if match is None:
                logger.error(f"No float number found in the answer: {result['answer']}")
                row_template[property_name] = -1
            else:
                logger.info(
                    f"Found float number in the answer: {match.group(1)}. Convert to Mt."
                )
                row_template[property_name] = float(match.group(1)) / 1e6

        rows.append(row_template)
        df_pred = pd.DataFrame(rows)

        # Save results to CSV incrementally after every report extraction
        df_pred.to_csv(
            f"{output_dir}/{start_timestamp}_ere_extraction.csv", index=False
        )
        logger.info(f"Write the CSV file to experiment results directory {output_dir}.")


if __name__ == "__main__":
    output_dir = "paper/data/experiments/tat_llm_extract_reason_exec"
    gt_path = "paper/data/processed/ground_truth/inferlink_ground_truth.csv"
    run_experiment(
        gt_path=gt_path,
        output_dir=output_dir,
    )
