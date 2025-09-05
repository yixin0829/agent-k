# %%

import json
import os
import re
import threading
from operator import add
from typing import Annotated

import litellm
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

import src.config.prompts_finqa as prompts_finqa
from src.config.logger import logger
from src.utils.code_interpreter import PythonExecTool
from src.utils.general import extract_xml

load_dotenv()

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Model Configuration
MODEL = "gpt-3.5-turbo-0125"
# MODEL = "gpt-4-0613"
MODEL_TEMPERATURE = 0.2

# Self-Reflection Configuration
MAX_REFLECTION_ITERATIONS = 3

# Dataset and File Path Configuration
DATASET = "test"  # FinQA test set
INPUT_PATH = f"data/raw/FinQA/{DATASET}.json"
PRED_OUTPUT_DIR = "data/experiments/FinQA"
PRED_OUTPUT_PATH = os.path.join(
    PRED_OUTPUT_DIR, f"{DATASET}_pred_{MODEL.replace('/', '_')}.json"
)


# %%
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: formatted LLM final answer
        context: financial report context
        extracted_facts: facts extracted from context
        self_reflection_issues_found: whether issues were found in self-reflection
        messages: accumulated messages for React pattern
        previous_code: store previous code for self consistency
    """

    question: str
    generation: str
    context: str

    extracted_facts: str  # Facts extracted by fact extraction agent
    self_reflection_issues_found: str  # Used for self-reflection check (yes/no)
    messages: Annotated[list[str], add]  # Short-term memory for React agent
    previous_code: Annotated[list[str], add]  # Store prev code for self consistency


### Nodes


def fact_extraction_agent(state: GraphState):
    """
    Extract relevant facts from the financial context
    """
    logger.info("---FACT EXTRACTION AGENT---")

    user_prompt = prompts_finqa.FACT_EXTRACTION_AGENT_USER_PROMPT.format(
        context=state["context"],
        question=state["question"],
    )

    response = litellm.completion(
        model=MODEL,
        temperature=MODEL_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": prompts_finqa.FACT_EXTRACTION_AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response["choices"][0]["message"]["content"]
    return {
        "extracted_facts": content,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ],
    }


def program_reasoner(state: GraphState):
    """
    Generate Python code to answer the financial question using React agent pattern
    """
    logger.info("---PROGRAM REASONER---")

    # Use accumulated messages with React agent system prompt
    response = litellm.completion(
        model=MODEL,
        temperature=MODEL_TEMPERATURE,
        messages=[
            {"role": "system", "content": prompts_finqa.REACT_AGENT_SYSTEM_PROMPT},
            *state["messages"],
            {"role": "user", "content": prompts_finqa.PROGRAM_REASONER_USER_PROMPT},
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return {
        "messages": [
            {"role": "user", "content": prompts_finqa.PROGRAM_REASONER_USER_PROMPT},
            {"role": "assistant", "content": content},
        ],
        "previous_code": [content],
    }


def self_reflection(state: GraphState):
    """
    Perform self-reflection on generated code to check for issues
    """
    logger.info("---SELF REFLECTION---")

    # Financial calculation domain knowledge for validation
    calculation_knowledge = (
        "Financial calculations should follow these principles: "
        "1. Percentage changes: ((new - old) / old) * 100. "
        "2. Growth rates should be calculated consistently. "
        "3. Decreases should result in negative values. "
        "4. Unit conversions must be handled correctly (millions, thousands, etc.). "
        "5. Time periods must be considered (annual, quarterly, etc.). "
        "6. The final answer must make sense in the financial context."
    )

    # Parse the code block from the last message (program reasoner)
    msg_w_code = state["messages"][-1]["content"]

    try:
        code = msg_w_code.split("```python")[1].split("```")[0].strip()
        logger.debug(f"Code:\n{code}\n")

        # Call the self-reflection grader
        response = litellm.completion(
            model=MODEL,
            temperature=MODEL_TEMPERATURE,
            messages=[
                {"role": "system", "content": prompts_finqa.REACT_AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompts_finqa.SELF_REFLECTION_USER_PROMPT.format(
                        question=state["question"],
                        facts=state["extracted_facts"],
                        code=code,
                        calculation_knowledge=calculation_knowledge,
                    ),
                },
            ],
        )
        content = response["choices"][0]["message"]["content"]
        self_reflection_issues_found = extract_xml(content, "issues_found").strip()
        self_reflection_feedback = extract_xml(content, "feedback").strip()

    except IndexError:
        logger.error(f"No code block found in the last message: {msg_w_code}")
        self_reflection_issues_found = "yes"
        self_reflection_feedback = (
            "No ```python code block found. Please generate code in the correct format."
        )

    logger.debug(f"Self-reflection issues found: {self_reflection_issues_found}")
    logger.debug(f"Self-reflection feedback: {self_reflection_feedback}")

    return {
        "self_reflection_issues_found": self_reflection_issues_found,
        "messages": [
            {
                "role": "assistant",
                "content": f"Issues found in self-reflection: {self_reflection_issues_found}\nFeedback: {self_reflection_feedback}",
            }
        ],
    }


def self_consistency(state: GraphState):
    """
    Apply self-consistency to select the best code from multiple attempts
    """
    logger.info("---SELF CONSISTENCY CODE PICKER---")

    response = litellm.completion(
        model=MODEL,
        temperature=MODEL_TEMPERATURE,
        messages=[
            {"role": "system", "content": prompts_finqa.REACT_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts_finqa.SELF_CONSISTENCY_USER_PROMPT.format(
                    previous_code="\n\n".join(state["previous_code"])
                ),
            },
        ],
    )

    content = response["choices"][0]["message"]["content"]

    return {
        "messages": [{"role": "assistant", "content": content}],
        "previous_code": [content],
    }


def execute(state: GraphState):
    """
    Execute the Python code and capture the output

    Args:
        state: Current graph state with final code

    Returns:
        Updated state with execution results
    """
    logger.info("--EXECUTING THE PYTHON PROGRAM--")

    # Execute the last code block (either from self-reflection or self-consistency)
    code_block_msg = state["previous_code"][-1]
    output = PythonExecTool().run_code_block(code_block_msg)

    logger.info("--EXECUTION OUTPUT--")
    logger.info(output)

    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"Code interpreter execution result: {output}",
            },
        ],
        "generation": output,
    }


# --------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------


def reflection_router(state: GraphState):
    """
    Route based on self-reflection check result

    Args:
        state: Current graph state with reflection results

    Returns:
        Next node to execute based on reflection outcome
    """
    if state["self_reflection_issues_found"].lower() == "no":
        logger.info("---DECISION: CODE PASSED SELF-REFLECTION (NO ISSUES FOUND)---")
        return "execute"
    elif len(state["previous_code"]) >= MAX_REFLECTION_ITERATIONS:
        logger.info(
            "---DECISION: MAX REFLECTION ITERATIONS REACHED. USE SELF CONSISTENCY TO PICK CODE---"
        )
        return "self_consistency"
    else:
        logger.info("---DECISION: CODE ISSUES FOUND IN SELF-REFLECTION, RE-TRY---")
        return "program_reasoner"


# --------------------------------------------------------------------------------------
# Build Graph
# --------------------------------------------------------------------------------------

graph_builder = StateGraph(GraphState)

# Define the nodes with new naming consistent with agent_k.py
graph_builder.add_node("fact_extraction_agent", fact_extraction_agent)
graph_builder.add_node("program_reasoner", program_reasoner)
graph_builder.add_node("self_reflection", self_reflection)
graph_builder.add_node("self_consistency", self_consistency)
graph_builder.add_node("execute", execute)

# Define edges following the agent_k.py pattern
graph_builder.add_edge(START, "fact_extraction_agent")
graph_builder.add_edge("fact_extraction_agent", "program_reasoner")
graph_builder.add_edge("program_reasoner", "self_reflection")

# Conditional routing based on self-reflection
graph_builder.add_conditional_edges(
    "self_reflection",
    reflection_router,
    {
        "execute": "execute",
        "program_reasoner": "program_reasoner",
        "self_consistency": "self_consistency",
    },
)

graph_builder.add_edge("self_consistency", "execute")
graph_builder.add_edge("execute", END)


# %%
def write_json_async(filename, data):
    def _write():
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    t = threading.Thread(target=_write)
    t.daemon = True
    t.start()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_graph(question, context):
    """
    Invoke the graph to process a financial QA question

    Args:
        question: The financial question to answer
        context: The financial document context

    Returns:
        Graph execution results including the final answer
    """
    graph_inputs = {
        "question": question,
        "context": context,
    }

    # Compile graph and invoke
    graph = graph_builder.compile()
    value = graph.invoke(input=graph_inputs)
    return value


if __name__ == "__main__":
    os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

    # Load test data from the input path
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} {DATASET} examples")

    # Load previously generated predictions if exists
    try:
        with open(PRED_OUTPUT_PATH, "r") as f:
            pred_list = json.load(f)
            unique_ids = set(pred_dict["id"] for pred_dict in pred_list)
    except FileNotFoundError:
        pred_list = []
        unique_ids = set()

    total_examples = len(data)
    for i, d in enumerate(data):
        # Resume capability (skip already processed examples)
        if d["id"] in unique_ids:
            logger.info(f"Skipping ID: {d['id']} (already processed)")
            continue

        logger.info(f"Processing ID ({i + 1}/{total_examples}): {d['id']}")

        question = d["qa"]["question"]

        pre_text = d.get("pre_text", [])
        table = d.get("table", [])
        post_text = d.get("post_text", [])

        # Convert list of strings to a single string
        pre_text_str = " ".join(pre_text)
        table_str = "\n".join([", ".join(row) for row in table])
        post_text_str = " ".join(post_text)

        # Construct context
        context = pre_text_str + "\n\n" + table_str + "\n\n" + post_text_str

        value = invoke_graph(question, context)

        # Final generation
        logger.info("---FINAL GENERATION---")
        logger.info(value["generation"])

        # Parse the final generation to number
        generation = value["generation"]
        try:
            parsed_output = re.sub(r"[^0-9.]", "", generation)
            pred_ans = float(parsed_output)
        except ValueError:
            logger.error(
                f"Error parsing output for ID: {d['id']}. Generation: {generation}"
            )
            # If boolean answer, convert to integer
            if generation.lower() in ["yes", "no", "true", "false"]:
                pred_ans = 1 if generation.lower() in ["yes", "true"] else 0
            else:
                pred_ans = -1

        pred_dict = {
            "id": d["id"],
            "question": question,
            "gold": d["qa"].get("answer"),
            "gold_exact": d["qa"].get("exe_ans"),
            "pred": pred_ans,
        }

        pred_list.append(pred_dict)

        # Checkpoint write every 50 examples (non-blocking)
        if i % 20 == 0:
            write_json_async(PRED_OUTPUT_PATH, list(pred_list))
            logger.info(f"Checkpoint written at {i + 1}/{total_examples} examples")

    # Save the predictions to a JSON file (non-blocking)
    write_json_async(PRED_OUTPUT_PATH, pred_list)
