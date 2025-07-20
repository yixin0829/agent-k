# %%

import json
import re
import threading
from operator import add
from typing import Annotated

import litellm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

import agent_k.config.experiment_config as config_experiment
from agent_k.config.logger import logger
from agent_k.tools.python_code_interpreter import PythonExecTool

# %%
### Hallucination Grader

GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination grader validating whether there is hallucination in a LLM's generated code. The generated code is for answering a financial question based on hybrid tabular and text context. Focus on the calculation logic and unit conversions.

Guidelines:
1. The final answer should be assigned to a variable called `ans` in the code.
2. The calculation logic should be correct and grounded in the retrieved facts.
3. A decrease in the final answer should be a negative number and vice versa.

Show your feedback and give a binary score 'yes' or 'no' and reasoning. 'yes' means that the LLM generated code is consistent with the retrieved documents and no hallucination."""

GRADE_HALLUCINATION_USER_PROMPT = """## Question
{question}

## Retrieved Facts
{facts}

## LLM Generated Code
```python
{code}
```

---
Now take a deep breath and grade the LLM generated code."""


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    feedback: str = Field(
        description="Reasoning whether the raw facts in the code are aligned with the retrieved documents + feedback on how to improve"
    )
    binary_score: str = Field(
        description="Raw facts in the code are aligned with the retrieved documents, 'yes' or 'no'"
    )


llm = ChatOpenAI(
    model=config_experiment.GRADE_HALLUCINATION_MODEL,
    temperature=config_experiment.GRADE_HALLUCINATION_TEMPERATURE,
)

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_HALLUCINATION_SYSTEM_PROMPT),
        ("human", GRADE_HALLUCINATION_USER_PROMPT),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


# %%
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: formatted LLM final answer
        documents: list of documents
    """

    question: str
    generation: str
    context: str

    extracted_facts: str  # Used for hallucination check
    hallucination_grade: str  # Used for hallucination check
    messages: Annotated[list[str], add]  # Short-term memory
    previous_code: Annotated[list[str], add]  # Store prev code for self consistency


### Nodes


DEEP_EXTRACT_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions based on the attached financial report snippet. Your responses should be grounded in the report's content using the code interpreter tool for numerical calculations.

## Guidelines
1. Identify the most up-to-date relevant facts in the context needed for answering the question.
2. Perform calculations: Use the code interpreter tool for operations like summation, multiplication, or other numerical operations.
"""

EXTRACT_FACTS_USER_PROMPT = """## Context
{context}

## Question
{question}

---
Now perform the first step by identifying the most up-to-date relevant facts in the context needed for answering the question. All facts must be present in the original context. If no relevant facts are found, return "No relevant facts found"."""


def extract(state: GraphState):
    logger.info("---EXTRACT FACTS---")

    user_prompt = EXTRACT_FACTS_USER_PROMPT.format(
        context=state["context"],
        question=state["question"],
    )
    response = litellm.completion(
        model=config_experiment.PYTHON_AGENT_MODEL,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
        messages=[
            {"role": "system", "content": DEEP_EXTRACT_SYSTEM_PROMPT},
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


PROGRAM_REASONER_USER_PROMPT = """Now perform the second step by generating a python program to perform the calculation base on the previously extracted facts.

Please follow the following guidelines:
1. The generated python program should be executable with correct syntax.
2. The final answer should be assigned to the variable `ans`.
3. The `ans` variable should be a float number.
4. Enclose the python code in a code block using "```python" and "```".
5. If there is feedback on the previous generation, please incorporate it into the python program."""


def program_reasoner(state: GraphState):
    """
    Generate the code to answer the question
    """

    logger.info("---PROGRAM REASONER---")
    response = litellm.completion(
        model=config_experiment.PYTHON_AGENT_MODEL,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
        messages=[
            {"role": "system", "content": DEEP_EXTRACT_SYSTEM_PROMPT},
            *state["messages"],
            {"role": "user", "content": PROGRAM_REASONER_USER_PROMPT},
        ],
    )
    content = response["choices"][0]["message"]["content"]
    return {
        "messages": [
            {"role": "user", "content": PROGRAM_REASONER_USER_PROMPT},
            {"role": "assistant", "content": content},
        ],
        "previous_code": [content],
    }


def check_hallucination(state: GraphState):
    logger.info("---CHECK HALLUCINATIONS---")

    # Parse the code block from the last message (program reasoner)
    msg_w_code = state["messages"][-1]["content"]
    code = msg_w_code.split("```python")[1].split("```")[0].strip()
    logger.debug(f"Code:\n{code}\n")

    score: GradeHallucinations = hallucination_grader.invoke(
        {
            "facts": state["extracted_facts"],
            "question": state["question"],
            "code": code,
        }
    )
    grade = score.binary_score
    hallucination_grader_feedback = score.feedback
    return {
        "hallucination_grade": grade,
        "messages": [
            {
                "role": "assistant",
                "content": f"Passed code hallucination check: {grade}\nFeedback: {hallucination_grader_feedback}",
            }
        ],
    }


SELF_CONSISTENCY_SYSTEM_PROMPT = """You are a self-consistency code picker that picks the most popular code based on the previous code generations. Identify common patterns and choose the one with the highest confidence."""

SELF_CONSISTENCY_USER_PROMPT = """## Previous Code Generations
{previous_code}

---
Now take a deep breath and pick the most popular code based on the previous code generations. Enclose the code in a code block using "```python" and "```"."""


def self_consistency(state: GraphState):
    logger.info("---SELF CONSISTENCY CODE PICKER---")

    response = litellm.completion(
        model=config_experiment.PYTHON_AGENT_MODEL,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
        messages=[
            {"role": "system", "content": SELF_CONSISTENCY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SELF_CONSISTENCY_USER_PROMPT.format(
                    previous_code="\n".join(state["previous_code"])
                ),
            },
        ],
    )

    content = response["choices"][0]["message"]["content"]

    return {
        "messages": [{"role": "assistant", "content": f"{content}"}],
        "previous_code": [content],
    }


EXECUTE_USER_PROMPT = """Structure the Response Correctly: Format your final output with XML tags as follows
- Reasoning: Explain your retrieval or computation process within `<reasoning>` tags.
- Code: Show the executed code within `<code>` tags
- Final Answer: Provide the final response from the code execution within `<answer>` tags. Do not include other extra XML tags (e.g., `<output>`) or filler words.

## Key Constraints
- No Hallucination: If the required information is unavailable, return the default value specified in the question in the `<answer>` tag."""


def execute(state: GraphState):
    logger.info("--EXECUTING THE PYTHON PROGRAM--")

    # Can be either the last code block if hallucination check passed or the self-consistency code block
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


# def format_output(state: GraphState):
#     logger.info("--FORMAT OUTPUT--")

#     response = litellm.completion(
#         model=config_experiment.PYTHON_AGENT_MODEL,
#         temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
#         messages=[
#             *state["messages"],
#             {"role": "user", "content": EXECUTE_USER_PROMPT},
#         ],
#     )

#     content = response["choices"][0]["message"]["content"]
#     return {
#         "generation": content,
#     }


# --------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------


def hallucination_router(state: GraphState):
    """
    Route based on hallucination check result
    """
    if state["hallucination_grade"].lower() == "yes":
        logger.info(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS (NO HALLUCINATION)---"
        )
        return "execute"
    elif len(state["previous_code"]) >= 5:
        logger.info(
            "---DECISION: LOOPING DETECT. USE SELF CONSISTENCY TO PICK CODE. END---"
        )
        return "self_consistency"
    else:
        logger.info(
            "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY (HALLUCINATION)---"
        )
        return "program_reasoner"


# ## Build Graph
graph_builder_v6 = StateGraph(GraphState)

# Define the nodes and edges
graph_builder_v6.add_node("extract", extract)
graph_builder_v6.add_node("program_reasoner", program_reasoner)
graph_builder_v6.add_node("execute", execute)
# graph_builder_v6.add_node("format_output", format_output)
graph_builder_v6.add_node("check_hallucination", check_hallucination)
graph_builder_v6.add_node("self_consistency", self_consistency)
graph_builder_v6.add_edge(START, "extract")

graph_builder_v6.add_edge("extract", "program_reasoner")
graph_builder_v6.add_edge("program_reasoner", "check_hallucination")
graph_builder_v6.add_conditional_edges(
    "check_hallucination",
    hallucination_router,
    {
        "execute": "execute",
        "program_reasoner": "program_reasoner",
        "self_consistency": "self_consistency",
    },
)
graph_builder_v6.add_edge("self_consistency", "execute")
# graph_builder_v6.add_edge("execute", "format_output")
# graph_builder_v6.add_edge("format_output", END)
graph_builder_v6.add_edge("execute", END)

# --------------------------------------------------------------------------------------
# No hallucination check
# --------------------------------------------------------------------------------------
# agentic_rag_graph_builder.add_edge("generate", END)

# %%

if __name__ == "__main__":
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    display(
        Image(
            graph_builder_v6.compile()
            .get_graph()
            .draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )


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
    graph_inputs = {
        "question": question,
        "context": context,
    }

    # Compile graph and invoke
    graph = graph_builder_v6.compile()
    value = graph.invoke(input=graph_inputs)
    return value


if __name__ == "__main__":
    DATASET = "test"
    with open(f"paper/data/raw/FinQA/{DATASET}.json", "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} {DATASET} examples")

    # Load previously generated predictions if exists
    try:
        with open(
            f"paper/data/processed/FinQA/{DATASET}_pred_{config_experiment.OUR_METHOD_MODEL}.json",
            "r",
        ) as f:
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

        # Note: for debugging, set i > 3 to break
        # if i > 3:
        #     break

        question = d["qa"]["question"]

        pre_text = d.get("pre_text", [])
        table = d.get("table", [])
        post_text = d.get("post_text", [])

        # Convert list of strings to a single string
        pre_text_str = " ".join(pre_text)
        table_str = "\n".join([" ".join(row) for row in table])
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
        if i % 50 == 0:
            write_json_async(
                f"paper/data/processed/FinQA/{DATASET}_pred_{config_experiment.OUR_METHOD_MODEL}.json",
                list(pred_list),
            )
            logger.info(f"Checkpoint written at {i + 1}/{total_examples} examples")

    # Save the predictions to a JSON file (non-blocking)
    write_json_async(
        f"paper/data/processed/FinQA/{DATASET}_pred_{config_experiment.OUR_METHOD_MODEL}.json",
        pred_list,
    )
