# %%
from operator import add
from typing import Annotated, Any, List

import litellm
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import agent_k.config.experiment_config as config_experiment
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    QUESTION_TEMPLATE,
)
from agent_k.config.schemas import (
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from agent_k.notebooks.agentic_rag_v5 import create_markdown_retriever
from agent_k.tools.python_code_interpreter import PythonExecTool

load_dotenv()

CLIENT = OpenAI()
litellm.drop_params = True  # Ignore temperature parameter if model doesn't support it


# %%
### Hallucination Grader

GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination grader validating whether there is hallucination in a LLM's generated code. Focus on the calculation logic and unit conversions.

Guidelines:
1. Total mineral resource tonnage should be the sum of one or more of inferred, indicated, and measured mineral resources. If not, a default value of 0 should be returned.
2. Total mineral reserve tonnage should be the sum of one or more of proven and probable mineral reserves. If not, a default value of 0 should be returned.
3. The tonnage or grade unit used in the LLM generation should be consistent with the unit used in the retrieved documents. For example, "Tonnes 000", "Tonnes (000)", or "(000) Tonnes" mean thousand tonnes (Kt) or 1000 tonnes (t).
4. The unit of grade should be correctly converted to decimal before used in the calculation. For example, "10% Cu" should be converted to 0.10.
5. The final answer variable `ans` in the code should have its unit converted correctly to tonnes (t).

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
    documents: List[str]
    previous_answers: Annotated[list[str], add]  # Used for global hallucination check
    retriever: Any

    extracted_facts: str  # Used for hallucination check
    hallucination_grade: str  # Used for hallucination check
    global_hallucination_grade: str  # Used for global hallucination check
    messages: Annotated[list[str], add]  # Short-term memory
    previous_code: Annotated[list[str], add]  # Store prev code for self consistency
    generation: str


### Nodes


def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = state["retriever"].invoke(question)

    return {"documents": documents}


def get_mode_or_last(lst):
    """
    Return the unique mode if it exists, otherwise the last item.
    Optimized for short lists.
    """
    if not lst:
        logger.exception("Empty list provided to get_mode_or_last")

    try:
        # Use statistics.mode for unique mode, fallback to last if error
        from statistics import StatisticsError, mode

        return mode(lst)
    except (ImportError, StatisticsError):
        return lst[-1]


DEEP_EXTRACT_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions based on the attached NI 43-101 mineral report. Your responses should be grounded in the report's content using the code interpreter tool for numerical calculations.

## Guidelines
1. Identify the most up-to-date relevant facts in the context needed for answering the question in case there are multiple mineral estimates. Pay special attention to the unit of the field (e.g. "Tonnes 000" or "Kt" mean thousand tonnes).
2. Perform calculations: Use the code interpreter tool for operations like summation, multiplication, or other numerical operations.

## Key Constraints:
- No Hallucination: If the required information is unavailable, return the default value specified in the question."""

EXTRACT_FACTS_USER_PROMPT = """## Context
{context}

## Question
{question}

---
Now perform the first step by identifying the most up-to-date relevant facts in the context needed for answering the question. All facts must be present in the original context. If no relevant facts are found, return "No relevant facts found"."""


def extract(state: GraphState):
    logger.info("---EXTRACT FACTS---")

    user_prompt = EXTRACT_FACTS_USER_PROMPT.format(
        context=state["documents"],
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
    code_block_msg = state["previous_code"][-1]
    code = code_block_msg.split("```python")[1].split("```")[0].strip()
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


# %%
### Hallucination Grader

GLOBAL_GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination grader validating whether a LLM's generated code is consistent with previously extracted facts and generated code solutions.

Show your feedback and give a binary score 'yes' or 'no' and reasoning. 'yes' means that the LLM generated code is consistent with the previously extracted facts and generated code solutions."""

GLOBAL_GRADE_HALLUCINATION_USER_PROMPT = """## Previous QA Pairs
{previous_answers}

## Current Question
{question}

## LLM Generated Code
```python
{code}
```

---
Now take a deep breath and grade the LLM generated code."""


class GlobalGradeHallucinations(BaseModel):
    feedback: str = Field(
        description="Reasoning whether the LLM generated code is consistent with the previously extracted facts and generated code solutions + feedback on how to improve"
    )
    binary_score: str = Field(
        description="LLM generated code is consistent with the previously extracted facts and generated code solutions, 'yes' or 'no'"
    )


llm = ChatOpenAI(
    model=config_experiment.GRADE_HALLUCINATION_MODEL,
    temperature=config_experiment.GRADE_HALLUCINATION_TEMPERATURE,
)

global_structured_llm_grader = llm.with_structured_output(GlobalGradeHallucinations)
global_hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GLOBAL_GRADE_HALLUCINATION_SYSTEM_PROMPT),
        ("human", GLOBAL_GRADE_HALLUCINATION_USER_PROMPT),
    ]
)

global_hallucination_grader = global_hallucination_prompt | global_structured_llm_grader


def global_validator(state: GraphState):
    logger.info("---GLOBAL CHECK HALLUCINATION---")

    code_block_msg = state["previous_code"][-1]
    code = code_block_msg.split("```python")[1].split("```")[0].strip()

    if not state["previous_answers"]:
        previous_answers_str = "No previous QA pairs yet."
    else:
        previous_answers_str = ""
        for i, ans in enumerate(state["previous_answers"]):
            previous_answers_str += f"### QA Pair {i + 1}\n"
            previous_answers_str += f"{ans['question']}\n\nAnswer: {ans['answer']}\n\n"

    score: GlobalGradeHallucinations = global_hallucination_grader.invoke(
        {
            "previous_answers": previous_answers_str,
            "question": state["question"],
            "code": code,
        }
    )
    grade = score.binary_score
    feedback = score.feedback
    return {
        "global_hallucination_grade": grade,
        "messages": [
            {
                "role": "assistant",
                "content": f"Passed global hallucination check: {grade}\nFeedback: {feedback}",
            }
        ],
    }


# %%
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
    }


def format_output(state: GraphState):
    logger.info("--FORMAT OUTPUT--")

    response = litellm.completion(
        model=config_experiment.PYTHON_AGENT_MODEL,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
        messages=[
            *state["messages"],
            {"role": "user", "content": EXECUTE_USER_PROMPT},
        ],
    )

    content = response["choices"][0]["message"]["content"]
    return {
        "generation": content,
    }


# --------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------


def global_hallucination_router(state: GraphState):
    """
    Route based on global hallucination check result
    """
    if state["global_hallucination_grade"].lower() == "yes":
        logger.info("---DECISION: GLOBAL HALLUCINATION CHECK PASSED---")
        return "execute"
    elif len(state["previous_code"]) >= 5:
        logger.info(
            "---DECISION: LOOPING DETECT. USE SELF CONSISTENCY TO PICK CODE. END---"
        )
        return "self_consistency"
    else:
        logger.info("---DECISION: GLOBAL HALLUCINATION CHECK FAILED---")
        return "program_reasoner"


# ## Build Graph
graph_builder_v7 = StateGraph(GraphState)

# Define the nodes and edges
graph_builder_v7.add_node("retrieve", retrieve)
graph_builder_v7.add_node("extract", extract)
graph_builder_v7.add_node("program_reasoner", program_reasoner)
graph_builder_v7.add_node("execute", execute)
graph_builder_v7.add_node("format_output", format_output)
graph_builder_v7.add_node("check_hallucination", check_hallucination)
graph_builder_v7.add_node("global_validator", global_validator)
graph_builder_v7.add_node("self_consistency", self_consistency)
graph_builder_v7.add_edge(START, "retrieve")

graph_builder_v7.add_edge("retrieve", "extract")
graph_builder_v7.add_edge("extract", "program_reasoner")
graph_builder_v7.add_edge("program_reasoner", "check_hallucination")
graph_builder_v7.add_edge("check_hallucination", "global_validator")
graph_builder_v7.add_conditional_edges(
    "global_validator",
    global_hallucination_router,
    {
        "execute": "execute",
        "program_reasoner": "program_reasoner",
        "self_consistency": "self_consistency",
    },
)
graph_builder_v7.add_edge("self_consistency", "execute")
graph_builder_v7.add_edge("execute", "format_output")
graph_builder_v7.add_edge("format_output", END)

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
            graph_builder_v7.compile()
            .get_graph()
            .draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )


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

    graph_inputs = {
        "question": question,
        "retriever": retriever,
        "previous_answers": [],
    }

    # Compile graph and invoke
    graph = graph_builder_v7.compile()
    value = graph.invoke(input=graph_inputs)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
