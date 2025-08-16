# %%
from operator import add
from typing import Annotated, Any, List

import litellm
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import agent_k.config.experiment_config as config_experiment
from agent_k.config.logger import logger
from agent_k.config.prompts_fast_n_slow import (
    QUESTION_TEMPLATE,
)
from agent_k.config.schemas import (
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from agent_k.tools.python_code_interpreter import PythonExecTool
from agent_k.utils.general import extract_xml
from paper.experiments.utils import (
    count_tokens,
    create_markdown_retriever,
    invoke_model_messages,
)

load_dotenv()

litellm.drop_params = True  # Ignore temperature parameter if model doesn't support it


# %%
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: formatted LLM final answer
        documents: list of documents
    """

    markdown_path: str
    question: str
    generation: str
    documents: List[str]

    retriever: Any
    extracted_facts: str  # Used for hallucination check
    hallucination_grade: str  # Used for hallucination check
    messages: Annotated[list[str], add]  # Short-term memory
    previous_code: Annotated[list[str], add]  # Store prev code for self consistency


### Nodes


RETRIEVAL_SYSTEM_PROMPT = """You are an advanced AI assistant that retrieves relevant snippets and tables from the attached NI 43-101 mineral report for answering the question. Return all the retrieved snippets and tables in a list of Markdown strings as they are in the document."""

RETRIEVAL_USER_PROMPT = """## NI 43-101 Mineral Report
{md_content}

## Question
{question}

---
Now retrieve the most relevant snippets from the document for answering the question."""


def retrieve(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info(
        f"---RETRIEVE VIA {config_experiment.RETRIEVAL_METHOD.value.upper()}---"
    )
    question = state["question"]

    if config_experiment.RETRIEVAL_METHOD == config_experiment.RetrievalMethod.RAG:
        documents = state["retriever"].invoke(question)
    elif (
        config_experiment.RETRIEVAL_METHOD
        == config_experiment.RetrievalMethod.LONG_CONTEXT
    ):
        with open(state["markdown_path"], "r") as f:
            md_content = f.read()
            md_content_token_count = count_tokens(md_content)
            logger.info(f"MD content token count: {md_content_token_count}")

        # call model to retrieve relevant snippets from the document
        messages = [
            {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": RETRIEVAL_USER_PROMPT.format(
                    md_content=md_content, question=question
                ),
            },
        ]
        content = invoke_model_messages(
            model_name=config_experiment.RETRIEVAL_MODEL,
            messages=messages,
            temperature=config_experiment.RETRIEVAL_TEMPERATURE,
        )
        documents = [content]

    return {"documents": documents}


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
    messages = [
        {"role": "system", "content": DEEP_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    content = invoke_model_messages(
        model_name=config_experiment.PYTHON_AGENT_MODEL,
        messages=messages,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
    )
    return {
        "extracted_facts": content,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ],
    }


PROGRAM_REASONER_USER_PROMPT = """Now perform the second step by generating a python program to perform the calculation base on the previously extracted facts.

Please follow the following guidelines:
- The generated python program should be executable with correct syntax.
- The final answer should be assigned to the variable `ans`.
- The `ans` variable should be a float number and have its unit converted correctly to tonnes (t).
- Enclose the python code in a code block using "```python" and "```".
- If there is feedback on the previous generated python program, please incorporate it into the python program."""


def program_reasoner(state: GraphState):
    """
    Generate the code to answer the question
    """

    logger.info("---PROGRAM REASONER---")
    messages = [
        {"role": "system", "content": DEEP_EXTRACT_SYSTEM_PROMPT},
        *state["messages"],
        {"role": "user", "content": PROGRAM_REASONER_USER_PROMPT},
    ]
    content = invoke_model_messages(
        model_name=config_experiment.PYTHON_AGENT_MODEL,
        messages=messages,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
    )
    return {
        "messages": [
            {"role": "user", "content": PROGRAM_REASONER_USER_PROMPT},
            {"role": "assistant", "content": content},
        ],
        "previous_code": [content],
    }


# %%

GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination grader validating whether there is hallucination in a LLM's generated code (Yes means hallucination, No means no hallucination). Focus on the calculation logic and unit conversions.

Guidelines:
- Total mineral resource tonnage should be the sum of one or more of inferred, indicated, and measured mineral resources. If not, a default value of 0 should be returned.
- Total mineral reserve tonnage should be the sum of one or more of proven and probable mineral reserves. If not, a default value of 0 should be returned.
- The tonnage or grade unit used in the LLM generation should be consistent with the unit used in the retrieved documents. For example, "Tonnes 000", "Tonnes (000)", or "(000) Tonnes" mean thousand tonnes (Kt) or 1000 tonnes (t).
- The unit of grade should be correctly converted to decimal before used in the calculation. For example, "10% Cu" should be converted to 0.10.
- The final answer should be assigned to the variable `ans` in the code
- The final answer `ans` should have its unit converted correctly to tonnes (t).

Output Format:
Respond strictly with the following XML tags (no introductory or extra text):
<hallucination_grade>yes or no</hallucination_grade>
<feedback>Brief explanation of your assessment and detected issues if any</feedback>
"""

GRADE_HALLUCINATION_USER_PROMPT = """## Question
{question}

## Retrieved Facts
{facts}

## LLM Generated Code
```python
{code}
```

---
Now take a deep breath and grade the LLM generated code. Output ONLY the XML specified in the system prompt."""


def check_hallucination(state: GraphState):
    logger.info("---CHECK HALLUCINATIONS---")

    # Parse the code block from the last message (program reasoner)
    msg_w_code = state["messages"][-1]["content"]

    try:
        code = msg_w_code.split("```python")[1].split("```")[0].strip()
        logger.debug(f"Code:\n{code}\n")

        # Call the hallucination grader via provider-agnostic JSON
        messages = [
            {"role": "system", "content": GRADE_HALLUCINATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GRADE_HALLUCINATION_USER_PROMPT.format(
                    facts=state["extracted_facts"],
                    question=state["question"],
                    code=code,
                ),
            },
        ]
        content = invoke_model_messages(
            model_name=config_experiment.GRADE_HALLUCINATION_MODEL,
            messages=messages,
            temperature=config_experiment.GRADE_HALLUCINATION_TEMPERATURE,
        )
        grade = extract_xml(content, "hallucination_grade")
        hallucination_grader_feedback = extract_xml(content, "feedback")
    except IndexError:
        logger.error(f"No code block found in the last message: {msg_w_code}")
        grade = "no"
        hallucination_grader_feedback = "No ```python code block found"

    return {
        "hallucination_grade": grade,
        "messages": [
            {
                "role": "assistant",
                "content": f"Detected code hallucination: {grade}\nFeedback: {hallucination_grader_feedback}",
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

    messages = [
        {"role": "system", "content": SELF_CONSISTENCY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SELF_CONSISTENCY_USER_PROMPT.format(
                previous_code="\n".join(state["previous_code"])
            ),
        },
    ]
    content = invoke_model_messages(
        model_name=config_experiment.PYTHON_AGENT_MODEL,
        messages=messages,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
    )

    return {
        "messages": [{"role": "assistant", "content": f"{content}"}],
        "previous_code": [content],
    }


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


FORMAT_OUTPUT_USER_PROMPT = """Structure the Response Correctly: Format your final output with XML tags as follows
- Reasoning: Explain your retrieval or computation process within `<reasoning>` tags.
- Code: Show the executed code within `<code>` tags
- Final Answer: Provide the final response from the code execution within `<answer>` tags. Do not include other extra XML tags (e.g., `<output>`) or filler words.

## Key Constraints
- No Hallucination: If the required information is unavailable, return the default value specified in the question in the `<answer>` tag."""


def format_output(state: GraphState):
    logger.info("--FORMAT OUTPUT--")

    messages = [
        *state["messages"],
        {"role": "user", "content": FORMAT_OUTPUT_USER_PROMPT},
    ]
    content = invoke_model_messages(
        model_name=config_experiment.PYTHON_AGENT_MODEL,
        messages=messages,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
    )
    return {
        "generation": content,
    }


# --------------------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------------------


def hallucination_router(state: GraphState):
    """
    Route based on hallucination check result
    """
    if state["hallucination_grade"].lower() == "no":
        logger.info(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS (NO HALLUCINATION)---"
        )
        return "execute"
    elif len(state["previous_code"]) >= config_experiment.MAX_REFLECTION_ITERATIONS:
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
graph_builder_v6.add_node("retrieve", retrieve)
graph_builder_v6.add_node("extract", extract)
graph_builder_v6.add_node("program_reasoner", program_reasoner)
graph_builder_v6.add_node("execute", execute)
graph_builder_v6.add_node("format_output", format_output)
graph_builder_v6.add_node("check_hallucination", check_hallucination)
graph_builder_v6.add_node("self_consistency", self_consistency)
graph_builder_v6.add_edge(START, "retrieve")

graph_builder_v6.add_edge("retrieve", "extract")
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
graph_builder_v6.add_edge("execute", "format_output")
graph_builder_v6.add_edge("format_output", END)

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
if __name__ == "__main__":
    question = QUESTION_TEMPLATE.format(
        field="total_mineral_resource_tonnage",
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )

    markdown_path = "paper/data/processed/43-101-refined/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md"
    if config_experiment.RETRIEVAL_METHOD == config_experiment.RetrievalMethod.RAG:
        retriever = create_markdown_retriever(
            markdown_path,
            collection_name="rag-chroma",
        )
    elif (
        config_experiment.RETRIEVAL_METHOD
        == config_experiment.RetrievalMethod.LONG_CONTEXT
    ):
        retriever = None

    graph_inputs = {
        "markdown_path": markdown_path,
        "question": question,
        "retriever": retriever,
    }

    # Compile graph and invoke
    graph = graph_builder_v6.compile()
    value = graph.invoke(input=graph_inputs)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
