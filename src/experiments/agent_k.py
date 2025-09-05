# %%
from operator import add
from typing import Annotated, Any

import litellm
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import src.config.experiment_config as config_experiment
import src.config.prompts as prompts
from src.config.logger import logger
from src.config.schemas import (
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from src.utils.code_interpreter import PythonExecTool
from src.utils.general import count_tokens, extract_xml
from src.utils.llm import (
    create_markdown_retriever,
    invoke_model_messages,
)

load_dotenv()

# --------------------------------------------------------------------------------------
# Configuration Variables
# --------------------------------------------------------------------------------------

# LiteLLM configuration
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
    property_name: str
    question: str
    generation: str
    documents: list[str]

    retriever: Any
    extracted_facts: str  # Used for self-reflection check
    self_reflection_issues_found: str  # Used for self-reflection check
    messages: Annotated[list[str], add]  # Short-term memory
    previous_code: Annotated[list[str], add]  # Store prev code for self consistency


### Nodes


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
            {"role": "system", "content": prompts.RETRIEVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.RETRIEVAL_USER_PROMPT.format(
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


def fact_extraction_agent(state: GraphState):
    logger.info("---FACT EXTRACTION AGENT---")

    user_prompt = prompts.FACT_EXTRACTION_AGENT_USER_PROMPT.format(
        context=state["documents"],
        question=state["question"],
    )
    messages = [
        {"role": "system", "content": prompts.FACT_EXTRACTION_AGENT_SYSTEM_PROMPT},
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


def program_reasoner(state: GraphState):
    """
    Generate the code to answer the question
    """

    logger.info("---PROGRAM REASONER---")
    messages = [
        {"role": "system", "content": prompts.REACT_AGENT_SYSTEM_PROMPT},
        *state["messages"],
        {"role": "user", "content": prompts.PROGRAM_REASONER_USER_PROMPT},
    ]
    content = invoke_model_messages(
        model_name=config_experiment.PYTHON_AGENT_MODEL,
        messages=messages,
        temperature=config_experiment.PYTHON_AGENT_TEMPERATURE,
    )
    return {
        "messages": [
            {"role": "user", "content": prompts.PROGRAM_REASONER_USER_PROMPT},
            {"role": "assistant", "content": content},
        ],
        "previous_code": [content],
    }


# %%


def self_reflection(state: GraphState):
    logger.info("---SELF REFLECTION---")

    property_knowledge = {
        "total_mineral_resource_tonnage": (
            "Total mineral resource tonnage should be the sum of one or more of inferred, indicated, and measured mineral resources. "
            "If no relevant values are found, a default value of 0 should be returned."
        ),
        "total_mineral_reserve_tonnage": (
            "Total mineral reserve tonnage should be the sum of one or more of proven and probable mineral reserves. "
            "If no relevant values are found, a default value of 0 should be returned."
        ),
        "total_mineral_resource_contained_metal": (
            "Total mineral resource contained metal should be the sum of one or more of inferred, indicated, and measured mineral resources contained metal. "
            "How you calculate the contained metal is by multiplying the inferred, indicated, and measured mineral resources tonnage with the corresponding grade. "
            "If no relevant values are found, a default value of 0 should be returned."
        ),
        "total_mineral_reserve_contained_metal": (
            "Total mineral reserve contained metal should be the sum of one or more of proven and probable mineral reserves contained metal. "
            "How you calculate the contained metal is by multiplying the proven and probable mineral reserves tonnage with the corresponding grade. "
            "If no relevant values are found, a default value of 0 should be returned."
        ),
    }

    property_knowledge = property_knowledge[state["property_name"]]

    # Parse the code block from the last message (program reasoner)
    msg_w_code = state["messages"][-1]["content"]

    try:
        code = msg_w_code.split("```python")[1].split("```")[0].strip()
        logger.debug(f"Code:\n{code}\n")

        # Call the self-reflection grader via provider-agnostic JSON
        messages = [
            {"role": "system", "content": prompts.REACT_AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.SELF_REFLECTION_USER_PROMPT.format(
                    facts=state["extracted_facts"],
                    question=state["question"],
                    code=code,
                    property_knowledge=property_knowledge,
                ),
            },
        ]
        content = invoke_model_messages(
            model_name=config_experiment.GRADE_HALLUCINATION_MODEL,
            messages=messages,
            temperature=config_experiment.GRADE_HALLUCINATION_TEMPERATURE,
        )
        self_reflection_issues_found = extract_xml(content, "issues_found")
        self_reflection_feedback = extract_xml(content, "feedback")
    except IndexError:
        logger.error(f"No code block found in the last message: {msg_w_code}")
        self_reflection_issues_found = "no"
        self_reflection_feedback = "No ```python code block found"

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
    logger.info("---SELF CONSISTENCY CODE PICKER---")

    messages = [
        {"role": "system", "content": prompts.REACT_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompts.SELF_CONSISTENCY_USER_PROMPT.format(
                previous_code="\n\n".join(state["previous_code"])
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

    # Can be either the last code block if self-reflection passed or the self-consistency code block
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

    messages = [
        *state["messages"],
        {"role": "user", "content": prompts.FORMAT_OUTPUT_USER_PROMPT},
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


def reflection_router(state: GraphState):
    """
    Route based on self-reflection check result
    """
    if state["self_reflection_issues_found"].lower() == "no":
        logger.info("---DECISION: CODE PASSED SELF-REFLECTION (NO ISSUES FOUND)---")
        return "execute"
    elif len(state["previous_code"]) >= config_experiment.MAX_REFLECTION_ITERATIONS:
        logger.info(
            "---DECISION: MAX REFLECTION ITERATIONS REACHED. USE SELF CONSISTENCY TO PICK CODE---"
        )
        return "self_consistency"
    else:
        logger.info("---DECISION: CODE ISSUES FOUND IN SELF-REFLECTION, RE-TRY---")
        return "program_reasoner"


# ## Build Graph
graph_builder = StateGraph(GraphState)

# Define the nodes and edges
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("fact_extraction_agent", fact_extraction_agent)
graph_builder.add_node("program_reasoner", program_reasoner)
graph_builder.add_node("execute", execute)
graph_builder.add_node("format_output", format_output)
graph_builder.add_node("self_reflection", self_reflection)
graph_builder.add_node("self_consistency", self_consistency)
graph_builder.add_edge(START, "retrieve")

graph_builder.add_edge("retrieve", "fact_extraction_agent")
graph_builder.add_edge("fact_extraction_agent", "program_reasoner")
graph_builder.add_edge("program_reasoner", "self_reflection")
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
graph_builder.add_edge("execute", "format_output")
graph_builder.add_edge("format_output", END)


# %%

if __name__ == "__main__":
    from IPython.display import Image, display
    from langchain_core.runnables.graph import MermaidDrawMethod

    display(
        Image(
            graph_builder.compile()
            .get_graph()
            .draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )


# %%
if __name__ == "__main__":
    # Demo usage
    property_name = "total_mineral_resource_tonnage"
    question = prompts.QUESTION_TEMPLATE.format(
        field=property_name,
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )

    markdown_path = "data/processed/43-101_reports_refined/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md"
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
        "property_name": property_name,
        "question": question,
        "retriever": retriever,
    }

    # Compile graph and invoke
    graph = graph_builder.compile()
    value = graph.invoke(input=graph_inputs)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
