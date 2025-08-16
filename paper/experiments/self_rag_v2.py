# %% [markdown]
# Cleaned version of self_rag.ipynb ready for integrating into evaluation.
#
# Changes to the original LangGraph Implementation:
# 1. Used MarkdownTextSplitter
# 2. Customized QUESTION_TEMPLATE same as fast_n_slow experiments (question, dtype, default value, and description)
# 3. Decouple the route and state update logic for hallucination and answer grade
# 4. Optimize the prompt for RAG chain (w feedback and wo feedback)
# 5. Add unit (e.g. "Tonnes 000") checking knowledge to Hallucination grader system prompt
# 6. Integrate into pdf_agent_fast_n_slow.py run
# 7. Run 1st eval

# %%
from typing import Any, List

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import agent_k.config.experiment_config as config_experiment
from agent_k.config.logger import logger
from agent_k.config.schemas import TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION
from paper.experiments.utils import (
    create_markdown_retriever,
    invoke_json,
    invoke_model_messages,
)

load_dotenv()


# %%
### Retrieval Grader


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    reasoning: str = Field(
        description="Reasoning why the document is relevant to the question or not"
    )
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


QUESTION_TEMPLATE = """**Question:** What's the {field} of the mineral site in the attached 43-101 report?
**Data type of {field}:** {dtype}
**Default value of {field} if not found:** {default}
**Description of {field}:** {description}"""


# Retrieval grader prompts (plain strings)
RETRIEVAL_GRADER_SYSTEM_PROMPT = """
You are a grader assessing relevance of a retrieved document to a user question. No need to be super strict.
If the document contains keywords or semantic meaning related to the question, consider it relevant.

## Output format
- Respond with a single JSON object that conforms to the JSON schema (Pydantic v2): {schema}
"""

RETRIEVAL_GRADER_USER_PROMPT = """Retrieved document:\n\n{document}

User question:\n\n{question}

---
Grade the document relevance and return only JSON per the JSON schema."""

# %%
### Hallucination Grader (provider-agnostic JSON)


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    reasoning: str = Field(
        description="Reasoning why the answer is grounded in the facts or not grounded in the facts"
    )
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are a grader validating whether an LLM generation is grounded in a set of retrieved documents from a NI 43-101 mineral report.

Guidelines:
1. If the question is about mineral resources, check if the retrieved documents mention inferred, indicated, and measured resources. If none of the retrieved documents mention inferred, indicated, or measured resources, check if the LLM generation contains a default value of 0 for total mineral resource tonnage.
2. If the question is about mineral reserves, check if the retrieved documents mention proven and probable reserves. If none of the retrieved documents mention proven or probable reserves, check if the LLM generation contains a default value of 0 for total mineral reserve tonnage.
3. Check if the units of the mineral resources or reserves in the retrieved documents are consistent with the units of the mineral resources or reserves in the LLM generation. Especially pay attention if the retrieved documents mention "Tonnes 000" or something similar, which means that the tonnage is in thousands of tonnes.
4. Check if the final numerical answer is enclosed in `<answer>` XML tags without any other XML tags, filler words, or explicit unit.

Respond with a single JSON object and nothing else. Schema: {"reasoning": string, "binary_score": "yes" | "no"}
"""

HALLUCINATION_GRADER_USER_PROMPT = """Set of retrieved documents:

{documents}

LLM generation:

{generation}

Return only JSON per the schema.
"""

# %%
### Answer Grader (provider-agnostic JSON)


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    reasoning: str = Field(
        description="Reasoning why the answer addresses the question or not"
    )
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


ANSWER_GRADER_SYSTEM_PROMPT = """
You are a grader assessing whether an answer addresses / resolves a question.

Sometimes a default value is returned because the retrieved documents do not contain relevant information. In this case, the answer should be 'yes' because the default value is still a valid answer to the question.

## Output format
- Respond with a single JSON object and nothing else. JSON schema (Pydantic v2): {schema}
"""

ANSWER_GRADER_USER_PROMPT = """Question:\n\n{question}\n\nLLM generation:\n\n{generation}\n\nReturn only JSON per the schema."""

# %%
### Question Re-writer

# Provider-agnostic question rewriter
QUESTION_REWRITE_SYSTEM_PROMPT = (
    "You are a question re-writer that converts an input question to a better version "
    "optimized for vectorstore retrieval. Consider semantic intent."
)

QUESTION_REWRITE_USER_PROMPT = "Here is the initial question:\n\n---\n{question}\n---\n\nFormulate an improved question."

# %% [markdown]
# # Graph


# %%
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    retriever: Any
    documents: List[str]

    hallucination_grade: str
    hallucination_grader_reasoning: str
    answer_grade: str
    answer_grader_reasoning: str


# %%
### Nodes


def retrieve(state):
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


# %%
### Generate (provider-agnostic)

# Deep Extraction prompts
DEEP_EXTRACT_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions based on the attached NI 43-101 mineral report snippets. Your responses should be grounded in the report's content using the code interpreter tool for numerical calculations if needed.

## Output Format
- Reasoning: Explain your retrieval or computation process within `<thinking>` XML tags.
- Final Answer: Provide the final response within `<answer>` XML tags.

## Key Constraints
- No Hallucination: If the required information is unavailable, return the default value specified in the JSON schema in the `<answer>` tag.
"""

GENERATION_USER_PROMPT_W_FEEDBACK = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just return the default value of the field in the question.

{question}

**Context:**
{context}

**Feedback:**
{feedback}
---
Now take a deep breath and return only the final answer wrapped in XML tags."""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    hallucination_grade = state["hallucination_grade"]
    hallucination_grader_reasoning = state["hallucination_grader_reasoning"]
    feedback = f"Hallucination grade: {hallucination_grade}\nHallucination grader reasoning: {hallucination_grader_reasoning}"

    # Provider-agnostic generation (no feedback path for now, to match original behavior)
    system_prompt = DEEP_EXTRACT_SYSTEM_PROMPT
    user_prompt = GENERATION_USER_PROMPT_W_FEEDBACK.format(
        question=question, context=format_docs(documents), feedback=feedback
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    generation = invoke_model_messages(
        model_name=config_experiment.SELF_RAG_GENERATION_MODEL,
        messages=messages,
        temperature=config_experiment.SELF_RAG_GENERATION_TEMPERATURE,
    )

    return {"generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc via provider-agnostic JSON grader
    filtered_docs = []
    for d in documents:
        user_prompt = RETRIEVAL_GRADER_USER_PROMPT.format(
            document=d.page_content, question=question
        )
        system_prompt = RETRIEVAL_GRADER_SYSTEM_PROMPT.format(
            schema=GradeDocuments.model_json_schema()
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        score = invoke_json(
            model_name=config_experiment.SELF_RAG_GRADE_RETRIEVAL_MODEL,
            messages=messages,
            temperature=config_experiment.SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE,
            schema=GradeDocuments,
        )
        grade = (score.binary_score or "").strip().lower()
        if grade == "yes":
            logger.info("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"documents": filtered_docs}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    logger.info("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question via provider-agnostic call
    user = QUESTION_REWRITE_USER_PROMPT.format(question=question)
    messages = [
        {"role": "system", "content": QUESTION_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    better_question = invoke_model_messages(
        model_name=config_experiment.SELF_RAG_QUESTION_REWRITER_MODEL,
        messages=messages,
        temperature=config_experiment.SELF_RAG_QUESTION_REWRITER_TEMPERATURE,
    )
    return {"documents": documents, "question": better_question}


def check_hallucination(state):
    """
    First node: Check if generation is grounded in documents.

    Args:
        state (dict): The current graph state
    Returns:
        dict: State updates and next node
    """
    logger.info("---CHECK HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]

    user = HALLUCINATION_GRADER_USER_PROMPT.format(
        documents=format_docs(documents), generation=generation
    )
    messages = [
        {"role": "system", "content": HALLUCINATION_GRADER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    score = invoke_json(
        model_name=config_experiment.SELF_RAG_GRADE_HALLUCINATION_MODEL,
        messages=messages,
        temperature=config_experiment.SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE,
        schema=GradeHallucinations,
    )
    grade = (score.binary_score or "").strip().lower()
    hallucination_grader_reasoning = getattr(score, "reasoning", "")

    return {
        "hallucination_grade": grade,
        "hallucination_grader_reasoning": hallucination_grader_reasoning,
    }


def check_answers_question(state):
    """
    Second node: Check if generation answers the question.

    Args:
        state (dict): The current graph state
    Returns:
        dict: State updates and next node
    """
    logger.info("---GRADE GENERATION vs QUESTION---")
    question = state["question"]
    generation = state["generation"]

    system_prompt = ANSWER_GRADER_SYSTEM_PROMPT.format(
        schema=GradeAnswer.model_json_schema()
    )
    user_prompt = ANSWER_GRADER_USER_PROMPT.format(
        question=question, generation=generation
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    score = invoke_json(
        model_name=config_experiment.SELF_RAG_GRADE_ANSWER_MODEL,
        messages=messages,
        temperature=config_experiment.SELF_RAG_GRADE_ANSWER_TEMPERATURE,
        schema=GradeAnswer,
    )
    grade = (score.binary_score or "").strip().lower()
    answer_grader_reasoning = getattr(score, "reasoning", "")

    return {
        "answer_grade": grade,
        "answer_grader_reasoning": answer_grader_reasoning,
    }


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.info("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        logger.info("---DECISION: GENERATE---")
        return "regenerate"


def hallucination_router(state):
    """
    Route based on hallucination check result
    """
    if state["hallucination_grade"].lower() == "yes":
        logger.info(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS (NO HALLUCINATION)---"
        )
        return "check_answers_question"
    else:
        logger.info(
            "---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY (HALLUCINATION)---"
        )
        logger.info(
            f"Reasoning on hallucination: {state['hallucination_grader_reasoning']}"
        )
        return "regenerate"


def answer_quality_router(state):
    """
    Route based on answer quality check result
    """
    if state["answer_grade"] == "yes":
        logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
        logger.info(
            f"Reasoning on why generation addresses question: {state['answer_grader_reasoning']}"
        )
        return "useful"
    else:
        logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        logger.info(
            f"Reasoning on why generation does not address question: {state['answer_grader_reasoning']}"
        )
        return "not_useful"


# %%
# Build Graph


def viz_graph(graph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass


self_rag_graph_builder = StateGraph(GraphState)

# Define the nodes
self_rag_graph_builder.add_node("retrieve", retrieve)  # retrieve
self_rag_graph_builder.add_node("grade_documents", grade_documents)  # grade documents
self_rag_graph_builder.add_node("generate", generate)  # generatae
self_rag_graph_builder.add_node("transform_query", transform_query)  # transform_query
self_rag_graph_builder.add_node(
    "check_hallucination", check_hallucination
)  # check hallucination
self_rag_graph_builder.add_node(
    "check_answers_question", check_answers_question
)  # check answers question

# Build graph
self_rag_graph_builder.add_edge(START, "retrieve")
self_rag_graph_builder.add_edge("retrieve", "grade_documents")
self_rag_graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "regenerate": "generate",
    },
)
self_rag_graph_builder.add_edge("generate", "check_hallucination")
self_rag_graph_builder.add_edge("transform_query", "retrieve")
self_rag_graph_builder.add_conditional_edges(
    "check_hallucination",
    hallucination_router,
    {
        "check_answers_question": "check_answers_question",
        "regenerate": "generate",
    },
)
self_rag_graph_builder.add_conditional_edges(
    "check_answers_question",
    answer_quality_router,
    {
        "useful": END,
        "not_useful": "transform_query",
    },
)

# Compile
self_rag_graph = self_rag_graph_builder.compile()

# Visualize the graph
viz_graph(self_rag_graph)

# %%
# Run
if __name__ == "__main__":
    # Log Self RAG Experiment hyper-parameters
    logger.info(
        f"SELF_RAG_GRADE_RETRIEVAL_MODEL: {config_experiment.SELF_RAG_GRADE_RETRIEVAL_MODEL}"
    )
    logger.info(
        f"SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE: {config_experiment.SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE}"
    )
    logger.info(
        f"SELF_RAG_GENERATION_MODEL: {config_experiment.SELF_RAG_GENERATION_MODEL}"
    )
    logger.info(
        f"SELF_RAG_GENERATION_TEMPERATURE: {config_experiment.SELF_RAG_GENERATION_TEMPERATURE}"
    )
    logger.info(
        f"SELF_RAG_GRADE_HALLUCINATION_MODEL: {config_experiment.SELF_RAG_GRADE_HALLUCINATION_MODEL}"
    )
    logger.info(
        f"SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE: {config_experiment.SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE}"
    )
    logger.info(
        f"SELF_RAG_GRADE_ANSWER_MODEL: {config_experiment.SELF_RAG_GRADE_ANSWER_MODEL}"
    )
    logger.info(
        f"SELF_RAG_GRADE_ANSWER_TEMPERATURE: {config_experiment.SELF_RAG_GRADE_ANSWER_TEMPERATURE}"
    )
    logger.info(
        f"SELF_RAG_QUESTION_REWRITER_MODEL: {config_experiment.SELF_RAG_QUESTION_REWRITER_MODEL}"
    )
    logger.info(
        f"SELF_RAG_QUESTION_REWRITER_TEMPERATURE: {config_experiment.SELF_RAG_QUESTION_REWRITER_TEMPERATURE}"
    )

    question = QUESTION_TEMPLATE.format(
        field="total_mineral_resource_tonnage",
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )
    inputs = {
        "question": question,
        "generation": "N/A",
        "retriever": create_markdown_retriever(
            "data/processed/43-101/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md",
            collection_name="rag-chroma",
        ),
        "hallucination_grade": "N/A",
        "hallucination_grader_reasoning": "N/A",
        "answer_grade": "N/A",
        "answer_grader_reasoning": "N/A",
    }

    value = self_rag_graph.invoke(inputs)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
