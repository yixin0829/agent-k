# %%
from typing import Any

from dotenv import load_dotenv
from IPython.display import Image, display
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import src.config.experiment_config as config_experiment
import src.config.prompts as prompts
from src.config.logger import logger
from src.config.schemas import TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION
from src.utils.llm import (
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
    documents: list[str]

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
    system_prompt = prompts.DEEP_EXTRACT_SYSTEM_PROMPT
    user_prompt = prompts.GENERATION_USER_PROMPT_W_FEEDBACK.format(
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
        user_prompt = prompts.RETRIEVAL_GRADER_USER_PROMPT.format(
            document=d.page_content, question=question
        )
        system_prompt = prompts.RETRIEVAL_GRADER_SYSTEM_PROMPT.format(
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
    user = prompts.QUESTION_REWRITE_USER_PROMPT.format(question=question)
    messages = [
        {"role": "system", "content": prompts.QUESTION_REWRITE_SYSTEM_PROMPT},
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

    user = prompts.HALLUCINATION_GRADER_USER_PROMPT.format(
        documents=format_docs(documents), generation=generation
    )
    messages = [
        {"role": "system", "content": prompts.HALLUCINATION_GRADER_SYSTEM_PROMPT},
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

    system_prompt = prompts.ANSWER_GRADER_SYSTEM_PROMPT.format(
        schema=GradeAnswer.model_json_schema()
    )
    user_prompt = prompts.ANSWER_GRADER_USER_PROMPT.format(
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


graph_builder = StateGraph(GraphState)

# Define the nodes
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("generate", generate)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("check_hallucination", check_hallucination)
graph_builder.add_node("check_answers_question", check_answers_question)

# Build graph
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "regenerate": "generate",
    },
)
graph_builder.add_edge("generate", "check_hallucination")
graph_builder.add_edge("transform_query", "retrieve")
graph_builder.add_conditional_edges(
    "check_hallucination",
    hallucination_router,
    {
        "check_answers_question": "check_answers_question",
        "regenerate": "generate",
    },
)
graph_builder.add_conditional_edges(
    "check_answers_question",
    answer_quality_router,
    {
        "useful": END,
        "not_useful": "transform_query",
    },
)

# Compile
self_rag_graph = graph_builder.compile()

# %%
if __name__ == "__main__":
    # Demo usage
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

    question = prompts.QUESTION_TEMPLATE.format(
        field="total_mineral_resource_tonnage",
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )
    inputs = {
        "question": question,
        "generation": "N/A",
        "retriever": create_markdown_retriever(
            "data/processed/43-101_reports_refined/02a2b93ee61f2863bcb417b27855cb63d63a3c53b73622174f7c5688b0d4dc159c.md",
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
