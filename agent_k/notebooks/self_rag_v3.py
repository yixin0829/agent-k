# %% [markdown]
# Cleaned version of self_rag.ipynb ready for integrating into evaluation.
#
# Changes compared to v2 implementation:
# - Switch hallucination grader to open assistant with code interpreter as well
# - Add messages list for long-term memory
# - Change tonnage and contained metal unit from Mt to tonnes to simplify the problem
# - If default value 0 is generated, then go directly to END since no need to check hallucination
#     - Assume 0 is always no hallucination (i.e. LLM does not hallucinate no hallucination)
# - Changed main validator prompt in VALIDATOR_SYSTEM_PROMPT to set contained metal to zero if tonnage is zero

# %%
import re
from typing import Annotated, Any, List, Optional

import chromadb
import tiktoken
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_k.config.logger import logger
from agent_k.config.schemas import (
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from agent_k.utils.general import (
    prompt_openai_assistant,
)

load_dotenv()

CLIENT = OpenAI()
NUM_RETRIEVED_DOCS = 5
MODEL = "gpt-4o-mini"


def count_tokens(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        logger.warning("Falling back to rough approximation (4 tokens per character)")
        return len(text) // 4


def create_markdown_retriever(
    markdown_path: str,
    collection_name: str,
    headers_to_split_on: Optional[list[tuple[str, str]]] = None,
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    Creates a Chroma retriever from a markdown document.

    Args:
        markdown_path: Path to the markdown file
        collection_name: Name for the Chroma collection
        headers_to_split_on: List of tuples containing markdown header levels and their names
        embedding_model: Name of the OpenAI embedding model to use

    Returns:
        Chroma retriever object
    """
    # Set default headers if none provided
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    # Read markdown file
    try:
        with open(markdown_path, "r", encoding="utf-8") as file:
            markdown_document = file.read()
    except Exception as e:
        logger.error(f"Error reading markdown file: {e}")
        raise

    # Split document
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    doc_splits = markdown_splitter.split_text(markdown_document)

    # Log splitting information
    try:
        markdown_document_tokens = count_tokens(markdown_document)
        doc_splits_len = len(doc_splits)
        avg_tokens = markdown_document_tokens / doc_splits_len

        logger.info(f"Number of tokens: {markdown_document_tokens}")
        logger.info(f"Number of splits: {doc_splits_len}")
        logger.info(f"Average tokens per split: {avg_tokens:.0f}")
    except Exception as e:
        logger.warning(f"Could not log token statistics: {e}")

    # Create vectorstore and retriever
    try:
        # Initialize the Chroma client
        client = chromadb.Client()

        # Delete existing in-memory collection if it exists
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

        # Create a new vectorstore with persist_directory=None to keep it in-memory
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=collection_name,
            embedding=OpenAIEmbeddings(model=embedding_model),
            persist_directory=None,  # Keep in-memory to avoid persistence
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": NUM_RETRIEVED_DOCS})
        return retriever
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


# Example usage:
retriever = create_markdown_retriever(
    "data/processed/43-101/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md",
    collection_name="rag-chroma",
)

results = retriever.invoke("What's the mineral site name?")

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


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

QUESTION_TEMPLATE = """**Question:** What's the {field} of the mineral site in the attached 43-101 report?
**Data type of {field}:** {dtype}
**Default value of {field} if not found:** {default}
**Description of {field}:** {description}"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "# Retrieved document\n{document}\n\n# User question\n{question}"),
    ]
)

template = "# Retrieved document\n{document}\n\n# User question\n{question}"
print(template.format(document="[sample document]", question="[sample question]"))

retrieval_grader = grade_prompt | structured_llm_grader

# %%
### Generate

# Deep Extraction Assistant (Sync with OpenAI Assistant)
DEEP_EXTRACT_SYSTEM_PROMPT = """You are an advanced AI assistant that answers questions based on the attached NI 43-101 mineral report. Your responses should be grounded in the report's content using the code interpreter tool for numerical calculations.

## Response Workflow:
1. Perform Aggregations: Use the code interpreter tool for operations like summation, multiplication, or other numerical operations.
2. Structure the Response Correctly: Format your final output with XML tags as follows:
    - Reasoning: Explain your retrieval or computation process within `<thinking>` tags.
    - Final Answer: Provide the final response within `<output>` tags. Do not include other extra XML tags (e.g., `<answer>`) or filler words.

## Key Constraints:
- No Hallucination: If the required information is unavailable, return the default value specified in the JSON schema in the `<output>` tag.
"""

GENERATION_USER_PROMPT_WO_FEEDBACK = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just return the default value of the field in the question.

{question}
**Context:** {context}
---
Now take a deep breath and answer the question step by step."""

GENERATION_USER_PROMPT_W_FEEDBACK = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just return the default value of the field in the question.

{question}
**Context:** {context}
**Previous generation and feedback:** {previous_messages}
---
Now take a deep breath and answer the question again step by step while considering the feedback to the previous incorrect answer."""


### Alternative RAG chain using OpenAI Assistant


# OpenAI Assistant with code interpreter tool
# @traceable(run_type="llm", name="deep_extract_wo_feedback")
def deep_extract_wo_feedback(question, context) -> str:
    # Use the same assistant for all deep extraction
    assistant = CLIENT.beta.assistants.retrieve("asst_ScLkkCGSfMVgR8xEWkRDbZMq")

    messages = [
        {
            "role": "user",
            "content": GENERATION_USER_PROMPT_WO_FEEDBACK.format(
                question=question,
                context=context,
            ),
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


# @traceable(run_type="llm", name="deep_extract_w_feedback")
def deep_extract_w_feedback(question, context, previous_messages) -> str:
    # Use the same assistant for all deep extraction
    assistant = CLIENT.beta.assistants.retrieve("asst_ScLkkCGSfMVgR8xEWkRDbZMq")

    messages = [
        {
            "role": "user",
            "content": GENERATION_USER_PROMPT_W_FEEDBACK.format(
                question=question,
                context=context,
                previous_messages=previous_messages,
            ),
        },
    ]

    content = prompt_openai_assistant(assistant, messages)

    return content


# %%
### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    reasoning: str = Field(
        description="Reasoning whether the raw facts in the answer are aligned with the retrieved documents"
    )
    binary_score: str = Field(
        description="Raw facts in the answer are aligned with the retrieved documents, 'yes' or 'no'"
    )


# generate json schema for GradeHallucinations (Update in OpenAI Assistant)
json_schema = GradeHallucinations.model_json_schema()

# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt (Update in OpenAI Assistant)
system = """You are a grader assistant validating whether a LLM's generation is consistent with the retrieved documents from a NI 43-101 mineral report.

Guidelines:
1. Check if total mineral resource tonnage is the sum of inferred, indicated, and measured mineral resources. Otherwise, it is a hallucination and specified default value should be returned.
2. Check if total mineral reserve tonnage is the sum of proven and probable mineral reserves. Otherwise, it is a hallucination and specified default value should be returned.
3. Check no hallucination in total mineral resource contained metal and total mineral reserve contained metal calculation.
4. Check if the tonnage or grade unit used in the LLM generation is consistent with the unit used in the retrieved documents. For example, "Tonnes 000", "Tonnes (000)", or "(000) Tonnes" mean thousand tonnes (Kt) or 1000 tonnes (t).
5. Check if the unit is converted correctly to tonnes (t) in the final answer.

Give a binary score 'yes' or 'no' and show your reasoning. 'Yes' means that the LLM generation is consistent with the retrieved documents."""

user_prompt_template = (
    "# Retrieved Documents\n{documents}\n\n# LLM Generation\n{generation}"
)

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", user_prompt_template),
    ]
)

print(
    user_prompt_template.format(
        documents="[sample documents]", generation="[sample generation]"
    )
)

hallucination_grader = hallucination_prompt | structured_llm_grader


def hallucination_grader_assistant(documents, generation) -> GradeHallucinations:
    # Use openai assistant to grade the generation
    assistant = CLIENT.beta.assistants.retrieve("asst_pgJ1WkxCaHeCcShpdIMIFl6C")

    messages = [
        {
            "role": "user",
            "content": user_prompt_template.format(
                documents=documents, generation=generation
            ),
        },
    ]
    content = prompt_openai_assistant(assistant, messages)

    content_structured = GradeHallucinations.model_validate_json(content)

    return content_structured


# %%
### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    reasoning: str = Field(
        description="Reasoning if the answer addresses the question or not"
    )
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question.

Guidelines:
1. Check if the reasoning is enclosed in `<thinking>` XML tags.
2. Check if the final numerical answer is enclosed in `<output>` XML tags without any other XML tags, filler words, or explicit unit.
3. A default value 0 also counts as an invalid answer.

Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "# Question\n{question}\n\n# LLM Generation\n{generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

# %%
### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question:\n\n---\n{question}\n--- \n\nFormulate an improved question.",
        ),
    ]
)

template = "Here is the initial question:\n\n---\n{question}\n--- \n\nFormulate an improved question."
print(template.format(question="[sample question]"))

question_rewriter = re_write_prompt | llm | StrOutputParser()

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
    documents: List[str]

    # Added by Yixin
    retriever: Any
    hallucination_grade: str
    answer_grade: str
    # Store previous generation and feedback
    messages: Annotated[list, add_messages]


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
    answer_grade = state["answer_grade"]

    if hallucination_grade.lower() == "no" or answer_grade.lower() == "no":
        # RAG generation with previous generation and feedback
        previous_messages = state["messages"]
        generation = deep_extract_w_feedback(question, documents, previous_messages)
    else:
        # Initial RAG generation without hallucination grader feedback
        generation = deep_extract_wo_feedback(question, documents)

    return {
        "generation": generation,
        "messages": [
            {"role": "assistant", "content": generation},
        ],
    }


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

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
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

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
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

    score = hallucination_grader_assistant(documents, generation)
    grade = score.binary_score
    hallucination_grader_reasoning = score.reasoning
    return {
        "hallucination_grade": grade,
        "messages": [
            {
                "role": "assistant",
                "content": f"Hallucination grade: {grade}, Reasoning: {hallucination_grader_reasoning}",
            }
        ],
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

    score = answer_grader.invoke({"question": question, "generation": generation})
    grade = score.binary_score
    answer_grader_reasoning = score.reasoning

    return {
        "answer_grade": grade,
        "messages": [
            {
                "role": "assistant",
                "content": f"Answer grade: {grade}, Reasoning: {answer_grader_reasoning}",
            }
        ],
    }


### Edges


def retriever_router(state):
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
        return "generate"


def generate_router(state):
    # Use regex to check if the generation matches the pattern "<output>X</output>"
    # where X is a number (integer or decimal)
    generation: str = state["generation"]
    output_pattern = re.compile(r"<output>(\d+(?:\.\d+)?)</output>")
    match = output_pattern.search(generation)

    # Extract the numeric value from the match and convert it to a number
    answer = None
    if match:
        numeric_string = match.group(1)
        try:
            answer = float(numeric_string)
        except ValueError:
            logger.warning(
                f"---FAILED TO CONVERT MATCHED STRING TO NUMBER: {numeric_string}---"
            )

    if answer != 0:
        logger.info(
            "---DECISION: GENERATION CONTAINS NUMERIC OUTPUT, CHECK HALLUCINATION---"
        )
        return "check_hallucination"
    else:
        logger.info("---DECISION: GENERATION RETURNS DEFAULT VALUE 0. END---")
        return END


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
        return "regenerate"


def answer_quality_router(state):
    """
    Route based on answer quality check result
    """
    if state["answer_grade"] == "yes":
        logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        logger.info(
            f"Reasoning on why generation does not address question: {state['answer_grader_reasoning']}"
        )
        return "regenerate"


# %% [markdown]
# ## Build Graph
#


# %%
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
    retriever_router,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
# self_rag_graph_builder.add_edge("generate", "check_hallucination")
self_rag_graph_builder.add_conditional_edges(
    "generate",
    generate_router,
    {"check_hallucination": "check_hallucination", END: END},
)
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
        "regenerate": "generate",
    },
)

# Compile
self_rag_graph = self_rag_graph_builder.compile()

# Visualize the graph
viz_graph(self_rag_graph)

# %%
# Run
if __name__ == "__main__":
    question = QUESTION_TEMPLATE.format(
        field="total_mineral_resource_tonnage",
        # field="total_mineral_reserve_contained_metal",
        dtype="float",
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
        # description=TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION.replace(
        #     "<main_commodity>", "nickel"
        # ),
    )

    retriever = create_markdown_retriever(
        "data/processed/43-101/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md",
        collection_name="rag-chroma",
    )

    inputs = {
        "question": question,
        "generation": "N/A",
        "retriever": retriever,
        "hallucination_grade": "N/A",
        "hallucination_grader_reasoning": "N/A",
        "answer_grade": "N/A",
        "answer_grader_reasoning": "N/A",
    }

    value = self_rag_graph.invoke(inputs)

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
