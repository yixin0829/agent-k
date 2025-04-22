# %% [markdown]
# Changes compared to v4 implementation:
# - Optimizer implemented with litellm completion API in pdf_agent_fast_n_slow.py

# %%
import re
from collections import Counter
from operator import add
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
from langgraph.pregel import RetryPolicy
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agent_k.config.logger import logger
from agent_k.config.schemas import (
    TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
)
from agent_k.notebooks.react_code_agent import react_agent

load_dotenv()

# Retry for generate node (code interpreter easily fails)
RETRY_POLICY = RetryPolicy(max_attempts=3)

CLIENT = OpenAI()
NUM_RETRIEVED_DOCS = 5
PYTHON_AGENT_MODEL = "gpt-4o-mini"
GRADE_HALLUCINATION_MODEL = "gpt-4o-mini"
QUESTION_REWRITER_MODEL = "gpt-4o-mini"
QUESTION_REWRITER_TEMPERATURE = 0.5


def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


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

## Guidelines
1. Identify relevant facts in the context needed for answering the question.
2. Perform Aggregations: Use the code interpreter tool for operations like summation, multiplication, or other numerical operations.
3. Structure the Response Correctly: Format your final output with XML tags as follows:
    - Reasoning: Explain your retrieval or computation process within `<thinking>` tags.
    - Code: Show the executed code within `<code>`  tags.
    - Final Answer: Provide the final response with unit within `<output>` tags (e.g. `<output>1000 tonnes</output>`).

## Key Constraints:
- No Hallucination: If the required information is unavailable, return the default value specified in the JSON schema in the `<output>` tag."""

GENERATION_USER_PROMPT_W_FEEDBACK = """You are an assistant for question-answering tasks. Use the following retrieved context and previous feedback (if any) to answer the question. If you don't know the answer, just return the default value of the field in the question.

## Context
{context}

## Previous Messages
{previous_messages}

## Question
{question}

---
Now take a deep breath and answer the question step by step while considering the previous feedback."""


def deep_extract_w_feedback(question, context, previous_messages) -> str:
    # Convert previous messages to a list of strings from a list of dicts
    previous_messages_str = [str(msg) for msg in previous_messages]
    result = react_agent(
        DEEP_EXTRACT_SYSTEM_PROMPT,
        GENERATION_USER_PROMPT_W_FEEDBACK.format(
            question=question,
            context=context,
            previous_messages="\n".join(previous_messages_str),
        ),
    )
    content = result["messages"][-1].content

    return content


# %%
### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    feedback: str = Field(
        description="Reasoning whether the raw facts in the answer are aligned with the retrieved documents + feedback on how to improve"
    )
    binary_score: str = Field(
        description="Raw facts in the answer are aligned with the retrieved documents, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model=GRADE_HALLUCINATION_MODEL)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt (Update in OpenAI Assistant)
GRADE_HALLUCINATION_SYSTEM_PROMPT = """You are a hallucination agent validating a LLM's generation against the retrieved documents. Focus on the calculation logic and unit conversions.

Guidelines:
1. Total mineral resource tonnage should be the sum of one or more of inferred, indicated, and measured mineral resources. If not, a default value of 0 should be returned.
2. Total mineral reserve tonnage should be the sum of one or more of proven and probable mineral reserves. If not, a default value of 0 should be returned.
3. The tonnage or grade unit used in the LLM generation should be consistent with the unit used in the retrieved documents. For example, "Tonnes 000", "Tonnes (000)", or "(000) Tonnes" mean thousand tonnes (Kt) or 1000 tonnes (t).
4. The unit of grade should be correctly converted to decimal before used in the calculation in the code.
5. The final answer enclosed in `<output>` tags should be converted correctly to tonnes (t).

Show your feedback and give a binary score 'yes' or 'no' and reasoning. 'Yes' means that the LLM generation is consistent with the retrieved documents and no hallucination."""

GRADE_HALLUCINATION_USER_PROMPT = """# Retrieved Documents
{documents}

# LLM Generation
{generation}

---
Now take a deep breath and grade the LLM generation"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADE_HALLUCINATION_SYSTEM_PROMPT),
        ("human", GRADE_HALLUCINATION_USER_PROMPT),
    ]
)

print(
    "\n"
    + GRADE_HALLUCINATION_USER_PROMPT.format(
        documents="[sample documents]", generation="[sample generation]"
    )
)

hallucination_grader = hallucination_prompt | structured_llm_grader

# %%
### Question Re-writer

# LLM
llm = ChatOpenAI(
    model=QUESTION_REWRITER_MODEL, temperature=QUESTION_REWRITER_TEMPERATURE
)

# Prompt
QUESTION_REWRITER_SYSTEM_PROMPT = """You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

QUESTION_REWRITER_USER_PROMPT = """Here is the initial question:\n\n---\n{question}\n--- \n\nFormulate an improved question."""

re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUESTION_REWRITER_SYSTEM_PROMPT),
        ("human", QUESTION_REWRITER_USER_PROMPT),
    ]
)

print(QUESTION_REWRITER_USER_PROMPT.format(question="[sample question]"))

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
    messages: Annotated[list[str], add]  # store prev generation + feedback
    previous_answers: Annotated[list[str], add]  # store prev ans for self consistency


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


def get_mode_or_last(lst):
    count = Counter(lst)
    max_count = max(count.values())

    # Check if the mode is unique
    modes = [item for item, cnt in count.items() if cnt == max_count]

    if max_count > 1 and len(modes) == 1:
        return modes[0]
    else:
        return lst[-1]


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

    if len(state["previous_answers"]) >= 3:
        # Self consistency if detected looping
        mode_answer = get_mode_or_last(state["previous_answers"])
        generation = f"<reasoning>Detect looping. Use self consistency to choose the most popular answer from previous generations.</reasoning><output>{mode_answer}</output>"
    else:
        previous_messages = state["messages"]
        generation = deep_extract_w_feedback(question, documents, previous_messages)

    try:
        parsed_output = generation.split("<output>")[1].split("</output>")[0].strip()
        parsed_output = re.sub(r"[^0-9.]", "", parsed_output)
    except IndexError:
        logger.exception(f"Error parsing <output> XML tags for content: {generation}")

    return {
        "generation": generation,
        "previous_answers": [parsed_output],
        "messages": [
            {"role": "Assistant", "content": generation},
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
    previous_messages = state["messages"]

    score: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
            "previous_messages": previous_messages,
        }
    )
    grade = score.binary_score
    hallucination_grader_feedback = score.feedback
    return {
        "hallucination_grade": grade,
        "messages": [
            {
                "role": "Hallucination Grader",
                "content": f"Passed hallucination check: {grade}\nFeedback: {hallucination_grader_feedback}",
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


def hallucination_router(state):
    """
    Route based on hallucination check result
    """
    if state["hallucination_grade"].lower() == "yes":
        logger.info(
            "---DECISION: GENERATION IS GROUNDED IN DOCUMENTS (NO HALLUCINATION)---"
        )
        return END
    elif len(state["previous_answers"]) >= 3:
        logger.info("---DECISION: LOOPING DETECT. USE SELF CONSISTENCY VALUE. END---")
        return END
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
self_rag_graph_builder.add_node("retrieve", retrieve)
self_rag_graph_builder.add_node("grade_documents", grade_documents)
self_rag_graph_builder.add_node("generate", generate, retry=RETRY_POLICY)
self_rag_graph_builder.add_node("transform_query", transform_query)
self_rag_graph_builder.add_node("check_hallucination", check_hallucination)

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
self_rag_graph_builder.add_edge("generate", "check_hallucination")
self_rag_graph_builder.add_edge("transform_query", "retrieve")
self_rag_graph_builder.add_conditional_edges(
    "check_hallucination",
    hallucination_router,
    {
        END: END,
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
        "data/processed/43-101-header-corrected/0200a1c6d2cfafeb485d815d95966961d4c119e8662b8babec74e05b59ba4759d2.md",
        collection_name="rag-chroma",
    )

    graph_inputs = {
        "question": question,
        "generation": "N/A",
        "retriever": retriever,
        "hallucination_grade": "N/A",
    }

    value = self_rag_graph.invoke(graph_inputs, config={"recursion_limit": 12})

    # Final generation
    logger.info("---FINAL GENERATION---")
    logger.info(value["generation"])
