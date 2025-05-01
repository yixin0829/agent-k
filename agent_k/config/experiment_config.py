"""
Experiment configurations for different agents and models.
"""

from enum import Enum

# --------------------------------------------------------------------------------------
# Batch Extraction Configs
# --------------------------------------------------------------------------------------
BATCH_EXTRACTION_MODEL = "o4-mini-2025-04-16"
BATCH_EXTRACTION_TEMPERATURE = 0.1


class BatchExtractionMethod(Enum):
    LONG_CONTEXT = "long_context"
    RAG_BASED = "rag_based"


BATCH_METHOD: BatchExtractionMethod = BatchExtractionMethod.RAG_BASED
MAX_NUM_RETRIEVED_DOCS = 5


# --------------------------------------------------------------------------------------
# PDF Extraction Configs
# --------------------------------------------------------------------------------------
PDF_EXTRACTION_SAMPLE_SIZE = None
PDF_EXTRACTION_METHOD = "F&S AGENTIC RAG"
PDF_EXTRACTION_EVAL_TYPE = "FULL"


# --------------------------------------------------------------------------------------
#  Slow Extraction Workflow Configs
# --------------------------------------------------------------------------------------
SLOW_EXTRACT_VALIDATION_MODEL = "gpt-4o-mini-2024-07-18"
SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.1
SLOW_EXTRACT_OPTIMIZER_MODEL = "gpt-4o-mini-2024-07-18"
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.1
RECURSION_LIMIT = 12
SLOW_WORKFLOW_RETRY_LIMIT = 5


# --------------------------------------------------------------------------------------
# Self RAG Configs
# --------------------------------------------------------------------------------------
SELF_RAG_GRADE_RETRIEVAL_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE = 0.1
SELF_RAG_GENERATION_MODEL = "o4-mini"
SELF_RAG_GENERATION_TEMPERATURE = 0.1
SELF_RAG_GRADE_HALLUCINATION_MODEL = "o4-mini"
SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE = 0.1
SELF_RAG_GRADE_ANSWER_MODEL = "o4-mini"
SELF_RAG_GRADE_ANSWER_TEMPERATURE = 0.1
SELF_RAG_QUESTION_REWRITER_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_QUESTION_REWRITER_TEMPERATURE = 0.5


# --------------------------------------------------------------------------------------
# Agentic RAG Configs
# --------------------------------------------------------------------------------------
NUM_RETRIEVED_DOCS = 5
GRADE_RETRIEVAL_MODEL = "gpt-4o-mini-2024-07-18"
GRADE_RETRIEVAL_TEMPERATURE = 0.1

# Code ReACT Agent Configs
PYTHON_AGENT_MODEL = "o4-mini-2025-04-16"
# Choose lower temperature to make generated code more deterministic (done an experiment with temp = 1 vs. 0.1)
PYTHON_AGENT_TEMPERATURE = 0.1

GRADE_HALLUCINATION_MODEL = "o4-mini-2025-04-16"
GRADE_HALLUCINATION_TEMPERATURE = 0.1
QUESTION_REWRITER_MODEL = "gpt-4o-mini-2024-07-18"
QUESTION_REWRITER_TEMPERATURE = 0.5
REACT_CODE_AGENT_RECURSION_LIMIT = 6
