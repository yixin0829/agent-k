"""
Experiment configurations for different agents and models.
"""

from enum import Enum

# --------------------------------------------------------------------------------------
# Batch Extraction Configs
# --------------------------------------------------------------------------------------
BATCH_EXTRACTION_MODEL = "o4-mini"
BATCH_EXTRACTION_TEMPERATURE = 0.1


class BatchExtractionMethod(Enum):
    LONG_CONTEXT = "long_context"
    RAG_BASED = "rag_based"


BATCH_METHOD: BatchExtractionMethod = BatchExtractionMethod.RAG_BASED
MAX_NUM_RETRIEVED_DOCS = 5

# --------------------------------------------------------------------------------------
# PDF Extraction Configs
# --------------------------------------------------------------------------------------
PDF_EXTRACTION_SAMPLE_SIZE = 1
PDF_EXTRACTION_METHOD = "F&S AGENTIC RAG"
PDF_EXTRACTION_EVAL_TYPE = "TEST"

# --------------------------------------------------------------------------------------
#  Slow Agent Configs
# --------------------------------------------------------------------------------------
SLOW_EXTRACT_VALIDATION_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.1
SLOW_EXTRACT_OPTIMIZER_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.1
RECURSION_LIMIT = 12
SLOW_WORKFLOW_RETRY_LIMIT = 5

# --------------------------------------------------------------------------------------
# Self RAG Configs
# --------------------------------------------------------------------------------------
NUM_RETRIEVED_DOCS = 5
GRADE_RETRIEVAL_MODEL = "gpt-4o-mini"
GRADE_RETRIEVAL_TEMPERATURE = 0.1
GRADE_HALLUCINATION_MODEL = "o4-mini"
GRADE_HALLUCINATION_TEMPERATURE = 0.1
QUESTION_REWRITER_MODEL = "gpt-4o-mini"
QUESTION_REWRITER_TEMPERATURE = 0.5
REACT_CODE_AGENT_RECURSION_LIMIT = 6

# --------------------------------------------------------------------------------------
# Code Agent React Configs
# --------------------------------------------------------------------------------------
PYTHON_AGENT_MODEL = "o4-mini"
# Choose lower temperature to make generated code more deterministic (done an experiment with temp = 1 vs. 0.1)
PYTHON_AGENT_TEMPERATURE = 0.1
