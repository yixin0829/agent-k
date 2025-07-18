"""
Experiment configurations for different agents and models.
"""

from enum import Enum

# --------------------------------------------------------------------------------------
# Batch Extraction Configs
# --------------------------------------------------------------------------------------
BATCH_EXTRACTION_MODEL = "gpt-4o-mini"
BATCH_EXTRACTION_TEMPERATURE = 0.1


class BatchExtractionMethod(Enum):
    LONG_CONTEXT = "long_context"  # Long context batch extraction
    RAG_BASED = "rag_based"  # RAG-based batch extraction


BATCH_METHOD: BatchExtractionMethod = BatchExtractionMethod.RAG_BASED
MAX_NUM_RETRIEVED_DOCS = 5


# --------------------------------------------------------------------------------------
# PDF Extraction Eval Configs
# --------------------------------------------------------------------------------------
PDF_EXTRACTION_SAMPLE_SIZE = None


class ExtractionMethod(Enum):
    FS_AGENTIC_RAG = "F&S AGENTIC RAG"  # v6
    FS_SELF_RAG = "F&S SELF RAG"
    FS_AGENTIC_RAG_V7 = "F&S AGENTIC RAG V7"


PDF_EXTRACTION_METHOD = ExtractionMethod.FS_AGENTIC_RAG_V7
PDF_EXTRACTION_EVAL_TYPE = "FULL"


# --------------------------------------------------------------------------------------
# Self RAG Configs
# --------------------------------------------------------------------------------------
SELF_RAG_GRADE_RETRIEVAL_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE = 0.1
SELF_RAG_GENERATION_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_GENERATION_TEMPERATURE = 0.1
SELF_RAG_GRADE_HALLUCINATION_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE = 0.1
SELF_RAG_GRADE_ANSWER_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_GRADE_ANSWER_TEMPERATURE = 0.1
SELF_RAG_QUESTION_REWRITER_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_QUESTION_REWRITER_TEMPERATURE = 0.5


# --------------------------------------------------------------------------------------
# Agentic RAG Configs
# --------------------------------------------------------------------------------------

#  Slow Extraction global workflow configs
SLOW_EXTRACT_VALIDATION_MODEL = "gpt-4.1-2025-04-14"
SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.1
SLOW_EXTRACT_OPTIMIZER_MODEL = "gpt-4.1-2025-04-14"
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.1
RECURSION_LIMIT = 25
SLOW_WORKFLOW_RETRY_LIMIT = 5


# Main Slow Extraction Workflow Configs
NUM_RETRIEVED_DOCS = 5
GRADE_RETRIEVAL_MODEL = "gpt-4.1-2025-04-14"
GRADE_RETRIEVAL_TEMPERATURE = 0.1

# Code ReACT Agent Configs
PYTHON_AGENT_MODEL = "gpt-4.1-2025-04-14"
# Choose lower temperature to make generated code more deterministic (done an experiment with temp = 1 vs. 0.1)
PYTHON_AGENT_TEMPERATURE = 0.1

GRADE_HALLUCINATION_MODEL = "gpt-4.1-2025-04-14"
GRADE_HALLUCINATION_TEMPERATURE = 0.1
QUESTION_REWRITER_MODEL = "gpt-4.1-2025-04-14"
QUESTION_REWRITER_TEMPERATURE = 0.5
REACT_CODE_AGENT_RECURSION_LIMIT = 6
