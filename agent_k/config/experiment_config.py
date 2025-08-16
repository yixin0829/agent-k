"""
Experiment configurations for different agents and models.
"""

from enum import Enum

# --------------------------------------------------------------------------------------
# Batch Extraction Configs
# --------------------------------------------------------------------------------------
# BATCH_EXTRACTION_MODEL = "openai/gpt-oss-20b"
# BATCH_EXTRACTION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# BATCH_EXTRACTION_MODEL = "deepseek-ai/DeepSeek-R1"
BATCH_EXTRACTION_MODEL = "Qwen/Qwen3-30B-A3B"
# BATCH_EXTRACTION_MODEL = "gpt-4.1-2025-04-14"
BATCH_EXTRACTION_TEMPERATURE = 0.1


class BatchExtractionMethod(Enum):
    LONG_CONTEXT = "long_context"  # Long context batch extraction
    RAG_BASED = "rag_based"  # RAG-based batch extraction


BATCH_METHOD: BatchExtractionMethod = BatchExtractionMethod.RAG_BASED
MAX_NUM_RETRIEVED_DOCS = 5


# --------------------------------------------------------------------------------------
# PDF Extraction Eval Configs
# --------------------------------------------------------------------------------------
class ExtractionMethod(Enum):
    FS_SELF_RAG = "F&S SELF RAG"
    FS_AGENTIC_RAG = "F&S AGENTIC RAG"


PDF_EXTRACTION_METHOD = ExtractionMethod.FS_SELF_RAG

PDF_EXTRACTION_SAMPLE_SIZE = None
PDF_EXTRACTION_EVAL_TYPE = "FULL"


# --------------------------------------------------------------------------------------
# Self RAG Configs
# --------------------------------------------------------------------------------------
# SELF_RAG_MODEL = "gpt-4o-mini-2024-07-18"
# SELF_RAG_MODEL = "gpt-oss-20b"
# SELF_RAG_MODEL = "Llama-3.3-70B-Instruct"
# SELF_RAG_MODEL = "gemini-2.5-flash"
# SELF_RAG_MODEL = "gpt-4.1-2025-04-14"
SELF_RAG_MODEL = "o4-mini-2025-04-16"
# SELF_RAG_MODEL = "deepseek-ai/DeepSeek-R1"
SELF_RAG_TEMPERATURE = 0.1

SELF_RAG_GRADE_RETRIEVAL_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GENERATION_MODEL = SELF_RAG_MODEL
SELF_RAG_GENERATION_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GRADE_HALLUCINATION_MODEL = SELF_RAG_MODEL
SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GRADE_ANSWER_MODEL = SELF_RAG_MODEL
SELF_RAG_GRADE_ANSWER_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_QUESTION_REWRITER_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_QUESTION_REWRITER_TEMPERATURE = 1


# --------------------------------------------------------------------------------------
# Our method configs (Agentic RAG)
# --------------------------------------------------------------------------------------
# OUR_METHOD_MODEL = "gpt-4.1-2025-04-14"
# OUR_METHOD_MODEL = "gpt-4o-mini-2024-07-18"
# OUR_METHOD_MODEL = "gpt-3.5-turbo-0125"
# OUR_METHOD_MODEL = "o4-mini-2025-04-16"
# OUR_METHOD_MODEL = "gpt-4-0613"
# OUR_METHOD_MODEL = "openai/gpt-oss-20b"
# OUR_METHOD_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# OUR_METHOD_MODEL = "deepseek-ai/DeepSeek-R1"
OUR_METHOD_MODEL = "Qwen/Qwen3-30B-A3B"
OUR_METHOD_TEMPERATURE = 0.1
MAX_REFLECTION_ITERATIONS = 5


class RetrievalMethod(Enum):
    RAG = "rag"
    LONG_CONTEXT = "long_context"


RETRIEVAL_METHOD = RetrievalMethod.RAG
RETRIEVAL_MODEL = "gpt-4.1-mini-2025-04-14"
RETRIEVAL_TEMPERATURE = 0.1

# Slow Extraction global workflow configs
SLOW_EXTRACT_VALIDATION_MODEL = OUR_METHOD_MODEL
SLOW_EXTRACT_VALIDATION_TEMPERATURE = OUR_METHOD_TEMPERATURE
SLOW_EXTRACT_OPTIMIZER_MODEL = OUR_METHOD_MODEL
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = OUR_METHOD_TEMPERATURE
RECURSION_LIMIT = 25
SLOW_WORKFLOW_RETRY_LIMIT = 3


# Main Slow Extraction Workflow Configs
NUM_RETRIEVED_DOCS = 5
GRADE_RETRIEVAL_MODEL = OUR_METHOD_MODEL
GRADE_RETRIEVAL_TEMPERATURE = OUR_METHOD_TEMPERATURE

# Code ReACT Agent Configs
PYTHON_AGENT_MODEL = OUR_METHOD_MODEL
PYTHON_AGENT_TEMPERATURE = OUR_METHOD_TEMPERATURE

GRADE_HALLUCINATION_MODEL = OUR_METHOD_MODEL
GRADE_HALLUCINATION_TEMPERATURE = OUR_METHOD_TEMPERATURE
QUESTION_REWRITER_MODEL = OUR_METHOD_MODEL
QUESTION_REWRITER_TEMPERATURE = 0.5
REACT_CODE_AGENT_RECURSION_LIMIT = 6
