"""
Experiment configurations for different methods.
"""

from enum import Enum

# --------------------------------------------------------------------------------------
# Batch Extraction Configs (Long Context, RAG-based)
# --------------------------------------------------------------------------------------
# BATCH_EXTRACTION_MODEL = "openai/gpt-oss-20b"
BATCH_EXTRACTION_MODEL = "meta-llama/Llama-3.3-70B-Instruct:cerebras"
# BATCH_EXTRACTION_MODEL = "deepseek-ai/DeepSeek-R1"
# BATCH_EXTRACTION_MODEL = "Qwen/Qwen3-30B-A3B"
# BATCH_EXTRACTION_MODEL = "gpt-4.1-2025-04-14"
BATCH_EXTRACTION_TEMPERATURE = 0.2


class BatchExtractionMethod(Enum):
    LONG_CONTEXT = "long_context"  # Long context batch extraction
    RAG_BASED = "rag_based"  # RAG-based batch extraction


BATCH_METHOD: BatchExtractionMethod = BatchExtractionMethod.LONG_CONTEXT
MAX_NUM_RETRIEVED_DOCS = 2

# None = full dataset. Otherwise, use first N samples
BATCH_EXTRACTION_SAMPLE_SIZE = None


# --------------------------------------------------------------------------------------
# PDF Extraction Eval Configs (TAT-LLM, SELF-RAG, AGENT-K)
# --------------------------------------------------------------------------------------
class ExtractionMethod(Enum):
    TAT_LLM = "TAT-LLM"
    SELF_RAG = "SELF-RAG"
    AGENT_K = "AGENT-K"


PDF_EXTRACTION_METHOD = ExtractionMethod.AGENT_K

# Number of PDF files to extract from. None = all 50 PDF files, K = first K PDF files
PDF_EXTRACTION_SAMPLE_SIZE = 3
PDF_EXTRACTION_EVAL_TYPE = "FULL"

# 1. TAT-LLM Configs

# TAT_LLM_MODEL = "gpt-4o-mini-2024-07-18"
# TAT_LLM_MODEL = "o4-mini"
# TAT_LLM_MODEL = "gpt-oss-20b"
# TAT_LLM_MODEL = "Llama-3.3-70B-Instruct"
# TAT_LLM_MODEL = "deepseek-ai/DeepSeek-R1"
TAT_LLM_MODEL = "gpt-4o-mini-2024-07-18"  # Default model
TAT_LLM_TEMPERATURE = 0.2

# 2. Self RAG Configs

# SELF_RAG_MODEL = "gpt-oss-20b"
# SELF_RAG_MODEL = "Llama-3.3-70B-Instruct"
# SELF_RAG_MODEL = "gpt-4.1-2025-04-14"
# SELF_RAG_MODEL = "o4-mini-2025-04-16"
# SELF_RAG_MODEL = "deepseek-ai/DeepSeek-R1"
# SELF_RAG_MODEL = "Qwen/Qwen3-30B-A3B:novita"
SELF_RAG_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_TEMPERATURE = 0.2

SELF_RAG_GRADE_RETRIEVAL_MODEL = SELF_RAG_MODEL
SELF_RAG_GRADE_RETRIEVAL_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GENERATION_MODEL = SELF_RAG_MODEL
SELF_RAG_GENERATION_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GRADE_HALLUCINATION_MODEL = SELF_RAG_MODEL
SELF_RAG_GRADE_HALLUCINATION_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_GRADE_ANSWER_MODEL = SELF_RAG_MODEL
SELF_RAG_GRADE_ANSWER_TEMPERATURE = SELF_RAG_TEMPERATURE
SELF_RAG_QUESTION_REWRITER_MODEL = SELF_RAG_MODEL
SELF_RAG_QUESTION_REWRITER_TEMPERATURE = 1


# 3. Agent-K configs

# AGENT_K_MODEL = "gpt-4.1-2025-04-14"
# AGENT_K_MODEL = "gpt-3.5-turbo-0125"
# AGENT_K_MODEL = "o4-mini-2025-04-16"
# AGENT_K_MODEL = "gpt-4-0613"
# AGENT_K_MODEL = "openai/gpt-oss-20b"
# AGENT_K_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# AGENT_K_MODEL = "deepseek-ai/DeepSeek-R1"
# AGENT_K_MODEL = "Qwen/Qwen3-30B-A3B:novita"
AGENT_K_MODEL = "gpt-4o-mini-2024-07-18"
OUR_METHOD_TEMPERATURE = 0.2


class RetrievalMethod(Enum):
    RAG = "rag"
    LONG_CONTEXT = "long_context"


RETRIEVAL_METHOD = RetrievalMethod.RAG
if RETRIEVAL_METHOD == RetrievalMethod.LONG_CONTEXT:
    RETRIEVAL_MODEL = "gpt-4.1-mini-2025-04-14"
    RETRIEVAL_TEMPERATURE = 0.1

# Global Agent-K Extraction workflow configs
SLOW_EXTRACT_VALIDATION_MODEL = AGENT_K_MODEL
SLOW_EXTRACT_VALIDATION_TEMPERATURE = OUR_METHOD_TEMPERATURE
SLOW_EXTRACT_OPTIMIZER_MODEL = AGENT_K_MODEL
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = OUR_METHOD_TEMPERATURE

# Unified retry configuration for all extraction methods
EXTRACTION_MAX_RETRIES = 3

NUM_RETRIEVED_DOCS = 5

## ReACT Agent Configs
PYTHON_AGENT_MODEL = AGENT_K_MODEL
PYTHON_AGENT_TEMPERATURE = OUR_METHOD_TEMPERATURE
GRADE_HALLUCINATION_MODEL = AGENT_K_MODEL
GRADE_HALLUCINATION_TEMPERATURE = OUR_METHOD_TEMPERATURE
MAX_REFLECTION_ITERATIONS = 5
