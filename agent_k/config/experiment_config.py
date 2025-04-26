"""
Experiment configurations for different agents and models.
"""

################################# Code Agent React Configs #################################
PYTHON_AGENT_MODEL = "o3-mini"
# Choose lower temperature to make generated code more deterministic (done an experiment with temp = 1 vs. 0.1)
PYTHON_AGENT_TEMPERATURE = 0.1
################################# Code Agent React Configs #################################

################################# Self RAG Configs #################################
NUM_RETRIEVED_DOCS = 5
GRADE_RETRIEVAL_MODEL = "gpt-4o-mini"
GRADE_RETRIEVAL_TEMPERATURE = 0.1
GRADE_HALLUCINATION_MODEL = "o3-mini"
GRADE_HALLUCINATION_TEMPERATURE = 0.1
QUESTION_REWRITER_MODEL = "gpt-4o-mini"
QUESTION_REWRITER_TEMPERATURE = 0.5
REACT_CODE_AGENT_RECURSION_LIMIT = 6
################################# Self RAG Configs #################################

################################# PDF Agent Configs #################################
SLOW_EXTRACT_VALIDATION_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.1
SLOW_EXTRACT_OPTIMIZER_MODEL = "gpt-4o-mini"
SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.1
RECURSION_LIMIT = 12  # Self-RAG recursion limit
SELF_RAG_RETRY_LIMIT = 5
################################# PDF Agent Configs #################################

################################# PDF Extraction Configs #################################
PDF_EXTRACTION_SAMPLE_SIZE = None
PDF_EXTRACTION_METHOD = "DPE MAP_REDUCE SELF RAG"
PDF_EXTRACTION_EVAL_TYPE = "TEST"
################################# PDF Extraction Configs #################################
