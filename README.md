# Agent-K: Zero-Shot Schema-Guided Complex Numerical Property Extraction from Documents 🤖

Agent-K is a novel framework for extracting mathematically complex properties from unstructured documents using large language models (LLMs). Unlike standard batch-extraction approaches via structured output prompting that often fail on multi-step numerical reasoning, Agent-K breaks down the task into three stages: extracting intermediate facts, generating and executing Python code via a ReAct agent with self-reflection, and validating outputs with inter-property constraints. On a benchmark built from real-world NI 43-101 mineral reports, Agent-K significantly reduces error (sMAPE -22.1%) and improves accuracy (pass@1 +15.8%) over baselines, and further generalizes to the financial domain (FinQA), improving pass@1 accuracy by up to 11.1% in a zero-shot setting. Our empirical results show that Agent-K can be applied as a robust framework for structured data extraction that does not rely on the availability of structured output APIs.

![Architecture of the framework](assets/minmod-paper-figures%20-%20main_v2.png)

## News
- **2025-08:** The first version of Agent-K is released!

## Prerequisites

- Python 3.12 or higher
- Docker (for running code interpreter tool)
- `uv` package manager (recommended for Python dependency management)
- Environment variables:
  - `OPENAI_API_KEY`: OpenAI API token
  - `HF_TOKEN`: HuggingFace token

## Setup Instructions

1. Clone the repository
2. Install `uv` package manager
3. Create and activate a virtual environment
4. Install dependencies using either `uv sync` or `pip install -r requirements.txt`
5. Build Docker image for code interpreter using `make build`
6. Run code interpreter Docker container using `make run`
7. Configure API tokens by renaming the `.env.example` file to `.env` and adding your API tokens
  1. `OPENAI_API_KEY`: needed for running OpenAI models (e.g. `gpt-4o-mini-2024-07-18`)
  2. `HF_TOKEN`: needed for running Open-Source models (e.g. `Qwen/Qwen3-30B-A3B`)

## Running Experiments

### FinQA Dataset

The FinQA experiments test Agent-K on financial question answering datasets.

1. **Run Agent-K predictions on FinQA test set:**
```bash
uv run python src/experiments/fin_qa/fin_qa_pred.py
```

2. **Evaluate the results:**
```bash
uv run python src/experiments/fin_qa/fin_qa_eval.py
```

### NI 43-101 Dataset

#### Baseline 1: Batch Extraction (Long Context, RAG-based)

Running batch extraction experiments separately for long context and RAG-based:
```bash
uv run python src/experiments/batch_extraction.py
```

Configure batch extraction settings in `src/config/experiment_config.py`:
- `BATCH_EXTRACTION_MODEL`: Model to use
- `MAX_NUM_RETRIEVED_DOCS`: Number of documents to retrieve in RAG-based batch extraction
- `BATCH_METHOD`: Choose between `LONG_CONTEXT` or `RAG_BASED`
- `BATCH_EXTRACTION_SAMPLE_SIZE`: Set to `None` for full dataset or specify number of samples

#### Baseline 2: TAT-LLM

To run TAT-LLM experiments:

1. **Configure the extraction method in `src/config/experiment_config.py`:**
```python
PDF_EXTRACTION_METHOD = ExtractionMethod.TAT_LLM
```

2. **Configure sample size and evaluation set:**
```python
PDF_EXTRACTION_SAMPLE_SIZE = None  # None for all 50 PDFs, or specify number
PDF_EXTRACTION_EVAL_TYPE = "FULL"
TAT_LLM_MODEL = "gpt-4o-mini-2024-07-18"
TAT_LLM_TEMPERATURE = 0.2
```

3. **Run the TAT-LLM extraction:**
```bash
uv run python src/experiments/multi_method_extraction.py
```

#### Baseline 3: Self-RAG

1. **Configure the extraction method in `src/config/experiment_config.py`:**
```python
PDF_EXTRACTION_METHOD = ExtractionMethod.SELF_RAG
```

2. **Configure sample size and evaluation set:**
```python
PDF_EXTRACTION_SAMPLE_SIZE = None  # None for all 50 PDFs, or specify number
PDF_EXTRACTION_EVAL_TYPE = "FULL"
SELF_RAG_MODEL = "gpt-4o-mini-2024-07-18"
SELF_RAG_TEMPERATURE = 0.2
```

3. **Run the Self-RAG extraction:**
```bash
uv run python src/experiments/multi_method_extraction.py
```

#### Our Method: Agent-K

1. **Configure the extraction method in `src/config/experiment_config.py`:**
```python
PDF_EXTRACTION_METHOD = ExtractionMethod.AGENT_K
```

2. **Configure sample size and evaluation set:**
```python
PDF_EXTRACTION_SAMPLE_SIZE = None  # None for all 50 PDFs, or specify number
PDF_EXTRACTION_EVAL_TYPE = "FULL"
AGENT_K_MODEL = "gpt-4o-mini-2024-07-18"
AGENT_K_TEMPERATURE = 0.2
MAX_REFLECTION_ITERATIONS = 5
```

3. **Run the Agent-K extraction:**
```bash
uv run python src/experiments/multi_method_extraction.py
```

### Evaluation

After running experiments, results will be stored in `data/experiment/` under the corresponding method directory. To evaluate the results:

1. **Copy experimental result path into `eval.py`:**
  - Find the result files from `data/experiments/` (e.g. `data/experiments/agent_k/agent-k_2025-08-29_11-55-40.csv`)
  - Copy the result path in `agent_extractions` list in `src/eval.py`. You can also add multiple result paths to the list to calculate pass@k scores.

2. **Run evaluation:**
```bash
uv run python src/eval.py
```

3. **The evaluation will output two files:**
  - `pdf_extraction_metrics_<timestamp>.csv`: aggregated metrics (absolute mean error, R-squared, SMAPE, pass@1) for each complex numerical property + the average of all metrics.
  - `df_merged_<timestamp>.csv`: Mineral report level metrics.

### Ablation Studies

To understand the contribution of different components:

```bash
uv run python src/experiments/ablation_tests.py
```

This will test various configurations with components disabled to measure their impact. The output will be saved in `data/experiments/ablation_tests/` under the corresponding variant directory.

### Hyperparameter Tuning

The parameter search script tests different combinations of key parameters to find the configuration that yields the best extraction performance. It evaluates three main parameters:

1. **Max Reflection Iterations** - Controls the maximum number of iterations for self-reflection before falling back to self-consistency.
2. **Temperature** - LLM temperature.
3. **Number of Retrieved Documents** - Number of documents to retrieve for context during experiment execution for each complex numerical property.

To find optimal hyperparameters for your specific use case:

1. **Configure search parameters in `src/experiments/parameter_search/parameter_search_config.yaml`**
  - **Model**: Which model to use (default: gpt-4o-mini)
  - **Sample Size**: Number of PDFs to process per experiment (default: 5)
  - **Evaluation Set**: DEV, TEST, or FULL dataset
  - **Parameter Values**: Specific values to test for each parameter

2. **Run parameter search:**
```bash
uv run python src/experiments/parameter_search/parameter_search.py
```

3. **Visualize results:** You can also specify the weights for the composite metric. The composite metric is calculated as: $\alpha \times (1-sMAPE) + \beta \times Pass@1$ where $\alpha$ and $\beta$ are configurable weights (default: 0.5 each).
```bash
# Prioritize SMAPE over pass@1 (α=0.8, β=0.2)
uv run python src/experiments/parameter_search/visualize_parameter_search.py --alpha 0.8 --beta 0.2
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yixin0829/agent-k/blob/main/LICENSE) file for details.
