# Parameter Search for Agentic RAG Optimization

This module provides a systematic way to search for optimal parameters for the Agentic RAG extraction method used in the mineral extraction pipeline.

## Overview

The parameter search script tests different combinations of key parameters to find the configuration that yields the best extraction performance. It evaluates three main parameters:

1. **Max Reflection Iterations** - Controls the maximum number of iterations for hallucination checking
2. **Temperature** - Controls the randomness/creativity of model outputs
3. **Number of Retrieved Documents** - Determines how many documents to retrieve for context

## Files

All parameter search files are now located in `paper/experiments/parameter_search/`:

- `parameter_search.py` - Main parameter search script
- `parameter_search_config.yaml` - Configuration file for search parameters
- `run_parameter_search.sh` - Shell script to run the parameter search
- `visualize_parameter_search.py` - Script for visualizing parameter search results
- `PARAMETER_SEARCH_README.md` - This documentation file

## Usage

### Quick Start

Run the parameter search script:

```bash
uv run python experiments/parameter_search/parameter_search.py
```

### Visualization

After running parameter search experiments, visualize results with:

```bash
# Use default equal weights (α=0.5, β=0.5)
uv run python experiments/parameter_search/visualize_parameter_search.py

# Prioritize SMAPE over pass@1 (α=0.7, β=0.3)
uv run python experiments/parameter_search/visualize_parameter_search.py --alpha 0.7 --beta 0.3

# Prioritize pass@1 over SMAPE (α=0.3, β=0.7)
uv run python experiments/parameter_search/visualize_parameter_search.py --alpha 0.3 --beta 0.7
```

The visualization creates a 4-column plot showing:
1. **SMAPE vs Parameter Value** - Lower SMAPE is better
2. **Execution Time vs Parameter Value** - Shorter time is better
3. **Pass@1 vs Parameter Value** - Higher Pass@1 is better
4. **Weighted Composite Metric vs Parameter Value** - Higher composite score is better

Red stars mark the optimal parameter values for each metric, with the composite metric helping balance SMAPE and pass@1 performance according to your priorities.

### Configuration

Edit `experiments/parameter_search/parameter_search_config.yaml` to adjust:

- **Model**: Which model to use (default: gpt-4o-mini)
- **Sample Size**: Number of PDFs to process per experiment (default: 5)
- **Evaluation Set**: DEV, TEST, or FULL dataset
- **Parameter Values**: Specific values to test for each parameter

Example configuration:

```yaml
model:
  name: "gpt-4o-mini-2024-07-18"

dataset:
  sample_size: 5  # Set to null to use all PDFs
  eval_type: "DEV"

parameters:
  max_reflection_iterations:
    values: [2, 3, 4, 5, 6, 7]
  temperature:
    values: [0.1, 0.25, 0.5, 0.75, 1.0]
  num_retrieved_docs:
    values: [1, 2, 3, 4, 5]
```

## Output

Results are saved to `paper/data/parameter_search/` with the following structure:

```
paper/data/parameter_search/
├── max_reflection_iterations/
│   ├── MAX_REFLECTION_ITERATIONS_2/
│   │   ├── extraction_results_*.csv
│   │   └── experiment_config_*.yaml
│   ├── MAX_REFLECTION_ITERATIONS_3/
│   └── parameter_search_results.csv
├── temperature/
│   ├── temperature_0.1/
│   ├── temperature_0.25/
│   └── parameter_search_results.csv
├── num_retrieved_docs/
│   ├── NUM_RETRIEVED_DOCS_1/
│   ├── NUM_RETRIEVED_DOCS_2/
│   └── parameter_search_results.csv
├── plots/
│   └── parameter_search_results.png
├── all_parameter_search_results.csv
└── parameter_summary_table.csv
```

### Output Files

- **extraction_results_*.csv** - Raw extraction results for each experiment
- **experiment_config_*.yaml** - Configuration used for each experiment
- **parameter_search_results.csv** - Summary results for each parameter type
- **all_parameter_search_results.csv** - Consolidated results from all experiments
- **parameter_search_results.png** - Visualization plots showing performance across all metrics
- **parameter_summary_table.csv** - Summary table with best parameter values for each metric

## Evaluation Metrics

The script evaluates performance using:

- **SMAPE** (Symmetric Mean Absolute Percentage Error) - Lower is better
- **Absolute Mean Error** - Lower is better
- **Pass@1** - Percentage of pass@1es (higher is better)
- **Execution Time** - Time taken per PDF
- **Weighted Composite Metric** - Combines (1-SMAPE) and Pass@1 with configurable weights (higher is better)

The composite metric is calculated as: `α × (1-SMAPE) + β × Pass@1` where α and β are configurable weights (default: 0.5 each).

The best parameter value is selected based on the lowest SMAPE score when evaluation data is available, otherwise by execution time. The visualization script also identifies optimal parameters based on the weighted composite metric.

### Composite Metric Weighting

The visualization script allows you to customize the importance of SMAPE vs. pass@1 performance:

- **Equal Priority** (α=0.5, β=0.5): Balances both metrics equally
- **SMAPE Priority** (α=0.7, β=0.3): Emphasizes lower SMAPE (better SMAPE)
- **pass@1 Priority** (α=0.3, β=0.7): Emphasizes higher Pass@1 (more pass@1es)
- **Custom Weights**: Use `--alpha` and `--beta` flags with any values (automatically normalized)

## Experiment Flow

1. **Parameter Reset**: Before each experiment, all configurations are reset to defaults
2. **Parameter Update**: The specific parameter being tested is updated to the test value
3. **Extraction**: The extraction pipeline runs on the configured sample of PDFs
4. **Evaluation**: Results are compared against ground truth (if available)
5. **Logging**: Results and configurations are saved incrementally
6. **Summary**: After all experiments, a summary is printed to the console
