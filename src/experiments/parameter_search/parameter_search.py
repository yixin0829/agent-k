"""
Parameter search script for optimizing Agent-K extraction parameters.
This script systematically tests different parameter combinations to find optimal settings.
"""

import os
from time import time
from typing import Any

import pandas as pd
import yaml

import src.config.experiment_config as config_experiment
from src.config.logger import logger
from src.config.schemas import MineralEvalDfColumns
from src.eval import calculate_float_metrics
from src.experiments.multi_method_extraction import extract_from_pdfs
from src.utils.general import get_curr_ts

# Configuration variables
GROUND_TRUTH_TEST_PATH = (
    "data/processed/43-101_ground_truth/43-101_ground_truth_test.csv"
)
GROUND_TRUTH_DEV_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth_dev.csv"
GROUND_TRUTH_FULL_PATH = "data/processed/43-101_ground_truth/43-101_ground_truth.csv"
DEFAULT_CONFIG_PATH = "src/experiments/parameter_search/parameter_search_config.yaml"


def evaluate_extraction_results(
    predictions_df: pd.DataFrame, eval_type: str = "DEV"
) -> dict[str, float]:
    """
    Evaluate extraction results against ground truth.

    Args:
        predictions_df: DataFrame with extraction results
        eval_type: Type of evaluation dataset ("TEST", "DEV", or "FULL")

    Returns:
        dictionary with evaluation metrics
    """
    # Load ground truth based on eval type
    if eval_type == "TEST":
        ground_truth_df = pd.read_csv(GROUND_TRUTH_TEST_PATH)
    elif eval_type == "DEV":
        ground_truth_df = pd.read_csv(GROUND_TRUTH_DEV_PATH)
    else:  # FULL
        ground_truth_df = pd.read_csv(GROUND_TRUTH_FULL_PATH)

    # Merge predictions with ground truth
    merged_df = pd.merge(
        predictions_df,
        ground_truth_df,
        on=MineralEvalDfColumns.CDR_RECORD_ID.value,
        suffixes=("_pred", "_gt"),
    )

    # Define columns to evaluate
    float_columns = [
        (
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_gt",
        ),
        (
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_gt",
        ),
        (
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_gt",
        ),
        (
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_gt",
        ),
    ]

    # Calculate metrics
    metrics_list = calculate_float_metrics(merged_df, float_columns)

    # Extract meta metrics (overall performance)
    meta_metrics = next(
        (m for m in metrics_list if m["column"] == "meta_float_columns"), None
    )

    if meta_metrics:
        return {
            "abs_mean_error": meta_metrics.get("abs_mean_error", None),
            "smape": meta_metrics.get("smape", None),
            "pass_at_1": meta_metrics.get("pass@1", None),
        }

    return {}


def modify_config_parameter(param_name: str, value: Any) -> None:
    """Dynamically modify a configuration parameter."""
    setattr(config_experiment, param_name, value)
    logger.info(f"Set {param_name} = {value}")


def reset_default_configs(
    model_name: str = "gpt-4o-mini-2024-07-18",
    sample_size: int = None,
    eval_type: str = "DEV",
) -> None:
    """Reset configurations to default values for Agent-K method."""
    # Set to use Agent-K method
    config_experiment.PDF_EXTRACTION_METHOD = config_experiment.ExtractionMethod.AGENT_K

    # Use specified model for all models in the experiment
    config_experiment.AGENT_K_MODEL = model_name
    config_experiment.PYTHON_AGENT_MODEL = model_name
    config_experiment.GRADE_HALLUCINATION_MODEL = model_name
    config_experiment.SLOW_EXTRACT_VALIDATION_MODEL = model_name
    config_experiment.SLOW_EXTRACT_OPTIMIZER_MODEL = model_name

    # Default parameter values
    config_experiment.AGENT_K_TEMPERATURE = 0.2
    config_experiment.PYTHON_AGENT_TEMPERATURE = 0.2
    config_experiment.GRADE_HALLUCINATION_TEMPERATURE = 0.2
    config_experiment.SLOW_EXTRACT_VALIDATION_TEMPERATURE = 0.2
    config_experiment.SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = 0.2

    config_experiment.MAX_REFLECTION_ITERATIONS = 5
    config_experiment.NUM_RETRIEVED_DOCS = 2

    # Set sample size and eval type from parameters
    config_experiment.PDF_EXTRACTION_SAMPLE_SIZE = sample_size
    config_experiment.PDF_EXTRACTION_EVAL_TYPE = eval_type

    logger.info(
        f"Reset configurations to default values (sample_size={sample_size}, eval_type={eval_type})"
    )


def run_single_experiment(
    param_name: str, param_value: Any, experiment_name: str, output_base_dir: str
) -> dict[str, Any]:
    """
    Run a single experiment with specified parameter value.

    Returns:
        dict containing experiment results and metrics
    """
    timestamp = get_curr_ts()
    experiment_dir = os.path.join(
        output_base_dir, experiment_name, f"{param_name}_{param_value}"
    )
    os.makedirs(experiment_dir, exist_ok=True)

    # Set the parameter value
    if param_name == "TEMPERATURE":
        # Update all temperature parameters
        config_experiment.AGENT_K_TEMPERATURE = param_value
        config_experiment.PYTHON_AGENT_TEMPERATURE = param_value
        config_experiment.GRADE_HALLUCINATION_TEMPERATURE = param_value
        config_experiment.SLOW_EXTRACT_VALIDATION_TEMPERATURE = param_value
        config_experiment.SLOW_EXTRACT_OPTIMIZER_TEMPERATURE = param_value
    elif param_name == "MAX_REFLECTION_ITERATIONS":
        # Update max reflection iterations
        config_experiment.MAX_REFLECTION_ITERATIONS = param_value
    elif param_name == "NUM_RETRIEVED_DOCS":
        # Update number of retrieved docs
        config_experiment.NUM_RETRIEVED_DOCS = param_value
    else:
        raise ValueError(f"Invalid parameter name: {param_name}")

    # Run extraction
    start_time = time()
    output_filename = f"extraction_results_{timestamp}.csv"

    try:
        logger.info(
            f"Starting experiment: {experiment_name} with {param_name}={param_value}"
        )

        final_df = extract_from_pdfs(
            sample_size=config_experiment.PDF_EXTRACTION_SAMPLE_SIZE,
            method=config_experiment.PDF_EXTRACTION_METHOD.value,
            eval_type=config_experiment.PDF_EXTRACTION_EVAL_TYPE,
            output_dir=experiment_dir,
            output_filename=output_filename,
        )

        # Evaluate results if ground truth is available
        eval_results = None
        try:
            eval_results = evaluate_extraction_results(
                predictions_df=final_df,
                eval_type=config_experiment.PDF_EXTRACTION_EVAL_TYPE,
            )
        except Exception as e:
            logger.warning(f"Could not evaluate results: {e}")

        execution_time = time() - start_time

        # Create result summary
        result_summary = {
            "parameter_name": param_name,
            "parameter_value": param_value,
            "execution_time": execution_time,
            "num_pdfs_processed": len(final_df),
            "num_entities_extracted": len(final_df.columns)
            - 2,  # Exclude id and cdr_record_id
            "avg_time_per_pdf": execution_time / len(final_df)
            if len(final_df) > 0
            else 0,
            "output_file": os.path.join(experiment_dir, output_filename),
            "evaluation_results": eval_results,
            "timestamp": timestamp,
        }

        # Save experiment configuration
        config_dict = {
            "experiment_name": experiment_name,
            "parameter_tested": param_name,
            "parameter_value": param_value,
            "model": config_experiment.AGENT_K_MODEL,
            "method": config_experiment.PDF_EXTRACTION_METHOD.value,
            "sample_size": config_experiment.PDF_EXTRACTION_SAMPLE_SIZE,
            "eval_type": config_experiment.PDF_EXTRACTION_EVAL_TYPE,
            "temperature": config_experiment.AGENT_K_TEMPERATURE,
            "max_reflection_iterations": config_experiment.MAX_REFLECTION_ITERATIONS,
            "num_retrieved_docs": config_experiment.NUM_RETRIEVED_DOCS,
            "execution_time": execution_time,
            "timestamp": timestamp,
        }

        config_file = os.path.join(
            experiment_dir, f"experiment_config_{timestamp}.yaml"
        )
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        logger.info(
            f"Experiment completed: {experiment_name} with {param_name}={param_value}"
        )
        return result_summary

    except Exception as e:
        logger.error(
            f"Experiment failed: {experiment_name} with {param_name}={param_value}: {e}"
        )
        return {
            "parameter_name": param_name,
            "parameter_value": param_value,
            "error": str(e),
            "timestamp": timestamp,
        }


def run_parameter_search(
    param_name: str,
    param_values: list[Any],
    experiment_name: str,
    output_base_dir: str,
    model_name: str = "gpt-4o-mini-2024-07-18",
    sample_size: int = None,
    eval_type: str = "DEV",
) -> pd.DataFrame:
    """
    Run parameter search for a specific parameter with multiple values.

    Returns:
        DataFrame with results for all parameter values
    """
    results = []

    for value in param_values:
        # Reset to default configs before each experiment
        reset_default_configs(model_name, sample_size, eval_type)

        # Run experiment
        result = run_single_experiment(
            param_name=param_name,
            param_value=value,
            experiment_name=experiment_name,
            output_base_dir=output_base_dir,
        )
        results.append(result)

        # Save intermediate results
        results_df = pd.DataFrame(results)
        results_file = os.path.join(
            output_base_dir, experiment_name, "parameter_search_results.csv"
        )
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved intermediate results to {results_file}")

    return pd.DataFrame(results)


def load_search_config(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load parameter search configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def main():
    """Main function to run all parameter search experiments."""
    # Load configuration
    config = load_search_config()

    # Base output directory for all experiments
    output_base_dir = config["output"]["base_dir"]
    os.makedirs(output_base_dir, exist_ok=True)

    # Set dataset configuration
    # Note: null in YAML becomes None in Python, which means "use all PDFs"
    sample_size = config["dataset"]["sample_size"]  # This will be None if null in YAML
    eval_type = config["dataset"]["eval_type"]

    config_experiment.PDF_EXTRACTION_SAMPLE_SIZE = sample_size
    config_experiment.PDF_EXTRACTION_EVAL_TYPE = eval_type

    logger.info(f"Using model: {config['model']['name']}")
    logger.info(f"Sample size: {sample_size if sample_size is not None else 'ALL'}")
    logger.info(f"Eval type: {eval_type}")

    # Store all results
    all_results = {}

    # Experiment 1: Max Reflection Iterations
    logger.info("=" * 80)
    logger.info("Starting Experiment 1: Max Reflection Iterations")
    logger.info("=" * 80)

    reflection_values = config["parameters"]["max_reflection_iterations"]["values"]
    reflection_results = run_parameter_search(
        param_name="MAX_REFLECTION_ITERATIONS",
        param_values=reflection_values,
        experiment_name="max_reflection_iterations",
        output_base_dir=output_base_dir,
        model_name=config["model"]["name"],
        sample_size=sample_size,
        eval_type=eval_type,
    )
    all_results["Max Reflection Iterations"] = reflection_results

    # Experiment 2: Temperature
    logger.info("=" * 80)
    logger.info("Starting Experiment 2: Temperature")
    logger.info("=" * 80)

    temperature_values = config["parameters"]["temperature"]["values"]
    temperature_results = run_parameter_search(
        param_name="TEMPERATURE",
        param_values=temperature_values,
        experiment_name="temperature",
        output_base_dir=output_base_dir,
        model_name=config["model"]["name"],
        sample_size=sample_size,
        eval_type=eval_type,
    )
    all_results["Temperature"] = temperature_results

    # Experiment 3: Number of Retrieved Documents
    logger.info("=" * 80)
    logger.info("Starting Experiment 3: Number of Retrieved Documents")
    logger.info("=" * 80)

    num_docs_values = config["parameters"]["num_retrieved_docs"]["values"]
    num_docs_results = run_parameter_search(
        param_name="NUM_RETRIEVED_DOCS",
        param_values=num_docs_values,
        experiment_name="num_retrieved_docs",
        output_base_dir=output_base_dir,
        model_name=config["model"]["name"],
        sample_size=sample_size,
        eval_type=eval_type,
    )
    all_results["Number of Retrieved Documents"] = num_docs_results

    # Create consolidated results DataFrame
    consolidated_df = pd.concat(all_results.values(), ignore_index=True)
    consolidated_file = os.path.join(
        output_base_dir, "all_parameter_search_results.csv"
    )
    consolidated_df.to_csv(consolidated_file, index=False)

    logger.info("=" * 80)
    logger.info("Parameter search completed!")
    logger.info(f"Results saved to: {output_base_dir}")
    logger.info("=" * 80)

    # Print summary to console
    logger.info("\n" + "=" * 80)
    logger.info("PARAMETER SEARCH SUMMARY")
    logger.info("=" * 80)

    for experiment_name, results_df in all_results.items():
        logger.info(f"\n{experiment_name}:")
        logger.info("-" * 40)

        valid_results = results_df[
            ~results_df.get("error", pd.Series([False] * len(results_df))).notna()
        ]
        if len(valid_results) > 0:
            # Find best parameter value based on evaluation metrics if available
            if "evaluation_results" in valid_results.columns:
                eval_results_valid = valid_results[
                    valid_results["evaluation_results"].notna()
                ]
                if len(eval_results_valid) > 0:
                    smape_scores = []
                    for idx, row in eval_results_valid.iterrows():
                        eval_res = row["evaluation_results"]
                        if isinstance(eval_res, dict) and "smape" in eval_res:
                            smape_scores.append((idx, eval_res["smape"]))

                    if smape_scores:
                        smape_scores.sort(key=lambda x: x[1])
                        best_idx = smape_scores[0][0]
                    else:
                        best_idx = valid_results["execution_time"].idxmin()
                else:
                    best_idx = valid_results["execution_time"].idxmin()
            else:
                best_idx = valid_results["execution_time"].idxmin()

            best_result = valid_results.loc[best_idx]
            logger.info(
                f"Best {best_result['parameter_name']}: {best_result['parameter_value']}"
            )
            logger.info(f"Execution Time: {best_result['execution_time']:.2f} seconds")
            logger.info(
                f"Avg Time per PDF: {best_result['avg_time_per_pdf']:.2f} seconds"
            )

            # Print evaluation metrics if available
            if (
                "evaluation_results" in best_result
                and best_result["evaluation_results"]
            ):
                eval_res = best_result["evaluation_results"]
                if isinstance(eval_res, dict):
                    logger.info("Evaluation Metrics:")
                    for metric, value in eval_res.items():
                        if value is not None:
                            logger.info(f"  - {metric}: {value:.4f}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
