"""
Visualization script for parameter search results.
Creates plots to visualize the performance of different parameter values.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.logger import logger

# Configuration variables
DEFAULT_BASE_DIR = "data/parameter_search"
DEFAULT_PLOTS_OUTPUT_DIR = "data/parameter_search/plots"
DEFAULT_PLOT_FILENAME = "parameter_search_results.png"
DEFAULT_SUMMARY_FILENAME = "data/parameter_search/parameter_summary_table.csv"
DEFAULT_ALPHA = 0.5  # Weight for (1-SMAPE) in composite metric
DEFAULT_BETA = 0.5  # Weight for Pass@1 in composite metric

# Parameter directories mapping
PARAM_DIRS = {
    "Max Reflection Iterations": "max_reflection_iterations",
    "Temperature": "temperature",
    "Number of Retrieved Docs": "num_retrieved_docs",
}


def load_parameter_search_results(
    base_dir: str = DEFAULT_BASE_DIR,
) -> dict[str, pd.DataFrame]:
    """
    Load parameter search results from the output directory.

    Args:
        base_dir: Base directory containing parameter search results

    Returns:
        dictionary mapping experiment names to DataFrames
    """
    results = {}

    # Load results for each parameter type
    for param_name, dir_name in PARAM_DIRS.items():
        results_file = os.path.join(base_dir, dir_name, "parameter_search_results.csv")
        if os.path.exists(results_file):
            results[param_name] = pd.read_csv(results_file)
            logger.info(
                f"Loaded {param_name} results: {len(results[param_name])} experiments"
            )
        else:
            logger.info(f"Warning: No results found for {param_name} at {results_file}")

    return results


def extract_metrics_from_results(
    df: pd.DataFrame, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA
) -> pd.DataFrame:
    """
    Extract evaluation metrics from the evaluation_results column.

    Args:
        df: DataFrame with parameter search results
        alpha: Weight for (SMAPE) in composite metric (default 0.5)
        beta: Weight for Pass@1 in composite metric (default 0.5)

    Returns:
        DataFrame with extracted metrics including weighted composite metric
    """
    import ast

    metrics_data = []

    for _, row in df.iterrows():
        data = {
            "parameter_value": row["parameter_value"],
            "execution_time": row.get("execution_time", None),
            "avg_time_per_pdf": row.get("avg_time_per_pdf", None),
        }

        # Parse evaluation results if available
        if "evaluation_results" in row and pd.notna(row["evaluation_results"]):
            try:
                # Try to parse as string representation of dict
                if isinstance(row["evaluation_results"], str):
                    eval_dict = ast.literal_eval(row["evaluation_results"])
                else:
                    eval_dict = row["evaluation_results"]

                if isinstance(eval_dict, dict):
                    data["smape"] = eval_dict.get("smape", None)
                    data["abs_mean_error"] = eval_dict.get("abs_mean_error", None)
                    data["pass_at_1"] = eval_dict.get("pass_at_1", None)

                    # Calculate weighted composite metric
                    if data["smape"] is not None and data["pass_at_1"] is not None:
                        # Composite metric: α × (SMAPE) + β × Pass@1
                        data["composite_metric"] = (
                            alpha * (1 - data["smape"]) + beta * data["pass_at_1"]
                        )
            except Exception as e:
                pass

        metrics_data.append(data)

    return pd.DataFrame(metrics_data)


def plot_parameter_performance(
    results: dict[str, pd.DataFrame],
    output_dir: str = DEFAULT_PLOTS_OUTPUT_DIR,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
):
    """
    Create visualization plots for parameter search results.

    Args:
        results: dictionary of parameter search results
        output_dir: Directory to save plots
        alpha: Weight for (SMAPE) in composite metric (default 0.5)
        beta: Weight for Pass@1 in composite metric (default 0.5)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Create subplots for each parameter type - now with 4 columns
    fig, axes = plt.subplots(len(results), 4, figsize=(20, 3 * len(results)))

    if len(results) == 1:
        axes = axes.reshape(1, -1)

    for idx, (param_name, df) in enumerate(results.items()):
        if df is None or df.empty:
            continue

        # Extract metrics with composite metric calculation
        metrics_df = extract_metrics_from_results(df, alpha=alpha, beta=beta)

        # Plot 1: SMAPE vs Parameter Value
        ax1 = axes[idx, 0]
        if "smape" in metrics_df.columns and metrics_df["smape"].notna().any():
            ax1.plot(
                metrics_df["parameter_value"], metrics_df["smape"], "o-", markersize=8
            )
            ax1.set_xlabel("Parameter Value")
            ax1.set_ylabel("SMAPE (lower is better)")
            ax1.set_title(f"{param_name}: SMAPE Performance")
            ax1.grid(True, alpha=0.3)

            # Mark best value
            best_idx = metrics_df["smape"].idxmin()
            if pd.notna(best_idx):
                best_value = metrics_df.loc[best_idx, "parameter_value"]
                best_smape = metrics_df.loc[best_idx, "smape"]
                ax1.plot(
                    best_value,
                    best_smape,
                    "r*",
                    markersize=15,
                    label=f"Best: {best_value}",
                )
                ax1.legend()
        else:
            ax1.text(
                0.5,
                0.5,
                "No SMAPE data available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title(f"{param_name}: SMAPE Performance")

        # Plot 2: Average Time per PDF vs Parameter Value
        ax2 = axes[idx, 1]
        if (
            "avg_time_per_pdf" in metrics_df.columns
            and metrics_df["avg_time_per_pdf"].notna().any()
        ):
            ax2.plot(
                metrics_df["parameter_value"],
                metrics_df["avg_time_per_pdf"],
                "s-",
                markersize=8,
                color="green",
            )
            ax2.set_xlabel("Parameter Value")
            ax2.set_ylabel("Avg Time per PDF (seconds)")
            ax2.set_title(f"{param_name}: Average Time per PDF")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No timing data available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title(f"{param_name}: Average Time per PDF")

        # Plot 3: Pass@1 vs Parameter Value
        ax3 = axes[idx, 2]
        if "pass_at_1" in metrics_df.columns and metrics_df["pass_at_1"].notna().any():
            ax3.plot(
                metrics_df["parameter_value"],
                metrics_df["pass_at_1"],
                "^-",
                markersize=8,
                color="purple",
            )
            ax3.set_xlabel("Parameter Value")
            ax3.set_ylabel("Pass@1 (higher is better)")
            ax3.set_title(f"{param_name}: Pass@1 Performance")
            ax3.grid(True, alpha=0.3)

            # Mark best value
            best_idx = metrics_df["pass_at_1"].idxmax()
            if pd.notna(best_idx):
                best_value = metrics_df.loc[best_idx, "parameter_value"]
                best_pass = metrics_df.loc[best_idx, "pass_at_1"]
                ax3.plot(
                    best_value,
                    best_pass,
                    "r*",
                    markersize=15,
                    label=f"Best: {best_value}",
                )
                ax3.legend()
        else:
            ax3.text(
                0.5,
                0.5,
                "No Pass@1 data available",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title(f"{param_name}: Pass@1 Performance")

        # Plot 4: Weighted Composite Metric vs Parameter Value
        ax4 = axes[idx, 3]
        if (
            "composite_metric" in metrics_df.columns
            and metrics_df["composite_metric"].notna().any()
        ):
            ax4.plot(
                metrics_df["parameter_value"],
                metrics_df["composite_metric"],
                "d-",
                markersize=8,
                color="darkblue",
            )
            ax4.set_xlabel("Parameter Value")
            ax4.set_ylabel(f"Composite Metric (higher is better)\nα={alpha}, β={beta}")
            ax4.set_title(f"{param_name}: Weighted Composite Metric")
            ax4.grid(True, alpha=0.3)

            # Mark best value with a star (consistent with other plots)
            best_idx = metrics_df["composite_metric"].idxmax()
            if pd.notna(best_idx):
                best_value = metrics_df.loc[best_idx, "parameter_value"]
                best_composite = metrics_df.loc[best_idx, "composite_metric"]
                ax4.plot(
                    best_value,
                    best_composite,
                    "r*",
                    markersize=15,
                    label=f"Best: {best_value}",
                )
                ax4.legend()
        else:
            ax4.text(
                0.5,
                0.5,
                "No Composite Metric data available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title(f"{param_name}: Weighted Composite Metric")

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, DEFAULT_PLOT_FILENAME)
    plt.savefig(plot_file, dpi=100, bbox_inches="tight")
    logger.info(f"Saved plots to {plot_file}")

    plt.show()

    return fig


def create_summary_table(
    results: dict[str, pd.DataFrame],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> pd.DataFrame:
    """
    Create a summary table of best parameters for each metric.

    Args:
        results: dictionary of parameter search results
        alpha: Weight for (SMAPE) in composite metric (default 0.5)
        beta: Weight for Pass@1 in composite metric (default 0.5)

    Returns:
        Summary DataFrame
    """
    summary_data = []

    for param_name, df in results.items():
        if df is None or df.empty:
            continue

        metrics_df = extract_metrics_from_results(df, alpha=alpha, beta=beta)

        summary_row = {"Parameter": param_name}

        # Find best value for each metric
        if "smape" in metrics_df.columns and metrics_df["smape"].notna().any():
            best_idx = metrics_df["smape"].idxmin()
            summary_row["Best for SMAPE"] = metrics_df.loc[best_idx, "parameter_value"]
            summary_row["SMAPE Score"] = f"{metrics_df.loc[best_idx, 'smape']:.4f}"

        if "pass_at_1" in metrics_df.columns and metrics_df["pass_at_1"].notna().any():
            best_idx = metrics_df["pass_at_1"].idxmax()
            summary_row["Best for Pass@1"] = metrics_df.loc[best_idx, "parameter_value"]
            summary_row["Pass@1 Score"] = f"{metrics_df.loc[best_idx, 'pass_at_1']:.4f}"

        if (
            "avg_time_per_pdf" in metrics_df.columns
            and metrics_df["avg_time_per_pdf"].notna().any()
        ):
            best_idx = metrics_df["avg_time_per_pdf"].idxmin()
            summary_row["Fastest"] = metrics_df.loc[best_idx, "parameter_value"]
            summary_row["Avg Time per PDF (s)"] = (
                f"{metrics_df.loc[best_idx, 'avg_time_per_pdf']:.2f}"
            )

        # Add composite metric best value
        if (
            "composite_metric" in metrics_df.columns
            and metrics_df["composite_metric"].notna().any()
        ):
            best_idx = metrics_df["composite_metric"].idxmax()
            summary_row["Best Composite"] = metrics_df.loc[best_idx, "parameter_value"]
            summary_row["Composite Score"] = (
                f"{metrics_df.loc[best_idx, 'composite_metric']:.4f}"
            )

        summary_data.append(summary_row)

    return pd.DataFrame(summary_data)


def find_best_overall_parameters(
    results: dict[str, pd.DataFrame],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict[str, float]:
    """
    Find the best overall parameter combination based on composite metric.

    Args:
        results: dictionary of parameter search results
        alpha: Weight for (SMAPE) in composite metric (default 0.5)
        beta: Weight for Pass@1 in composite metric (default 0.5)

    Returns:
        dictionary with best parameter values and their composite scores
    """
    best_params = {}

    for param_name, df in results.items():
        if df is None or df.empty:
            continue

        metrics_df = extract_metrics_from_results(df, alpha=alpha, beta=beta)

        if (
            "composite_metric" in metrics_df.columns
            and metrics_df["composite_metric"].notna().any()
        ):
            best_idx = metrics_df["composite_metric"].idxmax()
            best_params[param_name] = {
                "value": metrics_df.loc[best_idx, "parameter_value"],
                "composite_score": metrics_df.loc[best_idx, "composite_metric"],
                "smape": metrics_df.loc[best_idx, "smape"],
                "pass_at_1": metrics_df.loc[best_idx, "pass_at_1"],
            }

    return best_params


def main(alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA):
    """
    Main function to visualize parameter search results.

    Args:
        alpha: Weight for (SMAPE) in composite metric (default 0.5)
        beta: Weight for Pass@1 in composite metric (default 0.5)
    """

    # Load results
    logger.info("Loading parameter search results...")
    results = load_parameter_search_results()

    if not results:
        logger.info("No results found. Please run the parameter search first.")
        return

    # Create visualizations with composite metric
    logger.info(
        f"\nCreating visualizations with composite metric (α={alpha}, β={beta})..."
    )
    plot_parameter_performance(results, alpha=alpha, beta=beta)

    # Find best overall parameters
    logger.info("\nFinding best overall parameters based on composite metric...")
    best_params = find_best_overall_parameters(results, alpha=alpha, beta=beta)

    if best_params:
        logger.info("\n" + "=" * 80)
        logger.info("BEST OVERALL PARAMETERS (Based on Weighted Composite Metric)")
        logger.info(f"Composite Metric = {alpha} × (SMAPE) + {beta} × Pass@1")
        logger.info("=" * 80)

        for param_name, info in best_params.items():
            logger.info(f"\n- {param_name}:")
            logger.info(f"   Best Value: {info['value']:.1f}")
            logger.info(f"   Composite Score: {info['composite_score']:.4f}")
            logger.info(f"   SMAPE: {info['smape']:.4f}")
            logger.info(f"   Pass@1: {info['pass_at_1']:.4f}")

    # Create summary table
    logger.info("\nCreating summary table...")
    summary_df = create_summary_table(results, alpha=alpha, beta=beta)

    if not summary_df.empty:
        logger.info("\n" + "=" * 80)
        logger.info("PARAMETER SEARCH SUMMARY (All Metrics)")
        logger.info("=" * 80)
        logger.info(summary_df.to_string(index=False))

        # Save summary table
        summary_file = DEFAULT_SUMMARY_FILENAME
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\nSummary table saved to {summary_file}")

    logger.info("\nVisualization complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize parameter search results with weighted composite metric"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Weight for (SMAPE) in composite metric (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help=f"Weight for Pass@1 in composite metric (default: {DEFAULT_BETA})",
    )

    args = parser.parse_args()

    # Normalize weights if they don't sum to 1
    if args.alpha + args.beta != 1.0:
        total = args.alpha + args.beta
        args.alpha = args.alpha / total
        args.beta = args.beta / total
        logger.info(
            f"Note: Weights normalized to α={args.alpha:.2f}, β={args.beta:.2f}"
        )

    main(alpha=args.alpha, beta=args.beta)
