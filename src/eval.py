import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.config.logger import logger
from src.config.schemas import MineralEvalDfColumns
from src.utils.general import get_curr_ts


def load_data_and_process() -> pd.DataFrame:
    """
    Load PDF extraction data and ground truth data, then process and merge them.

    Returns:
        pd.DataFrame: Merged dataframe containing both PDF extraction and ground truth data.
    """
    float_columns = [
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    ]

    def standardize_float_column(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Standardize float columns to 0 if they have NaN values.
        """
        logger.info(f"Standardizing float columns: {cols} to 0")
        for col in cols:
            # Replace values containing "Not Found" with 0
            df[col] = df[col].replace(",", "").replace("Not Found", 0)
            # Convert to numeric using pd.to_numeric with errors='coerce'
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Fill any NaN values that resulted from the conversion with 0
            df[col] = df[col].fillna(0)
        return df

    # Note: Load PDF extraction data. Can be replaced with the following line to load a specific extraction file
    agent_extractions = [
        "data/experiments/rag_batch_extraction/2025-09-05_18-01-43_batch_extraction.csv"
    ]
    df_pdf_agent_extraction = pd.concat(
        [pd.read_csv(agent_extraction) for agent_extraction in agent_extractions]
    )

    df_pdf_agent_extraction = standardize_float_column(
        df_pdf_agent_extraction, float_columns
    )
    logger.info(f"PDF extraction dataframe has {len(df_pdf_agent_extraction)} rows")

    # Load ground truth data (inner join will only keep rows that have both a PDF extraction and a ground truth)
    ground_truth_path = "data/processed/43-101_ground_truth/43-101_ground_truth.csv"
    df_gt = pd.read_csv(ground_truth_path)
    df_gt = standardize_float_column(df_gt, float_columns)
    logger.info(f"Ground truth dataframe has {len(df_gt)} rows")

    # Merge the two dataframes
    df_merged = pd.merge(
        df_pdf_agent_extraction,
        df_gt,
        on=[
            MineralEvalDfColumns.ID.value,
            MineralEvalDfColumns.CDR_RECORD_ID.value,
        ],
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    logger.info(f"Merged dataframe has {len(df_merged)} rows")

    # Process numeric columns in merged dataframe
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
    # Replace "Not Found" with 0 and convert to float
    df_merged[numeric_columns] = df_merged[numeric_columns].replace("Not Found", 0)
    df_merged[numeric_columns] = df_merged[numeric_columns].astype(float)

    return df_merged


def calculate_float_metrics(
    df_merged: pd.DataFrame, float_columns: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """
    Calculate numerical metrics between predicted and ground truth values.

    Args:
        df_merged: Dataframe containing both predicted and ground truth values
        float_columns: list of tuples containing (predicted_column, ground_truth_column)

    Returns:
        list of dictionaries containing metrics for each float column
    """
    float_metrics_rows = []

    meta_metrics = {
        "meta_abs_mean_errors": [],
        "meta_smapes": [],
        "meta_pass_1s": [],
    }
    for pdf_col, gt_col in float_columns:
        # Convert to numeric values, coercing errors to NaN
        gt_values = pd.to_numeric(df_merged[gt_col], errors="coerce")
        pdf_values = pd.to_numeric(df_merged[pdf_col], errors="coerce")

        # Convert to numpy arrays for proper operations
        gt_values_np = np.array(gt_values)
        pdf_values_np = np.array(pdf_values)

        # Calculate absolute mean error
        abs_mean_error = np.abs(
            gt_values_np - pdf_values_np
        )  # Calculate absolute differences
        meta_metrics["meta_abs_mean_errors"].extend(abs_mean_error.tolist())
        abs_mean_error = np.mean(abs_mean_error)

        # Calculate R-squared
        r_squared = r2_score(gt_values, pdf_values)  # y_true, y_pred

        # Calculate symmetric mean absolute percentage error (SMAPE)

        # Preprocess before calculating smape: if both ground truth and predicted values are 0, set them to a small value to avoid division by 0
        # Important because 0/0 = NaN, and NaN will be excluded from the smape calculation, making the denominator smaller and smape larger (incorrect)
        gt_values_np = np.where(gt_values_np == 0, 1e-6, gt_values_np)
        pdf_values_np = np.where(pdf_values_np == 0, 1e-6, pdf_values_np)

        # SMAPE formula that caps at 100% (0-100 range)
        smape = np.abs(gt_values_np - pdf_values_np) / (
            np.abs(gt_values_np) + np.abs(pdf_values_np)
        )
        meta_metrics["meta_smapes"].extend(smape.tolist())
        smape = np.mean(smape)

        # Calculate pass@1
        pass_1 = np.isclose(pdf_values_np, gt_values_np, atol=1e-6)
        meta_metrics["meta_pass_1s"].extend(pass_1.tolist())
        pass_1 = np.mean(pass_1)

        metrics = {
            "column": pdf_col,
            "metric_type": "float",
            "support": len(gt_values_np),
            "abs_mean_error": abs_mean_error,
            "r_squared": r_squared,
            "smape": smape,
            "pass@1": pass_1,
        }
        float_metrics_rows.append(metrics)

    # Calculate meta metrics
    assert (
        len(meta_metrics["meta_abs_mean_errors"])
        == len(meta_metrics["meta_smapes"])
        == len(meta_metrics["meta_pass_1s"])
    )
    meta_metrics = {
        "column": "meta_float_columns",
        "metric_type": "float",
        "support": len(meta_metrics["meta_abs_mean_errors"]),
        "abs_mean_error": np.mean(meta_metrics["meta_abs_mean_errors"]),
        "smape": np.mean(meta_metrics["meta_smapes"]),
        "pass@1": np.mean(meta_metrics["meta_pass_1s"]),
    }
    float_metrics_rows.append(meta_metrics)

    return float_metrics_rows


def save_metrics(df_metrics: pd.DataFrame) -> None:
    """
    Save evaluation metrics to a CSV file.

    Args:
        df_metrics: Dataframe containing evaluation metrics
    """
    output_file = f"data/eval/pdf_extraction_metrics_{get_curr_ts()}.csv"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_metrics.to_csv(output_file, index=False)
    logger.info(f"Evaluation metrics saved to {output_file}")


def main() -> None:
    """
    Main function to run the PDF extraction evaluation pipeline.
    """
    # Load data and process
    df_merged = load_data_and_process()

    float_columns = [
        (
            "total_mineral_resource_tonnage_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_gt",
        ),
        (
            "total_mineral_reserve_tonnage_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_gt",
        ),
        (
            "total_mineral_resource_contained_metal_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_gt",
        ),
        (
            "total_mineral_reserve_contained_metal_pred",
            MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_gt",
        ),
    ]

    # Calculate metrics
    float_metrics_rows = calculate_float_metrics(df_merged, float_columns)

    # Combine all metrics into a single dataframe
    all_metrics_rows = float_metrics_rows
    df_metrics = pd.DataFrame(all_metrics_rows)
    save_metrics(df_metrics)

    # Save merged dataframe for error analysis
    reordered_columns = [
        MineralEvalDfColumns.ID.value,
        MineralEvalDfColumns.CDR_RECORD_ID.value,
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_pred",
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_pred",
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_pred",
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_pred",
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_gt",
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_gt",
        MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_gt",
        MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_gt",
    ]
    df_merged = df_merged[reordered_columns]

    # For float columns, calculate the symmetric mean absolute percentage error
    for idx, row in df_merged.iterrows():
        for pdf_col, hyper_col in float_columns:
            # SMAPE will cap at 100% if the difference between the two values is huge
            if pd.notna(row[pdf_col]) and pd.notna(row[hyper_col]):
                smape = np.abs(row[hyper_col] - row[pdf_col]) / (
                    np.abs(row[hyper_col]) + np.abs(row[pdf_col])
                )
                df_merged.at[idx, f"{pdf_col}_smape"] = smape

            # If both values are 0, set the smape to 0
            if row[pdf_col] == 0 and row[hyper_col] == 0:
                df_merged.at[idx, f"{pdf_col}_smape"] = 0

        for pdf_col, hyper_col in float_columns:
            # pass@1
            if np.isclose(row[pdf_col], row[hyper_col], atol=1e-6):
                df_merged.at[idx, f"{pdf_col}_pass@1"] = 1
            else:
                df_merged.at[idx, f"{pdf_col}_pass@1"] = 0

    df_merged.to_csv(f"data/eval/df_merged_{get_curr_ts()}.csv", index=False)


if __name__ == "__main__":
    main()
