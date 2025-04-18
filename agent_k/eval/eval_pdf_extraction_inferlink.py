import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from Levenshtein import distance
from sklearn.metrics import mean_absolute_error, r2_score

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import InferlinkEvalColumns
from agent_k.utils.eval_helper import load_latest_pdf_extraction
from agent_k.utils.general import get_current_timestamp


def load_data_and_process() -> pd.DataFrame:
    """
    Load PDF extraction data and ground truth data, then process and merge them.

    Returns:
        pd.DataFrame: Merged dataframe containing both PDF extraction and ground truth data.
    """
    str_columns = [
        InferlinkEvalColumns.MINERAL_SITE_NAME.value,
        InferlinkEvalColumns.STATE_OR_PROVINCE.value,
        InferlinkEvalColumns.COUNTRY.value,
    ]
    float_columns = [
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    ]

    def standardize_string_column(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        # Helper: Check if the string columns have "Not Found" values and no NaN values. If so, replace NaN with "Not Found".
        for col in cols:
            if df[col].isna().any():
                logger.warning(f"Column {col} has NaN values")
            df[col] = df[col].fillna("Not Found")
        return df

    def standardize_float_column(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        # Helper: Process float columns - replace NaN and "Not Found" with 0. Then cast to float.
        for col in cols:
            if df[col].isna().any():
                logger.warning(f"Column {col} has NaN values")
            # Replace values containing "Not Found" with 0
            df[col] = df[col].replace(",", "").replace("Not Found", 0)
            # Convert to numeric using pd.to_numeric with errors='coerce'
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Fill any NaN values that resulted from the conversion with 0
            df[col] = df[col].fillna(0)
        return df

    # Note: Load PDF extraction data. Can be replaced with the following line to load a specific extraction file
    # df_pdf_agent_extraction = pd.read_csv(
    #     "data/processed/inferlink_extraction_v3_filtered.csv"
    # )
    df_pdf_agent_extraction = load_latest_pdf_extraction(
        dir=os.path.join(config_general.PDF_AGENT_CACHE_DIR, "inferlink")
    )
    df_pdf_agent_extraction = standardize_string_column(
        df_pdf_agent_extraction, str_columns
    )
    df_pdf_agent_extraction = standardize_float_column(
        df_pdf_agent_extraction, float_columns
    )
    logger.info(f"PDF extraction dataframe has {len(df_pdf_agent_extraction)} rows")

    # Load ground truth data
    ground_truth_path = (
        "data/processed/ground_truth/inferlink_ground_truth_filtered.csv"
    )
    df_gt = pd.read_csv(ground_truth_path)
    df_gt = standardize_string_column(df_gt, str_columns)
    df_gt = standardize_float_column(df_gt, float_columns)
    logger.info(f"Ground truth dataframe has {len(df_gt)} rows")

    # Merge the two dataframes
    df_merged = pd.merge(
        df_pdf_agent_extraction,
        df_gt,
        on=InferlinkEvalColumns.CDR_RECORD_ID.value,
        how="inner",
    )
    logger.info(f"Merged dataframe has {len(df_merged)} rows")

    # Process numeric columns in merged dataframe
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
    # Replace "Not Found" with 0 and convert to float
    df_merged[numeric_columns] = df_merged[numeric_columns].replace("Not Found", 0)
    df_merged[numeric_columns] = df_merged[numeric_columns].astype(float)

    return df_merged


def calculate_string_metrics(
    df_merged: pd.DataFrame, string_columns: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """
    Calculate string comparison metrics between predicted and ground truth values.
    All nan have been converted to "Not Found" in standardize_string_column()

    Args:
        df_merged: Dataframe containing both predicted and ground truth values
        string_columns: List of tuples containing (predicted_column, ground_truth_column)

    Returns:
        List of dictionaries containing metrics for each string column
    """
    string_metrics_rows = []

    for pdf_col, hyper_col in string_columns:
        distances = []
        for _idx, row in df_merged.iterrows():
            if pd.notna(row[pdf_col]) and pd.notna(row[hyper_col]):
                dist = distance(str(row[pdf_col]).lower(), str(row[hyper_col]).lower())
                distances.append(dist)

        if distances:
            metrics = {
                "column": pdf_col,
                "metric_type": "string",
                "support": len(distances),
                "mean_edit_distance": np.mean(distances),
                "median_edit_distance": np.median(distances),
                "max_edit_distance": max(distances),
                "exact_matches": sum(d == 0 for d in distances) / len(distances),
            }
            string_metrics_rows.append(metrics)

    return string_metrics_rows


def calculate_float_metrics(
    df_merged: pd.DataFrame, float_columns: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """
    Calculate numerical metrics between predicted and ground truth values.

    Args:
        df_merged: Dataframe containing both predicted and ground truth values
        float_columns: List of tuples containing (predicted_column, ground_truth_column)

    Returns:
        List of dictionaries containing metrics for each float column
    """
    float_metrics_rows = []

    for pdf_col, gt_col in float_columns:
        # Convert to numeric values, coercing errors to NaN
        gt_values = pd.to_numeric(df_merged[gt_col], errors="coerce")
        pdf_values = pd.to_numeric(df_merged[pdf_col], errors="coerce")

        # Calculate metrics
        abs_mean_error = mean_absolute_error(gt_values, pdf_values)  # y_true, y_pred
        r_squared = r2_score(gt_values, pdf_values)  # y_true, y_pred

        # Calculate symmetric mean absolute percentage error (SMAPE)
        smape = (
            2
            * np.abs(gt_values - pdf_values)
            / (np.abs(gt_values) + np.abs(pdf_values))
        )
        smape = np.mean(smape) * 100

        metrics = {
            "column": pdf_col,
            "metric_type": "float",
            "support": len(pdf_values),
            "abs_mean_error": abs_mean_error,
            "r_squared": r_squared,
            "smape": smape,
        }
        float_metrics_rows.append(metrics)

    return float_metrics_rows


def save_metrics(df_metrics: pd.DataFrame) -> None:
    """
    Save evaluation metrics to a CSV file.

    Args:
        df_metrics: Dataframe containing evaluation metrics
    """
    output_file = os.path.join(
        config_general.EVAL_DIR,
        "inferlink",
        f"pdf_extraction_metrics_{get_current_timestamp()}.csv",
    )
    df_metrics.to_csv(output_file, index=False)
    logger.info(f"Evaluation metrics saved to {output_file}")


def main() -> None:
    """
    Main function to run the PDF extraction evaluation pipeline.
    """
    # Load data and process
    df_merged = load_data_and_process()

    # Evaluate metrics
    string_columns = [
        ("mineral_site_name_x", InferlinkEvalColumns.MINERAL_SITE_NAME.value + "_y"),
        ("state_or_province_x", InferlinkEvalColumns.STATE_OR_PROVINCE.value + "_y"),
        ("country_x", InferlinkEvalColumns.COUNTRY.value + "_y"),
    ]

    float_columns = [
        (
            "total_mineral_resource_tonnage_x",
            InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_y",
        ),
        (
            "total_mineral_reserve_tonnage_x",
            InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_y",
        ),
        (
            "total_mineral_resource_contained_metal_x",
            InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_y",
        ),
        (
            "total_mineral_reserve_contained_metal_x",
            InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_y",
        ),
    ]

    # Calculate metrics
    string_metrics_rows = calculate_string_metrics(df_merged, string_columns)
    float_metrics_rows = calculate_float_metrics(df_merged, float_columns)

    # Combine all metrics into a single dataframe
    all_metrics_rows = string_metrics_rows + float_metrics_rows
    df_metrics = pd.DataFrame(all_metrics_rows)
    save_metrics(df_metrics)

    # Save merged dataframe for error analysis
    reordered_columns = [
        InferlinkEvalColumns.CDR_RECORD_ID.value,
        InferlinkEvalColumns.MINERAL_SITE_NAME.value + "_x",
        InferlinkEvalColumns.STATE_OR_PROVINCE.value + "_x",
        InferlinkEvalColumns.COUNTRY.value + "_x",
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_x",
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_x",
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_x",
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_x",
        InferlinkEvalColumns.MAIN_COMMODITY.value,
        InferlinkEvalColumns.COMMODITY.value,
        InferlinkEvalColumns.MINERAL_SITE_NAME.value + "_y",
        InferlinkEvalColumns.STATE_OR_PROVINCE.value + "_y",
        InferlinkEvalColumns.COUNTRY.value + "_y",
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value + "_y",
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value + "_y",
        InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value + "_y",
        InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value + "_y",
    ]
    df_merged = df_merged[reordered_columns]

    # For string columns, calculate the edit distance
    for idx, row in df_merged.iterrows():
        for pdf_col, hyper_col in string_columns:
            if pd.notna(row[pdf_col]) and pd.notna(row[hyper_col]):
                dist = distance(str(row[pdf_col]).lower(), str(row[hyper_col]).lower())
                df_merged.at[idx, f"{pdf_col}_edit_distance"] = dist

    # For float columns, calculate the symmetric mean absolute percentage error
    for idx, row in df_merged.iterrows():
        for pdf_col, hyper_col in float_columns:
            # SMAPE will cap at 200% if the difference between the two values is huge
            if pd.notna(row[pdf_col]) and pd.notna(row[hyper_col]):
                smape = (
                    2
                    * np.abs(row[hyper_col] - row[pdf_col])
                    / (np.abs(row[hyper_col]) + np.abs(row[pdf_col]))
                )
                df_merged.at[idx, f"{pdf_col}_smape"] = smape * 100

            # If both values are 0, set the smape to 0
            if row[pdf_col] == 0 and row[hyper_col] == 0:
                df_merged.at[idx, f"{pdf_col}_smape"] = 0

    df_merged.to_csv(
        f"data/eval/inferlink/df_merged_{get_current_timestamp()}.csv", index=False
    )


if __name__ == "__main__":
    main()
