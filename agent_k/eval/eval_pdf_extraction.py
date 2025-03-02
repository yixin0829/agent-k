import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from Levenshtein import distance
from sklearn.metrics import mean_absolute_error, r2_score

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import MinModHyperCols
from agent_k.utils.eval_helper import load_latest_pdf_extraction


def load_data_and_process() -> pd.DataFrame:
    """
    Load PDF extraction data and ground truth data, then process and merge them.

    Returns:
        pd.DataFrame: Merged dataframe containing both PDF extraction and ground truth data.
    """
    str_columns = [
        MinModHyperCols.MINERAL_SITE_NAME.value,
        MinModHyperCols.STATE_OR_PROVINCE.value,
        MinModHyperCols.COUNTRY.value,
        MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
        # MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
    ]
    float_columns = [
        MinModHyperCols.TOTAL_GRADE.value,
        MinModHyperCols.TOTAL_TONNAGE.value,
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
                df[col] = df[col].fillna(0)
            df[col] = df[col].replace("Not Found", 0)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # Load PDF extraction data
    df_pdf_agent_extraction = load_latest_pdf_extraction()
    df_pdf_agent_extraction = standardize_string_column(
        df_pdf_agent_extraction, str_columns
    )
    df_pdf_agent_extraction = standardize_float_column(
        df_pdf_agent_extraction, float_columns
    )
    logger.info(
        f"PDF agent extraction dataframe has {len(df_pdf_agent_extraction)} rows"
    )

    # Load ground truth data
    ground_truth_path = os.path.join(
        config_general.GROUND_TRUTH_DIR,
        "minmod_hyper_response_enriched_nickel_subset_43_101_gt.csv",
    )
    df_hyper_43_101_subset = pd.read_csv(ground_truth_path)
    logger.info(
        f"Hyper dataframe (subset 43-101) filtered to {len(df_hyper_43_101_subset)} rows"
    )
    df_hyper_43_101_subset = standardize_string_column(
        df_hyper_43_101_subset, str_columns
    )
    df_hyper_43_101_subset = standardize_float_column(
        df_hyper_43_101_subset, float_columns
    )

    # Merge the two dataframes
    df_merged = pd.merge(
        df_pdf_agent_extraction,
        df_hyper_43_101_subset,
        left_on="cdr_record_id",
        right_on=MinModHyperCols.RECORD_VALUE.value,
        how="inner",
    )
    logger.info(f"Merged dataframe has {len(df_merged)} rows")

    # Process numeric columns in merged dataframe
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_columns] = df_merged[numeric_columns].fillna(0)

    return df_merged


def calculate_string_metrics(
    df_merged: pd.DataFrame, string_columns: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """
    Calculate string comparison metrics between predicted and ground truth values.
    Note: all nan have all been converted to "Not Found"

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

    for pdf_col, hyper_col in float_columns:
        # Convert to numeric values, coercing errors to NaN
        pdf_values = pd.to_numeric(df_merged[pdf_col], errors="coerce")
        hyper_values = pd.to_numeric(df_merged[hyper_col], errors="coerce")

        # Calculate metrics
        abs_mean_error = mean_absolute_error(hyper_values, pdf_values)  # y_true, y_pred
        r_squared = r2_score(hyper_values, pdf_values)  # y_true, y_pred

        # Calculate symmetric mean absolute percentage error (SMAPE)
        smape = (
            2
            * np.abs(hyper_values - pdf_values)
            / (np.abs(hyper_values) + np.abs(pdf_values))
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


def evaluate_metrics(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate metrics comparing PDF extraction results with ground truth data.

    Args:
        df_merged: Dataframe containing both PDF extraction and ground truth data

    Returns:
        pd.DataFrame: Dataframe containing evaluation metrics
    """
    # Define columns to evaluate
    string_columns = [
        ("mineral_site_name_x", MinModHyperCols.MINERAL_SITE_NAME.value + "_y"),
        ("state_or_province_x", MinModHyperCols.STATE_OR_PROVINCE.value + "_y"),
        ("country_x", MinModHyperCols.COUNTRY.value + "_y"),
        ("top_1_deposit_type_x", MinModHyperCols.TOP_1_DEPOSIT_TYPE.value + "_y"),
        # (
        #     "top_1_deposit_environment_x",
        #     MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value + "_y",
        # ),
    ]

    float_columns = [
        ("total_tonnage_x", MinModHyperCols.TOTAL_TONNAGE.value + "_y"),
        ("total_grade_x", MinModHyperCols.TOTAL_GRADE.value + "_y"),
    ]

    # Calculate metrics
    string_metrics_rows = calculate_string_metrics(df_merged, string_columns)
    float_metrics_rows = calculate_float_metrics(df_merged, float_columns)

    # Combine all metrics into a single dataframe
    all_metrics_rows = string_metrics_rows + float_metrics_rows
    df_metrics = pd.DataFrame(all_metrics_rows)

    return df_metrics


def save_metrics(df_metrics: pd.DataFrame) -> None:
    """
    Save evaluation metrics to a CSV file.

    Args:
        df_metrics: Dataframe containing evaluation metrics
    """
    output_file = os.path.join(
        config_general.EVAL_DIR,
        config_general.extraction_evaluation_metrics_file(config_general.COMMODITY),
    )
    df_metrics.to_csv(output_file, index=False)
    logger.info(f"Evaluation metrics saved to {output_file}")


def main() -> None:
    """
    Main function to run the PDF extraction evaluation pipeline.
    """
    df_merged = load_data_and_process()
    df_merged.to_csv("df_merged.csv", index=False)
    df_metrics = evaluate_metrics(df_merged)
    save_metrics(df_metrics)


if __name__ == "__main__":
    main()
