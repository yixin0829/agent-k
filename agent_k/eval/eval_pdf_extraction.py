import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Levenshtein import distance

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, MinModHyperCols
from agent_k.utils.eval_helper import load_latest_pdf_extraction

if __name__ == "__main__":
    df_pdf_agent_extraction = load_latest_pdf_extraction()
    logger.info(
        f"PDF agent extraction dataframe has {len(df_pdf_agent_extraction)} rows"
    )
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )

    # Filter the hyper dataframe to only include 43-101 data
    df_hyper = df_hyper[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isin(
            [DataSource.API_CDR_LAND.value]
        )
    ]
    logger.info(f"Hyper dataframe filtered to {len(df_hyper)} rows")

    # Merge the two dataframes on the mineral_site_name column
    df_merged = pd.merge(
        df_pdf_agent_extraction,
        df_hyper,
        left_on="cdr_record_id",
        right_on=MinModHyperCols.RECORD_VALUE.value,
        how="inner",
    )
    logger.info(f"Merged dataframe has {len(df_merged)} rows")

    # Define columns to evaluate
    string_columns = [
        ("mineral_site_name_x", MinModHyperCols.MINERAL_SITE_NAME.value + "_y"),
        ("state_or_province_x", MinModHyperCols.STATE_OR_PROVINCE.value + "_y"),
        ("country_x", MinModHyperCols.COUNTRY.value + "_y"),
        ("top_1_deposit_type_x", MinModHyperCols.TOP_1_DEPOSIT_TYPE.value + "_y"),
        (
            "top_1_deposit_environment_x",
            MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value + "_y",
        ),
        ("mineral_site_name_resolved", MinModHyperCols.MINERAL_SITE_NAME.value + "_x"),
        ("state_or_province_resolved", MinModHyperCols.STATE_OR_PROVINCE.value + "_x"),
        ("country_resolved", MinModHyperCols.COUNTRY.value + "_x"),
        (
            "top_1_deposit_type_resolved",
            MinModHyperCols.TOP_1_DEPOSIT_TYPE.value + "_x",
        ),
        (
            "top_1_deposit_environment_resolved",
            MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value + "_x",
        ),
    ]

    float_columns = [
        ("total_tonnage_x", MinModHyperCols.TOTAL_TONNAGE.value + "_y"),
        ("total_grade_x", MinModHyperCols.TOTAL_GRADE.value + "_y"),
    ]

    # Process string metrics
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
                "mean_distance": np.mean(distances),
                "median_distance": np.median(distances),
                "max_distance": max(distances),
                "exact_matches": sum(d == 0 for d in distances) / len(distances),
            }
            string_metrics_rows.append(metrics)

    # Process float metrics
    float_metrics_rows = []
    # set numeric columns to float and fill na with 0
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns
    df_merged[numeric_columns] = df_merged[numeric_columns].fillna(0)
    # Filter out top 1% and bottom 1% of total_tonnage_x, total_grade_x, total_tonnage_y, total_grade_y
    df_merged = df_merged[
        df_merged["total_tonnage_x"].abs()
        < df_merged["total_tonnage_x"].abs().quantile(0.99)
    ]
    df_merged = df_merged[
        df_merged["total_grade_x"].abs()
        < df_merged["total_grade_x"].abs().quantile(0.99)
    ]
    df_merged = df_merged[
        df_merged["total_tonnage_y"].abs()
        < df_merged["total_tonnage_y"].abs().quantile(0.99)
    ]
    df_merged = df_merged[
        df_merged["total_grade_y"].abs()
        < df_merged["total_grade_y"].abs().quantile(0.99)
    ]

    # Plot the kernel density of total_tonnage_x and total_tonnage_y
    df_total_tonnage = df_merged[["total_tonnage_x", "total_tonnage_y"]]
    plt.figure(figsize=(10, 5))
    sns.kdeplot(
        df_total_tonnage["total_tonnage_x"], clip=(0, np.inf), label="Yixin Pred"
    )
    sns.kdeplot(
        df_total_tonnage["total_tonnage_y"], clip=(0, np.inf), label="Inferlink Pred"
    )
    plt.title("Total Tonnage Kernel Density Distribution")
    plt.legend()
    plt.show()

    for pdf_col, hyper_col in float_columns:
        pdf_values = pd.to_numeric(df_merged[pdf_col], errors="coerce")
        hyper_values = pd.to_numeric(df_merged[hyper_col], errors="coerce")
        valid_mask = ~(pd.isna(pdf_values) | pd.isna(hyper_values))
        if valid_mask.any():
            # Calculate residuals between Hyper (reference) and PDF (observed) values
            residuals = hyper_values[valid_mask] - pdf_values[valid_mask]

            # Calculate RMSE (Root Mean Squared Error) and Coefficient of Determination (R-squared)
            rmse = np.sqrt(np.mean(residuals**2) / len(residuals))
            r_squared = 1 - (
                np.sum(residuals**2)
                / np.sum(
                    (hyper_values[valid_mask] - hyper_values[valid_mask].mean()) ** 2
                )
            )

            metrics = {
                "column": pdf_col,
                "metric_type": "float",
                "rmse": rmse,
                "r_squared": r_squared,
            }
            float_metrics_rows.append(metrics)

    # Combine all metrics into a single dataframe
    all_metrics_rows = string_metrics_rows + float_metrics_rows
    df_metrics = pd.DataFrame(all_metrics_rows)

    # Save metrics to CSV
    output_file = os.path.join(
        config_general.EVAL_DIR,
        config_general.extraction_evaluation_metrics_file(config_general.COMMODITY),
    )
    df_metrics.to_csv(output_file, index=False)
    logger.info(f"Evaluation metrics saved to {output_file}")
