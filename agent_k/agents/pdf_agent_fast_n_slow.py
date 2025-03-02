import os
from datetime import datetime
from time import time
from typing import Optional

import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import MinModHyperCols
from agent_k.notebooks.fast_n_slow import MineralSiteMetadata, extract_from_pdf


def extract_from_all_pdfs(
    mineral_report_dir: str = config_general.CDR_REPORTS_DIR,
    sample_size: Optional[int] = None,
    manually_checked_pdf_paths: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame
    """
    # Load PDF paths used for evaluation
    pdf_paths = []
    for _i, pdf_path in enumerate(os.listdir(mineral_report_dir)):
        pdf_paths.append(os.path.join(mineral_report_dir, pdf_path))

    if sample_size:
        pdf_paths = pdf_paths[:sample_size]

    if manually_checked_pdf_paths:
        pdf_paths = [path for path in pdf_paths if path in manually_checked_pdf_paths]

    # Load ground truth data
    ground_truth_path = os.path.join(
        config_general.GROUND_TRUTH_DIR,
        "minmod_hyper_response_enriched_nickel_subset_43_101_gt.csv",
    )
    df_hyper_43_101_subset = pd.read_csv(ground_truth_path)
    logger.info(
        f"Hyper dataframe (subset 43-101) filtered to {len(df_hyper_43_101_subset)} rows"
    )

    data_rows = []
    schema = MineralSiteMetadata.model_json_schema()
    for i, path in enumerate(pdf_paths):
        # Skip if the PDF file is not in the subset of ground truth
        cdr_record_id = path.split("/")[-1].split(".")[0]
        if (
            cdr_record_id
            not in df_hyper_43_101_subset[MinModHyperCols.RECORD_VALUE.value].values
        ):
            logger.warning(
                f"{i + 1}/{len(pdf_paths)}: Skipping {path} because it is not in the ground truth"
            )
            continue

        logger.info(f"{i + 1}/{len(pdf_paths)}: Extracting entities from {path}")
        retries = 3
        for attempt in range(retries):
            try:
                entities = extract_from_pdf(path, schema, method="DPE + MAP_REDUCE")
                if entities:
                    entities = entities.model_dump(mode="json")
                    entities.update({"cdr_record_id": cdr_record_id})
                    data_rows.append(entities)
                    break
            except Exception as e:
                if attempt == retries - 1:  # Last attempt
                    logger.error(
                        f"Failed to extract from {path} after {retries} attempts: {e}"
                    )
                    entities = None
                    break
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")

    # Filter out empty results
    data_rows = [row for row in data_rows if row]

    df = pd.DataFrame(data_rows)

    if not os.path.exists(config_general.PDF_AGENT_CACHE_DIR):
        logger.info(f"Creating directory {config_general.PDF_AGENT_CACHE_DIR}")
        os.makedirs(config_general.PDF_AGENT_CACHE_DIR)

    logger.info(f"Saving extraction results to {config_general.PDF_AGENT_CACHE_DIR}")
    df.to_csv(
        os.path.join(
            config_general.PDF_AGENT_CACHE_DIR,
            f"pdf_agent_extraction_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
        ),
        index=False,
    )

    return df


if __name__ == "__main__":
    # Log metadata about the extraction (total time, number of PDFs, number of entities extracted)
    start_time = time()

    manually_checked_pdf_paths = None
    sample_size = 1
    df = extract_from_all_pdfs(
        sample_size=sample_size, manually_checked_pdf_paths=manually_checked_pdf_paths
    )

    logger.info("Extracting entities from all PDF files")
    logger.info(f"Sample size: {sample_size}")
    logger.info(f"Total time taken: {time() - start_time} seconds")
    logger.info(f"Average time per PDF: {(time() - start_time) / len(df)} seconds")
    logger.info(f"Number of PDFs: {len(df)}")
    logger.info(f"Number of entities extracted: {len(df.index)}")
    print(df.head())
