import os
from datetime import datetime

import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.notebooks.fast_n_slow import MineralSiteMetadata, extract_from_pdf


def extract_from_all_pdfs(
    mineral_report_dir: str = config_general.CDR_REPORTS_DIR,
    full_eval: bool = False,
) -> pd.DataFrame:
    """
    Extract entities from all the PDF files in parallel and return as a DataFrame
    """
    pdf_paths = []
    for i, pdf_path in enumerate(os.listdir(mineral_report_dir)):
        if i > 1 and not full_eval:
            break
        pdf_paths.append(os.path.join(mineral_report_dir, pdf_path))

    data_rows = []
    schema = MineralSiteMetadata.model_json_schema()
    for i, path in enumerate(pdf_paths):
        logger.info(f"{i+1}/{len(pdf_paths)}: Extracting entities from {path}")
        retries = 3
        for attempt in range(retries):
            try:
                entities = extract_from_pdf(path, schema, method="DPE + MAP_REDUCE")
                if entities:
                    entities = entities.model_dump()
                    entities.update(
                        {"cdr_record_id": path.split("/")[-1].split(".")[0]}
                    )
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

    # Replace the "Not Found" values with None
    df = df.replace("Not Found", None)

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
    df = extract_from_all_pdfs(full_eval=False)
    print(df.head())
