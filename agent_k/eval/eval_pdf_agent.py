import os
from datetime import datetime

import pandas as pd

import agent_k.config.general as config_general
from agent_k.agents.pdf_agent_openai import extract_from_pdf
from agent_k.config.logger import logger
from agent_k.config.schemas import RelevantEntitiesPredefined


def extract_from_all_pdfs(full_eval: bool = False) -> pd.DataFrame:
    """
    Extract entities from all the PDF files and return as a DataFrame
    """
    mineral_report_dir = config_general.CDR_REPORTS_DIR
    pdf_paths = []
    for i, pdf_path in enumerate(os.listdir(mineral_report_dir)):
        if i > 0 and not full_eval:
            break
        pdf_paths.append(os.path.join(mineral_report_dir, pdf_path))

    data_rows = []
    for i, pdf_path in enumerate(pdf_paths):
        logger.info(f"{i+1}/{len(pdf_paths)}: Extracting entities from {pdf_path}")
        entities = extract_from_pdf(
            pdf_path, RelevantEntitiesPredefined.model_json_schema()
        )
        if entities:
            entities.update({"cdr_record_id": pdf_path.split("/")[-1].split(".")[0]})
            data_rows.append(entities)
        else:
            logger.error(f"Failed to extract entities from {pdf_path}")
            continue

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


def eval_pdf_agent(full_eval: bool = False, eval_set_version: str = "v3"):
    """
    Evaluate the PDF agent with the eval set.
    """
    # Read the latest extraction results based on the file creation time
    latest_file = os.path.join(
        config_general.PDF_AGENT_CACHE_DIR,
        f"pdf_agent_extraction_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
    )
    if not os.path.exists(latest_file):
        logger.error(
            f"No extraction results found in {config_general.PDF_AGENT_CACHE_DIR}"
        )
        return
    # df_pdf_agent = pd.read_csv(latest_file)

    # Evaluate agent extraction results against the eval set
    logger.info(
        f"Evaluating agent extraction results against the eval set {eval_set_version}"
    )
    logger.info("TODO: Implement the evaluation")


if __name__ == "__main__":
    full_eval = False
    extract_from_all_pdfs(full_eval=full_eval)
    eval_pdf_agent(full_eval=full_eval, eval_set_version="v3")
