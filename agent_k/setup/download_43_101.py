"""
1. Fetch deduplicated mineral site entities for the current commodity from the MinMod API.
2. Download 43-101 PDF reports from the CDR API.
"""

import asyncio
import os
import warnings

import httpx
import pandas as pd
from tqdm import tqdm

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, MinModHyperCols
from agent_k.setup.download_hyper import download_minmod_hyper_csv, enrich_minmod_hyper

warnings.filterwarnings("ignore")
tqdm.pandas()


async def download_report(record_id: str, save_path: str, semaphore: asyncio.Semaphore):
    """
    Downloads a single PDF report from the CDR API using the provided record ID.

    Args:
        record_id: Unique identifier for the report to download
        save_path: Directory path where the PDF should be saved
        semaphore: Semaphore to control concurrent downloads

    Returns:
        bool: True if download was successful, False otherwise
    """
    url = (
        config_general.API_CDR_LAND_URL
        + config_general.DOCUMENT_BY_ID_ENDPOINT.format(doc_id=record_id)
    )
    async with semaphore:  # Limit concurrent requests
        try:
            async with httpx.AsyncClient() as client:
                # Download PDF with authentication headers and write to file in save_path
                res = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {os.getenv("API_CDR_AUTH_TOKEN")}"
                    },
                    follow_redirects=True,
                )
                res.raise_for_status()
                logger.info(f"Downloaded {record_id} successfully")

                os.makedirs(save_path, exist_ok=True)

                with open(os.path.join(save_path, f"{record_id}.pdf"), "wb") as f:
                    f.write(res.content)
                return True
        except Exception:
            logger.error(f"Error downloading {record_id}: {res.status_code}")
            return False


async def download_all_reports(df_hyper: pd.DataFrame, max_concurrent_requests: int):
    """
    Downloads all PDF reports referenced in the provided DataFrame concurrently.
    """
    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks = []

    # Create download tasks for records with 43-101 data source and not yet downloaded
    for idx, row in df_hyper.iterrows():
        if (
            row[MinModHyperCols.DATA_SOURCE.value] == DataSource.API_CDR_LAND.value
            and not row[MinModHyperCols.DOWNLOADED_PDF.value]
        ):
            record_id = row[MinModHyperCols.RECORD_VALUE.value]
            task = asyncio.create_task(
                download_report(
                    record_id,
                    save_path=config_general.CDR_REPORTS_DIR,
                    semaphore=semaphore,
                )
            )
            tasks.append((task, idx))

    # Wait for all downloads to complete
    results = []
    for task, idx in tasks:
        result: bool = await task
        results.append((idx, result))

    return results


def download_reports_main(max_concurrent_requests: int = 10):
    """
    Main function to orchestrate downloading all reports.
    """
    # Load report metadata
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )

    # Download all reports concurrently
    results = asyncio.run(download_all_reports(df_hyper, max_concurrent_requests))

    for idx, success in results:
        df_hyper.loc[idx, MinModHyperCols.DOWNLOADED_PDF.value] = success

    # logger.info summary and save updated DataFrame
    logger.info(df_hyper[MinModHyperCols.DOWNLOADED_PDF.value].value_counts())
    df_hyper.to_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )


if __name__ == "__main__":
    download_minmod_hyper_csv()
    enrich_minmod_hyper()
    download_reports_main()
