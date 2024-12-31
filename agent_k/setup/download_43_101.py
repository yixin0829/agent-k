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
from agent_k.utils.ms_model import MineralSite

warnings.filterwarnings("ignore")
tqdm.pandas()


def download_minmod_hyper_csv():
    # Create directories if they don't exist
    if not os.path.exists(config_general.DATA_DIR):
        os.makedirs(config_general.DATA_DIR)
    if not os.path.exists(config_general.RAW_DIR):
        os.makedirs(config_general.RAW_DIR)
    if not os.path.exists(config_general.MINMOD_DIR):
        os.makedirs(config_general.MINMOD_DIR)

    logger.info(f"Downloading MinMod {config_general.COMMODITY} sites data...")
    ms = MineralSite(commodity=config_general.COMMODITY)
    try:
        ms.init()
        df = ms.df
        # Update deposit type options based on the selected commodity
        deposit_options = [{"label": dt, "value": dt} for dt in ms.deposit_types]

        # Update country options based on the selected commodity
        country_options = [
            {"label": country, "value": country} for country in ms.country
        ]
    except Exception:
        return (
            deposit_options,
            country_options,
        )

    # Convert column names to lowercase with underscores
    df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    df.rename(
        columns={"state/province": MinModHyperCols.STATE_OR_PROVINCE.value},
        inplace=True,
    )

    df.to_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )

    logger.info("Download complete!")


def enrich_minmod_hyper():
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.hyper_reponse_file(config_general.COMMODITY),
        )
    )
    logger.info(f"df_hyper shape: {df_hyper.shape}")

    # Assert that the ms column is unique
    assert (
        df_hyper[MinModHyperCols.MINERAL_SITE_NAME.value].nunique() == df_hyper.shape[0]
    ), f"{MinModHyperCols.MINERAL_SITE_NAME.value} column is not unique"

    # If mineral site name contains (MRDS, DOI, 43-101) then set the data source to the appropriate enum
    # Examples:
    # [Unnamed Copper Prospect](https://minmod.isi.edu/resource/dedup_site__mrdata-usgs-gov-mrds__10013841) -> MRDS
    # [Unnamed Prospect](https://minmod.isi.edu/resource/dedup_site__doi-org-10-5066-p9htergk__13368) -> DOI
    # [Minago Nickel Mine](https://minmod.isi.edu/resource/dedup_site__api-cdr-land-v1-docs-documents__020ad3e9246df19d58b68751eb9e1e49bf8631d31c70d9737647bfab306354fa0c) -> 43-101
    for ds in DataSource:
        ds_pattern = ds.name.lower().replace("_", "-")
        ds_mask = df_hyper[MinModHyperCols.MINERAL_SITE_NAME.value].str.contains(
            ds_pattern
        )
        df_hyper.loc[ds_mask, MinModHyperCols.DATA_SOURCE.value] = ds.value
    # Set data source to OTHER if not found in the enum
    df_hyper.loc[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isna(),
        MinModHyperCols.DATA_SOURCE.value,
    ] = DataSource.OTHER.value

    url_pattern = r"(https?://[^\s\)]+)"
    df_hyper.loc[:, MinModHyperCols.SOURCE_VALUE.value] = df_hyper[
        MinModHyperCols.MINERAL_SITE_NAME.value
    ].str.extract(url_pattern, expand=False)
    df_hyper.loc[:, MinModHyperCols.RECORD_VALUE.value] = (
        df_hyper[MinModHyperCols.SOURCE_VALUE.value].str.split("__").str[-1]
    )

    # Check if PDF report with record value exist in CDR_REPORTS_DIR
    df_hyper[MinModHyperCols.DOWNLOADED_PDF.value] = False
    for idx, row in df_hyper.iterrows():
        if row[MinModHyperCols.DATA_SOURCE.value] == DataSource.API_CDR_LAND.value:
            record_id = row[MinModHyperCols.RECORD_VALUE.value]
            if os.path.exists(
                os.path.join(config_general.CDR_REPORTS_DIR, f"{record_id}.pdf")
            ):
                df_hyper.loc[idx, MinModHyperCols.DOWNLOADED_PDF.value] = True

    logger.info(
        f"{df_hyper[MinModHyperCols.DOWNLOADED_PDF.value].sum()}/{df_hyper.shape[0]} 43-101 reports already downloaded. Skipping download by setting DOWNLOADED_PDF to True."
    )

    # Assert all enriched columns are not null
    for col in [
        MinModHyperCols.DATA_SOURCE.value,
        MinModHyperCols.SOURCE_VALUE.value,
        MinModHyperCols.RECORD_VALUE.value,
        MinModHyperCols.DOWNLOADED_PDF.value,
    ]:
        assert df_hyper[col].notna().all(), f"{col} column has null values"

    # Extract mineral site name from the mineral site name column
    df_hyper[MinModHyperCols.MINERAL_SITE_NAME.value] = df_hyper[
        MinModHyperCols.MINERAL_SITE_NAME.value
    ].str.extract(r"\[(.*?)\]")

    df_hyper.to_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )
    logger.info(f"df_hyper shape: {df_hyper.shape}")
    logger.info("Successfully enriched hyper with new columns!")


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
