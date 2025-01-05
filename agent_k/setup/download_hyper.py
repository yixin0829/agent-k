import ast
import os

import pandas as pd

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, MinModHyperCols
from agent_k.utils.ms_model import MineralSite


def download_minmod_hyper_csv():
    # Create directories if they don't exist
    if not os.path.exists(config_general.DATA_DIR):
        os.makedirs(config_general.DATA_DIR)
    if not os.path.exists(config_general.RAW_DIR):
        os.makedirs(config_general.RAW_DIR)
    if not os.path.exists(config_general.GROUND_TRUTH_DIR):
        os.makedirs(config_general.GROUND_TRUTH_DIR)

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

    df.to_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )

    logger.info("Hyper response CSV download complete!")


def enrich_minmod_hyper():
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.hyper_reponse_file(config_general.COMMODITY),
        )
    )
    logger.info(f"df_hyper shape: {df_hyper.shape}")

    # Assert mineral site uri is unique
    assert (
        df_hyper[MinModHyperCols.MINERAL_SITE_URI.value].nunique() == df_hyper.shape[0]
    ), f"{MinModHyperCols.MINERAL_SITE_URI.value} column is not unique"

    # Convert Sites column to list
    df_hyper[MinModHyperCols.SITES.value] = df_hyper[MinModHyperCols.SITES.value].apply(
        ast.literal_eval
    )

    # Explode sites column
    df_hyper = df_hyper.explode(MinModHyperCols.SITES.value)

    # If sites column contains (MRDS, DOI, 43-101) then set the data source
    # Examples:
    # dedup_site__mrdata-usgs-gov-mrds__10013841 -> MRDS
    # dedup_site__doi-org-10-5066-p9htergk__13368 -> DOI
    # dedup_site__api-cdr-land-v1-docs-documents__020ad3e9246df19d58b654fa0c -> 43-101
    for ds in DataSource:
        ds_pattern = ds.name.lower().replace("_", "-")
        ds_mask = df_hyper[MinModHyperCols.SITES.value].str.contains(ds_pattern)
        df_hyper.loc[ds_mask, MinModHyperCols.DATA_SOURCE.value] = ds.value
    # Set data source to OTHER if not found in the enum
    df_hyper.loc[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isna(),
        MinModHyperCols.DATA_SOURCE.value,
    ] = DataSource.OTHER.value

    df_hyper.loc[:, MinModHyperCols.RECORD_VALUE.value] = (
        df_hyper[MinModHyperCols.SITES.value].str.split("__").str[-1]
    )

    df_hyper_43_101 = df_hyper[
        df_hyper[MinModHyperCols.DATA_SOURCE.value] == DataSource.API_CDR_LAND.value
    ]
    # Get unique record values
    unique_record_values = df_hyper_43_101[MinModHyperCols.RECORD_VALUE.value].unique()
    num_unique_43_101_reports = len(unique_record_values)
    logger.info(f"Number of 43-101 reports: {num_unique_43_101_reports}")

    # Check if PDF report with record value exist in CDR_REPORTS_DIR
    downloaded_pdf_record_ids = []
    for record_id in unique_record_values:
        if os.path.exists(
            os.path.join(config_general.CDR_REPORTS_DIR, f"{record_id}.pdf")
        ):
            downloaded_pdf_record_ids.append(record_id)

    df_hyper[MinModHyperCols.DOWNLOADED_PDF.value] = df_hyper[
        MinModHyperCols.RECORD_VALUE.value
    ].isin(downloaded_pdf_record_ids)

    logger.info(
        f"{len(downloaded_pdf_record_ids)}/{num_unique_43_101_reports} 43-101 reports already downloaded (skip downloading these by setting DOWNLOADED_PDF to True)."
    )

    # Assert all enriched columns are not null
    for col in [
        MinModHyperCols.DATA_SOURCE.value,
        MinModHyperCols.MINERAL_SITE_URI.value,
        MinModHyperCols.RECORD_VALUE.value,
        MinModHyperCols.DOWNLOADED_PDF.value,
    ]:
        assert df_hyper[col].notna().all(), f"{col} column has null values"

    df_hyper.to_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        ),
        index=False,
    )
    logger.info("Successfully enriched hyper with new columns!")


if __name__ == "__main__":
    download_minmod_hyper_csv()
    enrich_minmod_hyper()
