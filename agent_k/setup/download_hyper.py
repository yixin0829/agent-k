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
