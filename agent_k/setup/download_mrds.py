"""
Download MRDS data and filter for current commodity.
"""

import os
import shutil

import httpx
import pandas as pd
from loguru import logger
from tqdm import tqdm

import agent_k.config.general as config_general


def download_file(url, path):
    with httpx.stream("GET", url) as r:
        size = int(r.headers.get("content-length", 0)) or None
        with (
            tqdm(total=size, unit="iB", unit_scale=True) as p_bar,
            open(path, "wb") as f,
        ):
            for data in r.iter_bytes():
                p_bar.update(len(data))
                f.write(data)


def process_mrds(mrds_all_file_path: str, commodity: str):
    df = pd.read_csv(mrds_all_file_path)
    # Create temp columns for all commodity columns to lists
    for col in ["commod1", "commod2", "commod3"]:
        df[col + "_temp"] = df[col].apply(
            lambda x: x.split(",") if isinstance(x, str) and x else []
        )
    # Create a temporary column and concatenate all commodity columns into a single column
    df["commodity_all_temp"] = (
        df["commod1_temp"] + df["commod2_temp"] + df["commod3_temp"]
    )
    # Filter for rows where commodity_all contains commodity
    df = df[df["commodity_all_temp"].apply(lambda x: commodity in str(x).lower())]
    # Drop the temporary columns
    df = df.drop(
        columns=["commodity_all_temp", "commod1_temp", "commod2_temp", "commod3_temp"]
    )
    df.to_csv(
        os.path.join(config_general.MRDS_DIR, f"mrds_{commodity}.csv"),
        index=False,
    )


if __name__ == "__main__":
    if not os.path.exists(config_general.DATA_DIR):
        os.makedirs(config_general.DATA_DIR)
    if not os.path.exists(config_general.RAW_DIR):
        os.makedirs(config_general.RAW_DIR)
    if not os.path.exists(config_general.ALL_SOURCES_DIR):
        os.makedirs(config_general.ALL_SOURCES_DIR)
    if not os.path.exists(config_general.MRDS_DIR):
        os.makedirs(config_general.MRDS_DIR)

    # Download the zip file
    logger.info("Downloading MRDS data...")
    download_file(config_general.MRDS_URL, config_general.ZIP_PATH)

    # Extract using shutil (built-in)
    logger.info("Extracting zip file...")
    shutil.unpack_archive(config_general.ZIP_PATH, config_general.MRDS_DIR)

    # Process the MRDS data
    logger.info(
        f"Processing MRDS data to filter for current commodity: {config_general.COMMODITY}..."
    )
    process_mrds(
        mrds_all_file_path=os.path.join(config_general.MRDS_DIR, "mrds.csv"),
        commodity=config_general.COMMODITY,
    )

    # Clean up by removing the zip file if it exists
    if os.path.exists(config_general.ZIP_PATH):
        os.remove(config_general.ZIP_PATH)
    logger.info("Download and extraction complete!")
