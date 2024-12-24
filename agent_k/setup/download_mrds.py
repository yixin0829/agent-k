import os
import shutil

import httpx
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

    # Clean up by removing the zip file
    os.remove(config_general.ZIP_PATH)
    logger.info("Download and extraction complete!")
