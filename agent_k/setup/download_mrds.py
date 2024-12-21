import os
import httpx
import shutil
from loguru import logger
from tqdm import tqdm
from agent_k.config.general import DATA_DIR, RAW_DIR, MRDS_URL, ZIP_PATH, MRDS_DIR


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
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)
    if not os.path.exists(MRDS_DIR):
        os.makedirs(MRDS_DIR)

    # Download the zip file
    logger.info("Downloading MRDS data...")
    download_file(MRDS_URL, ZIP_PATH)

    # Extract using shutil (built-in)
    logger.info("Extracting zip file...")
    shutil.unpack_archive(ZIP_PATH, MRDS_DIR)

    # Clean up by removing the zip file
    os.remove(ZIP_PATH)
    logger.info("Download and extraction complete!")
