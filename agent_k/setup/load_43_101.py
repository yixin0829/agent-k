"""
Upload the 43-101 reports to OpenAI
"""

import os

from openai import OpenAI

import agent_k.config.general as config_general
from agent_k.config.logger import logger

client = OpenAI()


def upload_43_101_reports(dir_path: str):
    """
    Upload all 43-101 reports to OpenAI
    """
    for i, file in enumerate(os.listdir(dir_path)):
        if file.endswith(".pdf"):
            logger.info(f"{i+1}/{len(os.listdir(dir_path))} Uploading {file}")
            file_path = os.path.join(dir_path, file)
            client.files.create(file=open(file_path, "rb"), purpose="assistants")


def list_43_101_reports() -> dict[str, str]:
    """
    Helper function to list all 43-101 reports in OpenAI and create a filename to id mapping
    """
    response = client.files.list()
    # Create a filename to id mapping
    file_id_map = {file.filename: file.id for file in response.data}

    return file_id_map


if __name__ == "__main__":
    upload_43_101_reports(config_general.CDR_REPORTS_DIR)
    print(list_43_101_reports())
