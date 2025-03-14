"""
Source: https://github.com/DARPA-CRITICALMAAS/ta2-minmod-dashboard/blob/main/helpers/dataservice_utils.py
"""

import asyncio
import time

import aiohttp
import requests

from agent_k.config.general import MINMOD_API_URL
from agent_k.config.logger import logger


def fetch_api_data(path, ssl_flag=True, headers=None, params=None):
    """
    Fetches and returns data from the API.

    :param path: str - Endpoint path to append to the global MINMOD_API_ENDPOINT
    :param ssl_flag: bool - Boolean to enable SSL verification
    :param headers: dict - Optional headers for the API request
    :param timeout: int - Request timeout in seconds
    :return: dict - Parsed JSON data from the API
    """
    try:
        # Construct the full URL
        url = f"{MINMOD_API_URL.rstrip('/')}/{path.lstrip('/')}"

        # Make the GET request
        response = requests.get(url, params=params, headers=headers, verify=ssl_flag)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        logger.error(
            f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}"
        )
    except requests.exceptions.Timeout:
        logger.error("The request timed out. Please try again later.")
    except requests.exceptions.RequestException as err:
        logger.error(f"An error occurred while making the API request: {err}")
    except Exception as err:
        logger.critical(f"An unexpected error occurred: {err}")
    return None


# Decorator to log runtime using Python's logging
def log_async_runtime(func):
    """
    A decorator to log the runtime of a function using Python logging.
    """

    async def wrapper(*args, **kwargs):
        start_time = time.time()  # Start timer
        result = await func(*args, **kwargs)  # Execute the asynchronous function
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        logger.info(
            f"Function '{func.__name__}' args : {args[1]} executed in {elapsed_time:.2f} seconds"
        )
        return result

    return wrapper


# Async function to fetch JSON data with timing
@log_async_runtime
async def fetch_json(session, url, params=None):
    async with session.get(url, params=params) as response:
        response.raise_for_status()
        return await response.json()


# Async function to fetch data from all URLs
async def fetch_all(requests):
    requests = [(MINMOD_API_URL + url, params) for url, params in requests]
    connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_json(session, url, params) for url, params in requests]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    swagger_url = MINMOD_API_URL + "/mineral_site_grade_and_tonnage/zinc"
    swagger_data = fetch_api_data(swagger_url, False)
    print(swagger_data)
