"""
Helper script to clean up vector stores in OpenAI.
Source: https://community.openai.com/t/how-to-delete-all-vector-stores-batch-deletion-endpoint/1148648/2
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

client = AsyncOpenAI()


async def get_recent_vector_store_ids(only_after_timestamp: int) -> list:
    """
    Fetch IDs of vector stores created after the provided UNIX timestamp.

    Parameters:
        only_after_timestamp (int): UNIX epoch time to filter stores.

    Returns:
        list: List of vector store IDs.
    """
    vector_store_ids = []
    params = {"limit": 100, "order": "desc"}
    has_more = True
    after_cursor = None

    while has_more:
        if after_cursor:
            params["after"] = after_cursor
        response = await client.vector_stores.list(**params)
        stores = response.data

        for store in stores:
            if store.created_at <= only_after_timestamp:
                has_more = False
                break
            vector_store_ids.append(store.id)

        if len(stores) < params["limit"]:
            has_more = False
        else:
            after_cursor = stores[-1].id

    return vector_store_ids


async def delete_vector_store(
    store_id: str, semaphore: asyncio.Semaphore, max_retries=5
):
    """
    Attempt to delete a vector store ID with retries and exponential backoff.

    Parameters:
        store_id (str): ID of vector store to delete.
        semaphore (asyncio.Semaphore): Controls concurrency level.
        max_retries (int): Max retry attempts upon failure.

    Returns:
        bool: True if successfully deleted, False otherwise.
    """
    backoff = 1
    retries = 0
    while retries < max_retries:
        async with semaphore:
            try:
                deleted = await client.vector_stores.delete(vector_store_id=store_id)
                print(f"Deleted vector store ID: {deleted.id}")
                return True
            except RateLimitError:
                print(
                    f"Rate limit hit for {store_id}, retrying in {backoff} seconds..."
                )
            except (APIConnectionError, APIError) as e:
                print(
                    f"API error for {store_id}: {e}, retrying in {backoff} seconds..."
                )
            await asyncio.sleep(backoff)
            backoff *= 2
            retries += 1
    print(f"Failed to delete vector store ID: {store_id} after {max_retries} retries.")
    return False


async def delete_stores_in_parallel(store_ids: list):
    """
    Deletes provided vector store IDs with adaptive parallelism and exponential backoff.

    Parameters:
        store_ids (list): List of vector store IDs to delete.
    """
    concurrency = 10  # initial concurrency
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [delete_vector_store(store_id, semaphore) for store_id in store_ids]
    completed, total = 0, len(store_ids)

    for future in asyncio.as_completed(tasks):
        await future
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"Progress: {completed}/{total} deletions completed.")


async def main():
    """
    Main execution function prompting user input and executing deletion.
    """
    try:
        days = int(
            input("Enter number of recent days of vector stores to delete: ").strip()
        )
        if days < 0:
            raise ValueError
    except ValueError:
        print("Invalid input. Provide a non-negative integer.")
        return

    now = datetime.now(timezone.utc)
    cutoff_timestamp = int((now - timedelta(days=days)).timestamp())

    print(f"Fetching vector stores created after UNIX timestamp: {cutoff_timestamp}...")
    store_ids = await get_recent_vector_store_ids(cutoff_timestamp)
    count = len(store_ids)

    if count == 0:
        print("No vector stores found to delete.")
        return

    confirm = input(
        f"DIRE WARNING: Confirm deletion of {count} vector stores by typing 'DELETE': "
    ).strip()
    if confirm != "DELETE":
        print("Deletion aborted by user.")
        return

    start_time = time.time()
    await delete_stores_in_parallel(store_ids)
    duration = time.time() - start_time
    print(f"Deletion completed in {duration:.2f} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
