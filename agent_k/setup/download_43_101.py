import os
from dotenv import load_dotenv
import httpx
import pandas as pd
from tqdm import tqdm
import warnings
from agent_k.utils.minmod_sparql import run_minmod_query
from agent_k.config.schemas import MinModHyperCols
import asyncio
from agent_k.utils.ms_model import MineralSite

warnings.filterwarnings("ignore")
tqdm.pandas()

load_dotenv()

# Read ENV variables
API_USR_NAME: str = os.getenv("API_CDR_USR_NAME")
API_PASSWORD: str = os.getenv("API_CDR_PASSWORD")
AUTH_TOKEN: str = os.getenv("API_CDR_AUTH_TOKEN")

# CDR API
API_CDR_LAND_URL = "https://api.cdr.land/v1"
DOCUMENTS_ENDPOINT = "/docs/documents"
DOCUMENT_BY_ID_ENDPOINT = "/docs/document/{doc_id}"  # for querying pdf document by id
PROVENANCE_ENDPOINT = "/docs/documents/q/provenance/url"  # for querying record id based on source id (e.g. https://w3id.org/usgs/z/4530692/5CAAGFXV)


def download_hyper_csv():
    # Create data/raw/minmod directories if they don't exist
    data_dir = "data"
    raw_dir = os.path.join(data_dir, "raw")
    minmod_dir = os.path.join(raw_dir, "minmod")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(minmod_dir):
        os.makedirs(minmod_dir)

    print("Downloading MinMod nickel sites data...")
    ms = MineralSite(commodity="nickel")
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

    df.to_csv(os.path.join(minmod_dir, "minmod_hyper_response.csv"), index=False)

    print("Download complete!")


def download_minmod_site_record_id():
    query = """SELECT DISTINCT ?ms ?source ?record
    WHERE {
        ?ms a :MineralSite ; :source_id ?source ; :record_id ?record .
    }"""

    run_minmod_query(
        query, values=True, csv_path="data/raw/minmod/minmod_sites_record_id.csv"
    )


def enrich_hyper_w_record_id():
    df_hyper = pd.read_csv("./data/raw/minmod/minmod_hyper_response.csv")
    df_minmod_sites = pd.read_csv("./data/raw/minmod/minmod_sites_record_id.csv")
    print(f"df_hyper shape: {df_hyper.shape}")
    print(f"df minmod sites shape: {df_minmod_sites.shape}")

    # Deduplicate the ms column in df_hyper
    df_hyper = df_hyper.drop_duplicates(subset=["ms"])
    print(f"(after dedup) df_hyper shape: {df_hyper.shape}")

    # Deduplicate the ms.value column in df_minmod_sites
    df_minmod_sites = df_minmod_sites.drop_duplicates(subset=["ms.value"])
    print(f"(after dedup) df_minmod_sites shape: {df_minmod_sites.shape}")
    import ast

    # Parse ms from string to list type
    df_hyper["ms"] = df_hyper["ms"].apply(
        lambda x: [x] if not x.startswith("[") else ast.literal_eval(x)
    )

    # Explode the ms and ms_name columns
    print(f"df_hyper shape: {df_hyper.shape}")
    df_hyper = df_hyper.explode(["ms"])
    print(f"df_hyper exploded shape: {df_hyper.shape}")

    df_hyper = df_hyper.merge(
        df_minmod_sites, left_on="ms", right_on="ms.value", how="left"
    )
    df_hyper = df_hyper.drop(columns=["ms.value"])

    # Impute the record id for those `1` using `https://api.cdr.land/v1/docs/documents/q/provenance/url?pattern=...` CDR endpoint.
    df_hyper["record.value_imputed"] = df_hyper["record.value"]
    for idx, row in df_hyper.iterrows():
        if row["record.value"] == "1":
            source_id = row["source.value"]
            print(f"Imputing record id for row {idx} with source id: {source_id}")
            url = API_CDR_LAND_URL + PROVENANCE_ENDPOINT
            try:
                res = httpx.post(
                    url=url,
                    params={"pattern": source_id, "size": 10, "page": 0},
                    headers={
                        "Authorization": f"Bearer {os.getenv("API_CDR_AUTH_TOKEN")}"
                    },
                )
                res.raise_for_status()
                res_json = res.json()
                if res_json:
                    record_id = res_json[0]["id"]
                    df_hyper.loc[idx, "record.value_imputed"] = record_id
            except Exception as e:
                print(f"Error: {e}")
                continue

    df_hyper["record.value_imputed"].value_counts(ascending=False).head(5)
    df_hyper.to_csv("./data/raw/minmod/minmod_hyper_response_enriched.csv", index=False)


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
    url = API_CDR_LAND_URL + DOCUMENT_BY_ID_ENDPOINT.format(record_id=record_id)
    async with semaphore:  # Limit concurrent requests
        try:
            async with httpx.AsyncClient() as client:
                # Download PDF with authentication headers
                res = await client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {os.getenv("API_CDR_AUTH_TOKEN")}"
                    },
                    follow_redirects=True,
                )
                res.raise_for_status()

                # Create save directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)

                # Save PDF to file
                with open(os.path.join(save_path, f"{record_id}.pdf"), "wb") as f:
                    f.write(res.content)
        except Exception:
            print(f"Error downloading {record_id}: {res.status_code}")
            return False
    return True


async def download_all_reports(df_hyper: pd.DataFrame, max_concurrent_requests: int):
    """
    Downloads all PDF reports referenced in the provided DataFrame concurrently.

    Args:
        df_hyper: DataFrame containing report record IDs in 'record.value_imputed' column
        max_concurrent_requests: Maximum number of concurrent downloads allowed

    Returns:
        list: List of tuples containing (index, success) for each download attempt
    """
    # Create semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    tasks = []

    # Create download tasks for each record
    for idx, row in df_hyper.iterrows():
        record_id = row[MinModHyperCols.RECORD_VALUE_IMPUTED.value]
        task = asyncio.create_task(
            download_report(record_id, save_path="data/raw/43-101", semaphore=semaphore)
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

    Args:
        max_concurrent_requests: Maximum number of concurrent downloads allowed
    """
    # Load report metadata
    df_hyper = pd.read_csv("./data/raw/minmod/minmod_hyper_response_enriched.csv")

    # Download all reports concurrently
    results = asyncio.run(download_all_reports(df_hyper, max_concurrent_requests))

    # Update download status in DataFrame
    df_hyper["downloaded_pdf"] = False
    for idx, success in results:
        df_hyper.loc[idx, "downloaded_pdf"] = success

    # Print summary and save updated DataFrame
    print(df_hyper["downloaded_pdf"].value_counts())
    df_hyper.to_csv("./data/raw/minmod/minmod_hyper_response_enriched.csv", index=False)


download_hyper_csv()
# download_minmod_site_record_id()
# enrich_hyper_w_record_id()
# download_reports_main()
