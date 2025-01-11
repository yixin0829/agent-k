import os
from datetime import datetime

from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

# Note: Commodity is used for naming files and defining the scope of the data
COMMODITY = "nickel"

# MinMod API endpoint
MINMOD_API_URL = "https://minmod.isi.edu/api/v1"
DEDUP_MINERAL_SITES_ENDPOINT = "/dedup-mineral-sites"

# Read ENV variables
API_USR_NAME: str = os.getenv("API_CDR_USR_NAME")
API_PASSWORD: str = os.getenv("API_CDR_PASSWORD")
AUTH_TOKEN: str = os.getenv("API_CDR_AUTH_TOKEN")

# CDR API
API_CDR_LAND_URL = "https://api.cdr.land/v1"
DOCUMENTS_ENDPOINT = "/docs/documents"
DOCUMENT_BY_ID_ENDPOINT = "/docs/document/{doc_id}"  # for querying pdf document by id
PROVENANCE_ENDPOINT = "/docs/documents/q/provenance/url"  # for querying record id based on source id (e.g. https://w3id.org/usgs/z/4530692/5CAAGFXV)

# Data directories
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
ALL_SOURCES_DIR = os.path.join(RAW_DIR, "all_sources")
CDR_REPORTS_DIR = os.path.join(ALL_SOURCES_DIR, "43-101")
MRDS_DIR = os.path.join(ALL_SOURCES_DIR, "mrds")
DUCKDB_DB_PATH = os.path.join(ALL_SOURCES_DIR, "mrds.duckdb")
ZIP_PATH = os.path.join(RAW_DIR, "mrds.zip")
MRDS_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
GROUND_TRUTH_DIR = os.path.join(RAW_DIR, "ground_truth")
EVAL_DIR = os.path.join(DATA_DIR, "eval")


def hyper_reponse_file(commodity: str):
    """Returns the filename for the hyper response CSV file."""
    return f"minmod_hyper_response_{commodity}.csv"


def enriched_hyper_reponse_file(commodity: str):
    """Returns the filename for the enriched hyper response CSV file."""
    return f"minmod_hyper_response_enriched_{commodity}.csv"


def eval_set_matched_based_file(commodity: str, version: str = "vX"):
    """Returns the filename for the matched-based eval set file."""
    return f"eval_set_matched_based_{commodity}_{version}.jsonl"


def eval_results_file(commodity: str):
    """Returns the filename for the eval results file."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"eval_results_{commodity}_{timestamp}.csv"


MRDS_DTYPE = {
    "dep_id": str,
    "url": str,
    "mrds_id": str,
    "mas_id": str,
    "site_name": str,
    "latitude": float,
    "longitude": float,
    "region": str,
    "country": str,
    "state": str,
    "county": str,
    "com_type": str,
    "commod1": str,
    "commod2": str,
    "commod3": str,
    "oper_type": "category",
    "dep_type": str,
    "prod_size": "category",
    "dev_stat": str,
    "ore": str,
    "gangue": str,
    "other_matl": str,
    "orebody_fm": str,
    "work_type": str,
    "model": str,
    "alteration": str,
    "conc_proc": str,
    "names": str,
    "ore_ctrl": str,
    "reporter": str,
    "hrock_unit": str,
    "hrock_type": str,
    "arock_unit": str,
    "arock_type": str,
    "structure": str,
    "tectonic": str,
    "ref": str,
    "yfp_ba": "category",
    "yr_fst_prd": str,
    "ylp_ba": "category",
    "yr_lst_prd": str,
    "dy_ba": "category",
    "disc_yr": str,
    "prod_yrs": str,
    "discr": str,
    "score": "category",
}

# Autogen settings
AGENT_CACHE_DIR = os.path.join(DATA_DIR, "agent_cache")
OPENAI_MODEL_CLIENT = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

