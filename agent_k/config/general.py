import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# MinMod API endpoint
API_ENDPOINT = "https://minmod.isi.edu/api/v1"

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
ZIP_PATH = os.path.join(RAW_DIR, "mrds.zip")
MRDS_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
MINMOD_DIR = os.path.join(RAW_DIR, "minmod")
EVAL_DIR = os.path.join(DATA_DIR, "eval")

# Note: Commodity is used for naming files and defining the scope of the data
COMMODITY = "nickel"


def hyper_reponse_file(commodity: str):
    """Returns the filename for the hyper response CSV file."""
    return f"minmod_hyper_response_{commodity}.csv"


def enriched_hyper_reponse_file(commodity: str):
    """Returns the filename for the enriched hyper response CSV file."""
    return f"minmod_hyper_response_enriched_{commodity}.csv"


def eval_set_matched_based_file(commodity: str):
    """Returns the filename for the matched-based eval set file."""
    return f"eval_set_matched_based_{commodity}.jsonl"


def eval_results_file(commodity: str):
    """Returns the filename for the eval results file."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"eval_results_{commodity}_{timestamp}.csv"


mrds_dtype = {
    "dep_id": str,
    "url": str,
    "mrds_id": str,
    "mas_id": str,
    "site_name": str,
    "latitude": float,
    "longitude": float,
    "region": "category",
    "country": "category",
    "state": "category",
    "county": "category",
    "com_type": "category",
    "commod1": "category",
    "commod2": "category",
    "commod3": "category",
    "oper_type": "category",
    "dep_type": "category",
    "prod_size": "category",
    "dev_stat": "category",
    "ore": "category",
    "gangue": "category",
    "other_matl": "category",
    "orebody_fm": "category",
    "work_type": "category",
    "model": "category",
    "alteration": "category",
    "conc_proc": "category",
    "names": "category",
    "ore_ctrl": "category",
    "reporter": "category",
    "hrock_unit": "category",
    "hrock_type": "category",
    "arock_unit": "category",
    "arock_type": "category",
    "structure": "category",
    "tectonic": "category",
    "ref": "category",
    "yfp_ba": "category",
    "yr_fst_prd": "category",
    "ylp_ba": "category",
    "yr_lst_prd": "category",
    "dy_ba": "category",
    "disc_yr": "category",
    "prod_yrs": "category",
    "discr": "category",
    "score": "category",
}

# Agent cache directory
AGENT_CACHE_DIR = os.path.join(DATA_DIR, "agent_cache")

# Autogen config
AUTOGEN_CONFIG_LIST = [
    {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]
