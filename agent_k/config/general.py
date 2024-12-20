import os
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
