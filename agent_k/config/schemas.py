from agent_k.utils.listable_class import ListableClass
from enum import Enum


class MinModHyperCols(ListableClass):
    MINERAL_SITE_NAME = "Mineral Site Name"
    MINERAL_SITE_TYPE = "Mineral Site Type"
    MINERAL_SITE_RANK = "Mineral Site Rank"
    COUNTRY = "Country"
    STATE_OR_PROVINCE = "State/Province"
    LATITUDE = "Latitude"
    LONGITUDE = "Longitude"
    TOP_1_DEPOSIT_TYPE = "Top 1 Deposit Type"
    TOP_DEPOSIT_GROUP = "Top Deposit Group"
    TOP_1_DEPOSIT_ENVIRONMENT = "Top 1 Deposit Environment"
    TOP_1_DEPOSIT_CLASSIFICATION_CONFIDENCE = "Top 1 Deposit Classification Confidence"
    TOTAL_GRADE = "Total Grade"  # Unit %
    TOTAL_TONNAGE = "Total Tonnage"  # Unit MM tonnes

    # Enrich columns
    DATA_SOURCE = "Data Source"  # The source of the data see DataSource enum
    SOURCE_VALUE = "Source Value"  # The value of the source
    RECORD_VALUE = "Record Value"  # The value of the record
    DOWNLOADED_PDF = "Downloaded PDF"  # Whether the PDF has been downloaded


class DataSource(Enum):
    DOI_ORG = "DOI"
    MRDATA_USGS_GOV_MRDS = "MRDS"
    API_CDR_LAND = "43-101"
    MRDATA_USGS_GOV = "MRDS_OTHER"
    W3ID_ORG_USGS = "W3ID"
