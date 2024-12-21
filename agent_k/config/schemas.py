from agent_k.utils.listable_class import ListableClass
from enum import Enum


class MinModHyperCols(ListableClass):
    MINERAL_SITE_NAME = "mineral_site_name"
    MINERAL_SITE_TYPE = "mineral_site_type"
    MINERAL_SITE_RANK = "mineral_site_rank"
    COUNTRY = "country"
    STATE_OR_PROVINCE = "state_or_province"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    TOP_1_DEPOSIT_TYPE = "top_1_deposit_type"
    TOP_DEPOSIT_GROUP = "top_deposit_group"
    TOP_1_DEPOSIT_ENVIRONMENT = "top_1_deposit_environment"
    TOP_1_DEPOSIT_CLASSIFICATION_CONFIDENCE = "top_1_deposit_classification_confidence"
    TOTAL_GRADE = "total_grade"  # Unit %
    TOTAL_TONNAGE = "total_tonnage"  # Unit MM tonnes

    # Enriched columns
    DATA_SOURCE = "data_source"  # The source of the data see DataSource enum
    SOURCE_VALUE = "source_value"  # The value of the source
    RECORD_VALUE = "record_value"  # The value of the record
    DOWNLOADED_PDF = "downloaded_pdf"  # Whether the PDF has been downloaded


class DataSource(Enum):
    DOI_ORG = "DOI"
    MRDATA_USGS_GOV_MRDS = "MRDS"
    REPORTS_43_101 = "43-101"
    MRDATA_USGS_GOV = "MRDS_OTHER"
    W3ID_ORG_USGS = "W3ID"
