from enum import Enum

from agent_k.utils.listable_class import ListableClass


class MinModHyperCols(ListableClass):
    MINERAL_SITE_URI = "mineral_site_uri"
    MINERAL_SITE_NAME = "mineral_site_name"
    MINERAL_SITE_TYPE = "mineral_site_type"
    MINERAL_SITE_RANK = "mineral_site_rank"
    SITES = "sites"
    COUNTRY = "country"
    STATE_OR_PROVINCE = "state_or_province"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    TOP_1_DEPOSIT_TYPE = "top_1_deposit_type"
    TOP_1_DEPOSIT_GROUP = "top_1_deposit_group"
    TOP_1_DEPOSIT_ENVIRONMENT = "top_1_deposit_environment"
    TOP_1_DEPOSIT_CLASSIFICATION_CONFIDENCE = "top_1_deposit_classification_confidence"
    TOP_1_DEPOSIT_CLASSIFICATION_SOURCE = "top_1_deposit_classification_source"
    TOTAL_GRADE = "total_grade"
    TOTAL_TONNAGE = "total_tonnage"

    # Enriched columns
    DATA_SOURCE = "data_source"  # The source of the data see DataSource enum
    RECORD_VALUE = "record_value"  # The value of the record
    DOWNLOADED_PDF = "downloaded_pdf"  # Whether the PDF has been downloaded


class DataSource(Enum):
    """
    Data sources used for mineral site (MS) extraction.
    Note: The names of the data sources are used for regex matching. Do not change them.
    """

    DOI_ORG = "DOI"  # MS extracted from academic papers
    MRDATA_USGS_GOV = "MRDATA_OTHER"  # MS extracted from other datasets in USGS
    MRDATA_USGS_GOV_MRDS = "MRDATA_MRDS"  # MS extracted from MRDS
    API_CDR_LAND = "43-101"  # MS extracted from 43-101 reports using CDR Land
    W3ID_ORG_USGS = "W3ID"  # MS extracted from 43-101 reports Zotero (old, ignore)
    OTHER = "OTHER"  # MS extracted from other sources


class QATemplateType(Enum):
    SINGLE_STATE_OR_PROVINCE = "single_state_or_province"
    SINGLE_COUNTRY = "single_country"
    SINGLE_DEPOSIT_TYPE = "single_deposit_type"
    SINGLE_DEPOSIT_ENVIRONMENT = "single_deposit_environment"
    MULTIPLE_STATE_OR_PROVINCE = "multiple_state_or_province"
    MULTIPLE_COUNTRY = "multiple_country"
    MULTIPLE_DEPOSIT_TYPE = "multiple_deposit_type"
    MULTIPLE_DEPOSIT_ENVIRONMENT = "multiple_deposit_environment"
