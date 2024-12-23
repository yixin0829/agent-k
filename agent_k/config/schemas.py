from enum import Enum

from agent_k.utils.listable_class import ListableClass


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
    """
    The names of the data sources are used for regex matching. Do not change them.
    """

    OTHER = "MRDATA_OTHER"
    DOI_ORG = "DOI"
    MRDATA_USGS_GOV_MRDS = "MRDATA_MRDS"
    API_CDR_LAND = "43-101"
    W3ID_ORG_USGS = "W3ID"


class QATemplateType(Enum):
    SINGLE_STATE_OR_PROVINCE = "single_state_or_province"
    SINGLE_COUNTRY = "single_country"
    SINGLE_DEPOSIT_TYPE = "single_deposit_type"
    SINGLE_DEPOSIT_ENVIRONMENT = "single_deposit_environment"
    MULTIPLE_STATE_OR_PROVINCE = "multiple_state_or_province"
    MULTIPLE_COUNTRY = "multiple_country"
    MULTIPLE_DEPOSIT_TYPE = "multiple_deposit_type"
    MULTIPLE_DEPOSIT_ENVIRONMENT = "multiple_deposit_environment"
