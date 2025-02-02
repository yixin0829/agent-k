from enum import Enum

from pydantic import BaseModel, Field


class MinModHyperCols(Enum):
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


class MinModEntity(BaseModel):
    entity_name: str = Field(..., description="The name of the entity")
    entity_description: str = Field(..., description="The description of the entity")
    entity_data_type: str = Field(..., description="python data type of the entity")


class RelevantEntities(BaseModel):
    entities: list[MinModEntity] = Field(
        ..., description="The relevant entities needed to answer a given question"
    )


class RelevantEntitiesPredefined(BaseModel):
    mineral_site_name: str = Field(
        ..., description="The name of the mineral site that the report is about"
    )
    state_or_province: str = Field(
        ..., description="The state or province where the mineral site is located"
    )
    country: str = Field(
        ..., description="The country where the mineral site is located"
    )
    total_grade: float = Field(
        "Not Found",
        description="The total grade of all the nickel deposits in decimal format",
    )
    total_tonnage: float = Field(
        "Not Found",
        description="The total tonnage of all the nickel deposits in million tonnes",
    )
    top_1_deposit_type: str = Field(
        "Not Found", description="The most likely deposit type of the mineral site"
    )
    top_1_deposit_environment: str = Field(
        "Not Found",
        description="The most likely deposit environment of the mineral site",
    )


class EvalReport(BaseModel):
    qid: str = Field(default="Unknown", description="Question ID")
    question: str = Field(default="Unknown", description="Question")
    row_em_score: float = Field(default=0, description="Exact match score for all rows")
    row_precision: float = Field(default=0, description="Precision score for all rows")
    row_recall: float = Field(default=0, description="Recall score for all rows")
    row_f1: float = Field(default=0, description="F1 score for all rows")
    ms_em_score: float = Field(
        default=0, description="Exact match score for mineral site name"
    )
    ms_precision: float = Field(
        default=0, description="Precision score for mineral site name"
    )
    ms_recall: float = Field(
        default=0, description="Recall score for mineral site name"
    )
    ms_f1: float = Field(default=0, description="F1 score for mineral site name")

    def __str__(self) -> str:
        return f"EM: {self.row_em_score:.2f}, Precision: {self.row_precision:.2f}, Recall: {self.row_recall:.2f}, F1: {self.row_f1:.2f}, MS EM: {self.ms_em_score:.2f}, MS Precision: {self.ms_precision:.2f}, MS Recall: {self.ms_recall:.2f}, MS F1: {self.ms_f1:.2f}"

    def to_dict(self) -> dict:
        return self.model_dump()
