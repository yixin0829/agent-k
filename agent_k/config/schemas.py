from enum import Enum

from pydantic import BaseModel, Field, create_model


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
    qid: str = Field(default="Not Found", description="Question ID")
    question: str = Field(default="Not Found", description="Question")
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


class InferlinkEvalColumns(Enum):
    ID = "id"
    CDR_RECORD_ID = "cdr_record_id"
    MINERAL_SITE_NAME = "mineral_site_name"
    COUNTRY = "country"
    STATE_OR_PROVINCE = "state_or_province"
    MAIN_COMMODITY = "main_commodity"
    COMMODITY_OBSERVED_NAME = "commodity_observed_name"
    TOTAL_MINERAL_RESOURCE_TONNAGE = "total_mineral_resource_tonnage"
    TOTAL_MINERAL_RESERVE_TONNAGE = "total_mineral_reserve_tonnage"
    TOTAL_MINERAL_RESOURCE_CONTAINED_METAL = "total_mineral_resource_contained_metal"
    TOTAL_MINERAL_RESERVE_CONTAINED_METAL = "total_mineral_reserve_contained_metal"


MINERAL_SITE_NAME_DESCRIPTION = "The name of the mineral site that the report is about."
COUNTRY_DESCRIPTION = "The country where the mineral site is located."
STATE_OR_PROVINCE_DESCRIPTION = (
    "The state or province where the mineral site is located."
)
TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION = """The total mineral resource tonnage of the site is calculated by summing the tonnage of inferred, indicated, and measured mineral resources across all mineral zones. The final value should be converted to tonnes. For example, if we have 1,000 tonnes of inferred mineral resources, 2,000 tonnes of indicated resources, and 3,000 tonnes of measured resources, the total mineral resource tonnage is 1,000 + 2,000 + 3,000 = 6,000 tonnes."""

TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION = """The total mineral reserve tonnage of the site is calculated by summing the tonnage of proven and probable mineral reserves across all mineral zones. The final value should be converted to tonnes. For example, if we have 1,000 tonnes of proven mineral reserves and 2,000 tonnes of probable reserves, the total mineral reserve tonnage is 1,000 + 2,000 = 3,000 tonnes."""

TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION = """The total amount of <main_commodity> metal contained in all the mineral resources converted to tonnes.

1. Calculate the individual contained <main_commodity> metal for each mineral resource (inferred, indicated, and measured) by multiplying the mineral resource tonnage with the corresponding <main_commodity> grade across all the mineral zones.
2. Sum up the individual contained <main_commodity> metal amounts from step 1 to get the total contained <main_commodity> metal.

Example: if the report stated having 1000 tonnes of inferred mineral resources with a <main_commodity> grade of 2%, 2000 tonnes of indicated mineral resources with a <main_commodity> grade of 2.5%, and 3000 tonnes of measured mineral resources with a <main_commodity> grade of 3%, the total contained <main_commodity> metal is 1000 * 2% + 2000 * 2.5% + 3000 * 3% = 20 + 50 + 90 = 160t."""

TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION = """The total amount of <main_commodity> metal contained in all the mineral reserves converted to tonnes.

1. Calculate the individual contained <main_commodity> metal for each mineral reserve (proven and probable) by multiplying the mineral reserve tonnage with the corresponding <main_commodity> grade across all the mineral zones.
2. Sum up the individual contained <main_commodity> metal amounts from step 1 to get the total contained <main_commodity> metal.

Example: if we have 1000 tonnes of proven mineral reserves with a <main_commodity> grade of 2%, 2000 tonnes of probable mineral reserves with a <main_commodity> grade of 2.5%, the total contained <main_commodity> metal is 1000 * 2% + 2000 * 2.5% = 20 + 50 = 70t."""


class MineralSiteMetadata(BaseModel):
    """
    To be extended with more fields based on the mineral inventory of the report
    e.g. total mineral resource tonnage
    """

    # mineral_site_name: str = Field(..., description=MINERAL_SITE_NAME_DESCRIPTION)
    # country: str = Field(
    #     default="Not Found",
    #     description=COUNTRY_DESCRIPTION,
    # )
    # state_or_province: str = Field(
    #     default="Not Found",
    #     description=STATE_OR_PROVINCE_DESCRIPTION,
    # )
    total_mineral_resource_tonnage: float = Field(
        default=0,
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION,
    )
    total_mineral_reserve_tonnage: float = Field(
        default=0,
        description=TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION,
    )
    total_mineral_resource_contained_metal: float = Field(
        default=0,
        description=TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION,
    )
    total_mineral_reserve_contained_metal: float = Field(
        default=0,
        description=TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION,
    )


class MineralSiteMetadataResponseAPI(BaseModel):
    """
    Used for Structured Output with LiteLLM. Default values are not natively supported.
    Explicitly set default values in the description.
    """

    total_mineral_resource_tonnage: float = Field(
        description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION + " Default value: 0",
    )
    total_mineral_reserve_tonnage: float = Field(
        description=TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION + " Default value: 0",
    )
    total_mineral_resource_contained_metal: float = Field(
        description=TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION
        + " Default value: 0",
    )
    total_mineral_reserve_contained_metal: float = Field(
        description=TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION
        + " Default value: 0",
    )


def create_dynamic_mineral_model(commodity: str) -> BaseModel:
    """
    Create a dynamic mineral model based on the commodity. Used for Structured Output
    with LiteLLM. Default values are not natively supported. Explicitly set default
    values in the description.
    """
    model = create_model(
        "MineralSiteComplexProperties",
        total_mineral_resource_tonnage=(
            float,
            Field(
                description=TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION
                + " Default value: 0",
            ),
        ),
        total_mineral_reserve_tonnage=(
            float,
            Field(
                description=TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION
                + " Default value: 0",
            ),
        ),
        total_mineral_resource_contained_metal=(
            float,
            Field(
                description=TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION.replace(
                    "<main_commodity>", commodity
                )
                + " Default value: 0",
            ),
        ),
        total_mineral_reserve_contained_metal=(
            float,
            Field(
                description=TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION.replace(
                    "<main_commodity>", commodity
                )
                + " Default value: 0",
            ),
        ),
    )
    return model
