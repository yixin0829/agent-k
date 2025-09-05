from enum import Enum

from pydantic import BaseModel, Field, create_model


class MineralEvalDfColumns(Enum):
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


TOTAL_MINERAL_RESOURCE_TONNAGE_DESCRIPTION = """Total mineral resource tonnage, calculated as the sum of Measured, Indicated, and Inferred Mineral Resources (t). Each category represents a different level of geological confidence per CIM Definition Standards (2014). If a category is not reported, it is treated as 0. The final value should be converted to tonnes."""


TOTAL_MINERAL_RESERVE_TONNAGE_DESCRIPTION = """Total mineral reserve tonnage (t), calculated as the sum of Proven and Probable Mineral Reserves. Each category represents a different level of geological confidence per CIM Definition Standards (2014). If a category is not reported, it is treated as 0. The final value should be converted to tonnes."""


TOTAL_MINERAL_RESOURCE_CONTAINED_METAL_DESCRIPTION = """The total amount of <main_commodity> metal contained in all the mineral resources converted to tonnes per CIM Definition Standards (2014).

1. Calculate the individual contained <main_commodity> metal for inferred, indicated, and measured mineral resources by multiplying the mineral resource tonnage with the corresponding <main_commodity> grade across all the mineral zones if they are present.
2. Sum up the individual contained <main_commodity> metal amounts from step 1 to get the total contained <main_commodity> metal."""

TOTAL_MINERAL_RESERVE_CONTAINED_METAL_DESCRIPTION = """The total amount of <main_commodity> metal contained in all the mineral reserves converted to tonnes per CIM Definition Standards (2014).

1. Calculate the individual contained <main_commodity> metal for proven and probable mineral reserve by multiplying the mineral reserve tonnage with the corresponding <main_commodity> grade across all the mineral zones if they are present.
2. Sum up the individual contained <main_commodity> metal amounts from step 1 to get the total contained <main_commodity> metal."""


class MineralSiteMetadata(BaseModel):
    """
    To be extended with more fields based on the mineral inventory of the report
    e.g. total mineral resource tonnage
    """

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


def create_mineral_model_w_commodity(commodity: str) -> BaseModel:
    """
    Create a dynamic mineral model based on the input commodity. Default values are not
    natively supported. Explicitly set default values in the description.
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
