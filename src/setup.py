# %% [markdown]
# # Construct NI 43-101 Mineral Report Eval Dataset

# %%
import glob
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from docling.document_converter import DocumentConverter
from pydantic import BaseModel

from src.config.logger import logger
from src.config.schemas import MineralEvalDfColumns

NI_43_101_ANNOTATIONS_DIR = "data/raw/43-101_annotations"
NI_43_101_GROUND_TRUTH_DIR = "data/processed/43-101_ground_truth"
NI_43_101_GROUND_TRUTH_FILE = os.path.join(
    NI_43_101_GROUND_TRUTH_DIR, "43-101_ground_truth.csv"
)


# %%
"""
Read all metadata.csv files in the specified directory. Enrich the dataframe with the cdr_record_id.
"""

# Find all metadata.csv files recursively
metadata_files = sorted(
    glob.glob(f"{NI_43_101_ANNOTATIONS_DIR}/**/metadata.csv", recursive=True)
)

logger.info(f"Found {len(metadata_files)} metadata.csv files")

master_metadata_df = pd.DataFrame()

# Process each file
for file_path in metadata_files:
    try:
        # Get the parent directory name (i.e. the cdr_record_id)
        parent_dir = os.path.basename(os.path.dirname(file_path))

        df_metadata = pd.read_csv(file_path)
        df_metadata.insert(0, MineralEvalDfColumns.CDR_RECORD_ID.value, parent_dir)
        master_metadata_df = pd.concat(
            [master_metadata_df, df_metadata], ignore_index=True
        )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

logger.info("Processing master metadata dataframe complete!")

# %%
"""
Read all mineral_inventory_minimal.csv files in the specified directory. Enrich the dataframe with the cdr_record_id.
"""
# Find all mineral_inventory_minimal.csv files recursively
inventory_files = sorted(
    glob.glob(
        f"{NI_43_101_ANNOTATIONS_DIR}/**/mineral_inventory_minimal.csv",
        recursive=True,
    )
)

logger.info(f"Found {len(inventory_files)} mineral_inventory_minimal.csv files")

master_inventory_df = pd.DataFrame()

# Process each file
for file_path in inventory_files:
    try:
        # Get the parent directory name (which will be the cdr_record_id)
        parent_dir = os.path.basename(os.path.dirname(file_path))
        parent_parent_dir = os.path.basename(
            os.path.dirname(os.path.dirname(file_path))
        )

        # Read the CSV file
        df_inventory = pd.read_csv(file_path)

        # Insert the new column at the beginning
        df_inventory.insert(0, MineralEvalDfColumns.CDR_RECORD_ID.value, parent_dir)
        df_inventory.insert(
            1, MineralEvalDfColumns.MAIN_COMMODITY.value, parent_parent_dir
        )

        # If the df is empty, append a placeholder row with cdr_record_id
        if df_inventory.empty:
            df_inventory = pd.DataFrame(
                {
                    MineralEvalDfColumns.CDR_RECORD_ID.value: [parent_dir],
                    MineralEvalDfColumns.MAIN_COMMODITY.value: [parent_parent_dir],
                }
            )

        # Append to master inventory dataframe
        master_inventory_df = pd.concat(
            [master_inventory_df, df_inventory], ignore_index=True
        )

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

master_inventory_df[MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value] = (
    master_inventory_df[MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value]
    .str.lower()
    .str.strip()
)

logger.info("Processing master inventory dataframe complete!")


# %%
def normalize_ore_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ore units to million tonnes. Creates two new columns:
    - normalized_ore_value: The ore value converted to million tonnes
    - normalized_ore_unit: Set to 'Mt' (million tonnes)

    Args:
        df (pd.DataFrame): DataFrame containing ore values and units

    Returns:
        pd.DataFrame: DataFrame with added normalization columns
    """
    # Create copies of the original columns
    df["normalized_ore_value"] = df["ore_value"].copy()
    df["normalized_ore_unit"] = df["ore_unit_observed_name"].copy()

    # Define conversion factors to million tonnes (Mt)
    conversion_factors = {
        "mt": 1.0,  # Million tonnes
        "million tonnes": 1.0,  # Million tonnes (spelled out)
        "gt": 1000.0,  # Billion tonnes (Giga tonnes)
        "kt": 0.001,  # Thousand tonnes (kilo tonnes)
        "thousand tonnes": 0.001,  # Thousand tonnes (spelled out)
        "t": 0.000001,  # Tonnes
        "tonnes": 0.000001,  # Tonnes (spelled out)
        "tonnage": 0.000001,  # Tonnage (assuming equivalent to tonnes)
        "metric tons": 0.000001,  # Metric tons
        "tons": 0.0000009072,  # Tons (assuming short tons)
        "ton": 0.0000009072,  # Ton (assuming short tons)
        "short tons": 0.0000009072,  # Short tons (1 short ton = 0.9072 metric tonnes)
    }

    # Apply conversion
    for idx, row in df.iterrows():
        try:
            if pd.notna(row["ore_unit_observed_name"]) and pd.notna(row["ore_value"]):
                unit = (
                    row["ore_unit_observed_name"].strip().lower()
                )  # Convert to lowercase for case-insensitive matching
                if unit in conversion_factors:
                    # Convert the value to million tonnes
                    df.at[idx, "normalized_ore_value"] = (
                        float(row["ore_value"]) * conversion_factors[unit]
                    )
                    df.at[idx, "normalized_ore_unit"] = "Mt"
                else:
                    # If unit not recognized raise an error
                    raise ValueError(
                        f"Unit '{row['ore_unit_observed_name']}' not recognized"
                    )
        except Exception as e:
            logger.error(f"Error normalizing ore units for row {idx}: {e}")
            # Keep original values if there's an error

    return df


def normalize_grade_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize grade units to percent. Creates two new columns:
    - normalized_grade_value: The grade value converted to percent
    - normalized_grade_unit: Set to '%' (percent)

    Args:
        df (pd.DataFrame): DataFrame containing grade values and units

    Returns:
        pd.DataFrame: DataFrame with added normalization columns
    """
    # Create copies of the original columns
    df["normalized_grade_value"] = df["grade_value"].copy()
    df["normalized_grade_unit"] = df["grade_unit_observed_name"].copy()

    # Define conversion factors to percent (%)
    conversion_factors = {
        "%": 1.0,  # Already in percent
        "percent": 1.0,  # Spelled out percent
        "g/t": 0.0001,  # Grams per tonne (1 g/t = 0.0001%)
        "g/mt": 0.0001,  # Grams per metric tonne
        "g/tonne": 0.0001,  # Grams per tonne (spelled out)
        "grams/tonne": 0.0001,  # Grams per tonne (fully spelled)
        "ppm": 0.0001,  # Parts per million (1 ppm = 0.0001%)
        "parts per million": 0.0001,  # Parts per million (spelled out)
        "oz/t": 0.00343,  # Troy ounces per short ton (1 oz/t ≈ 0.00343%)
        "oz/ton": 0.00343,  # Troy ounces per ton
        "opt": 0.00343,  # Ounces per ton abbreviation
        "kg/t": 0.1,  # Kilograms per tonne (1 kg/t = 0.1%)
        "ppb": 0.0000001,  # Parts per billion (1 ppb = 0.0000001%)
        "wt%": 1.0,  # Weight percent
        "wt.%": 1.0,  # Weight percent (alternative notation)
        "grams per tonne": 0.0001,  # Grams per tonne (spelled out)
        "gram per tonne": 0.0001,  # Grams per tonne (spelled out)
    }

    # Apply conversion
    for idx, row in df.iterrows():
        try:
            if pd.notna(row["grade_unit_observed_name"]) and pd.notna(
                row["grade_value"]
            ):
                unit = (
                    row["grade_unit_observed_name"].strip().lower()
                )  # Convert to lowercase for case-insensitive matching
                if unit in conversion_factors:
                    # Convert the value to percent
                    df.at[idx, "normalized_grade_value"] = (
                        float(row["grade_value"]) * conversion_factors[unit]
                    )
                    df.at[idx, "normalized_grade_unit"] = "%"
                else:
                    # If unit not recognized raise an error
                    raise ValueError(
                        f"Grade unit '{row['grade_unit_observed_name']}' not recognized"
                    )
        except Exception as e:
            logger.error(f"Error normalizing grade units for row {idx}: {e}")
            # Keep original values if there's an error

    return df


master_inventory_df = normalize_ore_units(master_inventory_df)
master_inventory_df = normalize_grade_units(master_inventory_df)

# Assert there are no values in the "zone" column contain "total" to avoid double counting
assert not master_inventory_df["zone"].str.lower().str.contains("total").any()

# %%
category_to_resource_or_reserve = {
    "inferred": "resource",
    "measured": "resource",
    "indicated": "resource",
    "mineral resource": "resource",
    "measured+indicated": "resource",
    "proven+probable": "reserve",
    "proved": "reserve",
    "probable": "reserve",
    "proven": "reserve",
    "mineralresource": "resource",
}
# remove any whitespace from category (e.g. "proven + probable" -> "proven+probable")
master_inventory_df["category_observed_name"] = (
    master_inventory_df["category_observed_name"].str.replace(" ", "").str.lower()
)

master_inventory_df.insert(
    4,
    "resource_or_reserve",
    master_inventory_df["category_observed_name"].map(category_to_resource_or_reserve),
)
# Assert for all rows have "category_observed_name" it has a corresponding "resource_or_reserve" value. If not, raise an error.
assert (
    not master_inventory_df[
        master_inventory_df["category_observed_name"].notna()
        & master_inventory_df["resource_or_reserve"].isna()
    ]
    .any()
    .any()
)


master_inventory_df["contained_metal"] = (
    master_inventory_df["normalized_ore_value"]
    * master_inventory_df["normalized_grade_value"]
    / 100
)

# %% [markdown]
# ## Site-Level Ore and Contained Metal

# %%
selected_columns = [
    MineralEvalDfColumns.CDR_RECORD_ID.value,
    MineralEvalDfColumns.MAIN_COMMODITY.value,
    MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value,
    "resource_or_reserve",
    "normalized_ore_value",
    "contained_metal",
]


# Note: mineral site with no inventory data will have a NaN value in the "category_observed_name" column
# and thus a NaN value in the "resource_or_reserve" column. These rows will be dropped during the groupby operation.

# Group by cdr_record_id, resource_or_reserve and sum the normalized_ore_value
resource_n_reserve_total_tonnage = (
    master_inventory_df[selected_columns]
    .groupby(
        [
            MineralEvalDfColumns.CDR_RECORD_ID.value,
            MineralEvalDfColumns.MAIN_COMMODITY.value,
            MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value,
            "resource_or_reserve",
        ]
    )
    .agg(
        {
            "normalized_ore_value": "sum",
            "contained_metal": "sum",
        }
    )
    .reset_index()
)
resource_n_reserve_total_tonnage.head()

# %%
# pivot the resource_n_reserve column to wide format
resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage.pivot(
    index=[
        MineralEvalDfColumns.CDR_RECORD_ID.value,
        MineralEvalDfColumns.MAIN_COMMODITY.value,
        MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value,
    ],
    columns="resource_or_reserve",
    values=["normalized_ore_value", "contained_metal"],
)
# expand the index to separate columns
resource_n_reserve_total_tonnage.columns = [
    f"{col[0]}_{col[1]}" for col in resource_n_reserve_total_tonnage.columns
]
resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage.reset_index()
resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage[
    [
        MineralEvalDfColumns.CDR_RECORD_ID.value,
        MineralEvalDfColumns.MAIN_COMMODITY.value,
        MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value,
        "normalized_ore_value_resource",
        "normalized_ore_value_reserve",
        "contained_metal_resource",
        "contained_metal_reserve",
    ]
]
resource_n_reserve_total_tonnage.head()

# %%
# Left join master_metadata_df and master_inventory_df on cdr_record_id
master_df = pd.merge(
    resource_n_reserve_total_tonnage,
    master_metadata_df,
    on=MineralEvalDfColumns.CDR_RECORD_ID.value,
    how="left",
)

cols_to_drop = [
    "authors",
    "year",
    "month",
]
master_df = master_df.drop(columns=cols_to_drop)

cols_to_rename = {
    "normalized_ore_value_resource": MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
    "normalized_ore_value_reserve": MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
    "contained_metal_resource": MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
    "contained_metal_reserve": MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    "country_observed_name": MineralEvalDfColumns.COUNTRY.value,
    "state_or_province_observed_name": MineralEvalDfColumns.STATE_OR_PROVINCE.value,
    "mining_name": MineralEvalDfColumns.MINERAL_SITE_NAME.value,
}
master_df = master_df.rename(columns=cols_to_rename)

# Add unique identifier at the first column
master_df.insert(0, MineralEvalDfColumns.ID.value, range(1, len(master_df) + 1))

# Map "mvt_zinc" to "zinc" for the "main_commodity" column
master_df[MineralEvalDfColumns.MAIN_COMMODITY.value] = master_df[
    MineralEvalDfColumns.MAIN_COMMODITY.value
].str.replace("mvt_", "")

# Fill numerical columns with 0 if they are NaN
numerical_columns = master_df.select_dtypes(include=[np.number]).columns
master_df[numerical_columns] = master_df[numerical_columns].fillna(0)

logger.info(f"Master dataframe shape: {master_df.shape}")

# %%
# Calculate the max of total_mineral_resource_tonnage and total_mineral_reserve_tonnage for each cdr_record_id
master_df["total_mineral_resource_tonnage_max"] = master_df.groupby(
    MineralEvalDfColumns.CDR_RECORD_ID.value
)[MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value].transform("max")

master_df["total_mineral_reserve_tonnage_max"] = master_df.groupby(
    MineralEvalDfColumns.CDR_RECORD_ID.value
)[MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value].transform("max")


# %%
# Filter for rows where main_commodity equals commodity_observed_name or main_commodity == "earth_metals"
master_df = master_df[
    (
        master_df[MineralEvalDfColumns.MAIN_COMMODITY.value]
        == master_df[MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value]
    )
    | (master_df[MineralEvalDfColumns.MAIN_COMMODITY.value] == "earth_metals")
]

# For each cdr_record_id group, take the first row
master_df = (
    master_df.groupby(MineralEvalDfColumns.CDR_RECORD_ID.value).first().reset_index()
)

# %%
os.makedirs(NI_43_101_GROUND_TRUTH_DIR, exist_ok=True)
master_df.to_csv(NI_43_101_GROUND_TRUTH_FILE, index=False)

# %% [markdown]
# ### Dev-Test Split
#

# %%
np.random.seed(1)

# Read the dataframe
df = pd.read_csv(NI_43_101_GROUND_TRUTH_FILE)

# Get unique cdr_record_ids
unique_cdr_ids = df[MineralEvalDfColumns.CDR_RECORD_ID.value].unique()

# Randomly sample 80% of unique cdr_record_ids for test set
test_cdr_ids = np.random.choice(
    unique_cdr_ids, size=int(len(unique_cdr_ids) * 0.8), replace=False
)

# Create test and dev sets
test_df = df[df[MineralEvalDfColumns.CDR_RECORD_ID.value].isin(test_cdr_ids)].copy()
dev_df = df[~df[MineralEvalDfColumns.CDR_RECORD_ID.value].isin(test_cdr_ids)].copy()

# Sort by ID to maintain consistent ordering
test_df = test_df.sort_values(MineralEvalDfColumns.ID.value)
dev_df = dev_df.sort_values(MineralEvalDfColumns.ID.value)

# Save the splits
test_df.to_csv(
    "data/processed/43-101_ground_truth/43-101_ground_truth_test.csv", index=False
)
dev_df.to_csv(
    "data/processed/43-101_ground_truth/43-101_ground_truth_dev.csv", index=False
)

logger.info(
    f"Test set size: {len(test_df)} | Unique CDR Report count: {test_df[MineralEvalDfColumns.CDR_RECORD_ID.value].nunique()}"
)
logger.info(
    f"Dev set size: {len(dev_df)} | Unique CDR Report count: {dev_df[MineralEvalDfColumns.CDR_RECORD_ID.value].nunique()}"
)
logger.info(
    f"Combined set size: {len(df)} | Unique CDR Report count: {df[MineralEvalDfColumns.CDR_RECORD_ID.value].nunique()}"
)

# %% [markdown]
# ## JSONify

# %%


class ContainedMetalBreakdown(BaseModel):
    category_observed_name: str
    resource_or_reserve: str
    normalized_ore_value: float = 0
    normalized_grade: float = 0


class ContainedMetal(BaseModel):
    commodity_observed_name: str
    resource_contained_metal: float = 0
    reserve_contained_metal: float = 0
    resource_contained_metal_breakdown: list[ContainedMetalBreakdown] = []
    reserve_contained_metal_breakdown: list[ContainedMetalBreakdown] = []


class TonnageBreakdown(BaseModel):
    category_observed_name: str
    resource_or_reserve: str
    normalized_ore_value: float = 0


class MineralReportGroundTruth(BaseModel):
    cdr_record_id: str
    mineral_site_name: Optional[str] = None
    country: Optional[str] = None
    state_or_province: Optional[str] = None
    main_commodity: str
    total_mineral_resource_tonnage: float = 0
    total_mineral_reserve_tonnage: float = 0
    total_mineral_resource_tonnage_breakdown: list[TonnageBreakdown] = []
    total_mineral_reserve_tonnage_breakdown: list[TonnageBreakdown] = []
    contained_metal_inventory: list[ContainedMetal] = []


# Create a dictionary to store the ground truth data
mineral_report_ground_truth = {}

# Iterate through df rows to populate the ground truth
for _, row in df.iterrows():
    cdr_record_id = row[MineralEvalDfColumns.CDR_RECORD_ID.value]

    # Initialize the record if it doesn't exist (mineral site level)
    if cdr_record_id not in mineral_report_ground_truth:
        mineral_report_ground_truth[cdr_record_id] = MineralReportGroundTruth(
            cdr_record_id=cdr_record_id,
            mineral_site_name=row[MineralEvalDfColumns.MINERAL_SITE_NAME.value]
            if pd.notna(row[MineralEvalDfColumns.MINERAL_SITE_NAME.value])
            else None,
            country=row[MineralEvalDfColumns.COUNTRY.value]
            if pd.notna(row[MineralEvalDfColumns.COUNTRY.value])
            else None,
            state_or_province=row[MineralEvalDfColumns.STATE_OR_PROVINCE.value]
            if pd.notna(row[MineralEvalDfColumns.STATE_OR_PROVINCE.value])
            else None,
            main_commodity=row[MineralEvalDfColumns.MAIN_COMMODITY.value],
            total_mineral_resource_tonnage=row[
                MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value
            ]
            if pd.notna(row[MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value])
            else 0,
            total_mineral_reserve_tonnage=row[
                MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value
            ]
            if pd.notna(row[MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value])
            else 0,
            contained_metal_inventory=[],
        )

        # Get mineral inventory tonnage breakdown
        mineral_inventory = master_inventory_df[
            (
                master_inventory_df[MineralEvalDfColumns.CDR_RECORD_ID.value]
                == cdr_record_id
            )
        ]
        mineral_inventory_tonnage = mineral_inventory[
            [
                "category_observed_name",
                "resource_or_reserve",
                "normalized_ore_value",
            ]
        ].drop_duplicates()

        for _, row_mineral_inventory_tonnage in mineral_inventory_tonnage.iterrows():
            category_observed_name = row_mineral_inventory_tonnage[
                "category_observed_name"
            ]
            resource_or_reserve = row_mineral_inventory_tonnage["resource_or_reserve"]
            normalized_ore_value = row_mineral_inventory_tonnage["normalized_ore_value"]

            if resource_or_reserve == "resource":
                mineral_report_ground_truth[
                    cdr_record_id
                ].total_mineral_resource_tonnage_breakdown.append(
                    TonnageBreakdown(
                        category_observed_name=category_observed_name,
                        resource_or_reserve=resource_or_reserve,
                        normalized_ore_value=normalized_ore_value,
                    )
                )
            else:
                mineral_report_ground_truth[
                    cdr_record_id
                ].total_mineral_reserve_tonnage_breakdown.append(
                    TonnageBreakdown(
                        category_observed_name=category_observed_name,
                        resource_or_reserve=resource_or_reserve,
                        normalized_ore_value=normalized_ore_value,
                    )
                )

    # Add contained metal information
    commodity_name = row[MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value]
    resource_contained_metal = (
        row[MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value]
        if pd.notna(
            row[MineralEvalDfColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value]
        )
        else 0
    )
    reserve_contained_metal = (
        row[MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value]
        if pd.notna(
            row[MineralEvalDfColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value]
        )
        else 0
    )

    contained_metal = ContainedMetal(
        commodity_observed_name=commodity_name,
        resource_contained_metal=resource_contained_metal,
        reserve_contained_metal=reserve_contained_metal,
    )
    mineral_report_ground_truth[cdr_record_id].contained_metal_inventory.append(
        contained_metal
    )

    # Add contained metal breakdown for the current commodity
    mineral_inventory = master_inventory_df[
        (master_inventory_df[MineralEvalDfColumns.CDR_RECORD_ID.value] == cdr_record_id)
        & (
            master_inventory_df[MineralEvalDfColumns.COMMODITY_OBSERVED_NAME.value]
            == commodity_name
        )
    ]
    mineral_inventory_contained_metal = mineral_inventory[
        [
            "category_observed_name",
            "resource_or_reserve",
            "normalized_ore_value",
            "normalized_grade_value",
        ]
    ].drop_duplicates()

    for (
        _,
        row_mineral_inventory_contained_metal,
    ) in mineral_inventory_contained_metal.iterrows():
        category_observed_name = row_mineral_inventory_contained_metal[
            "category_observed_name"
        ]
        resource_or_reserve = row_mineral_inventory_contained_metal[
            "resource_or_reserve"
        ]
        normalized_ore_value = row_mineral_inventory_contained_metal[
            "normalized_ore_value"
        ]
        normalized_grade = row_mineral_inventory_contained_metal[
            "normalized_grade_value"
        ]

        if resource_or_reserve == "resource":
            mineral_report_ground_truth[cdr_record_id].contained_metal_inventory[
                -1
            ].resource_contained_metal_breakdown.append(
                ContainedMetalBreakdown(
                    category_observed_name=category_observed_name,
                    resource_or_reserve=resource_or_reserve,
                    normalized_ore_value=normalized_ore_value,
                    normalized_grade=normalized_grade,
                )
            )
        else:
            mineral_report_ground_truth[cdr_record_id].contained_metal_inventory[
                -1
            ].reserve_contained_metal_breakdown.append(
                ContainedMetalBreakdown(
                    category_observed_name=category_observed_name,
                    resource_or_reserve=resource_or_reserve,
                    normalized_ore_value=normalized_ore_value,
                    normalized_grade=normalized_grade,
                )
            )


# %%
# Split into test and dev sets
test_ground_truth = {}
for cdr_id, data in mineral_report_ground_truth.items():
    if cdr_id in test_cdr_ids:
        test_ground_truth[cdr_id] = data

dev_ground_truth = {}
for cdr_id, data in mineral_report_ground_truth.items():
    if cdr_id not in test_cdr_ids:
        dev_ground_truth[cdr_id] = data

logger.info(f"Test set: {len(test_ground_truth)} CDR records")
logger.info(f"Dev set: {len(dev_ground_truth)} CDR records")
logger.info(f"Total: {len(mineral_report_ground_truth)} CDR records")

# Model dump the dict values
mineral_report_ground_truth = {
    cdr_id: data.model_dump() for cdr_id, data in mineral_report_ground_truth.items()
}
test_ground_truth = {
    cdr_id: data.model_dump() for cdr_id, data in test_ground_truth.items()
}
dev_ground_truth = {
    cdr_id: data.model_dump() for cdr_id, data in dev_ground_truth.items()
}

# Save the ground truth data
with open("data/processed/43-101_ground_truth/43-101_ground_truth.json", "w") as f:
    json.dump(mineral_report_ground_truth, f, indent=2)

with open("data/processed/43-101_ground_truth/43-101_ground_truth_test.json", "w") as f:
    json.dump(test_ground_truth, f, indent=2)

with open("data/processed/43-101_ground_truth/43-101_ground_truth_dev.json", "w") as f:
    json.dump(dev_ground_truth, f, indent=2)


# %% [markdown]
# # Parse using Docling

# %%

df_mineral_gt = pd.read_csv(NI_43_101_GROUND_TRUTH_FILE)
record_ids = list(set(df_mineral_gt[MineralEvalDfColumns.CDR_RECORD_ID.value].tolist()))

converter = DocumentConverter()
os.makedirs("data/processed/43-101_reports", exist_ok=True)

# Get all PDF files, and process them in one loop
files = os.listdir("data/raw/43-101_reports")

# Process files
for i, file in enumerate(files):
    pdf_path = os.path.join("data/raw/43-101_reports", file)
    record_id = file.replace(".pdf", "")

    logger.info(f"{i + 1}/{len(files)}: Processing {pdf_path}")

    # Check if the markdown file already exists in "data/processed/43-101_reports". If so, skip
    if os.path.exists(os.path.join("data/processed/43-101_reports", f"{record_id}.md")):
        logger.info(
            "Skipping PDF to Markdown conversion because the markdown file already exists"
        )
        continue

    result = converter.convert(pdf_path)
    # Save the markdown to a file in processed/ directory
    with open(
        os.path.join("data/processed/43-101_reports", f"{record_id}.md"), "w"
    ) as f:
        f.write(result.document.export_to_markdown())


# %% [markdown]
# # Refine Parsed PDF Md
#

# %%


def preprocess_markdown(markdown_content: str) -> str:
    """
    Replace false positive headers in markdown (e.g. "## Notes:") with normal content "Notes:"
    """

    # Replace some false positive headers with normal content
    notes_header_regex = re.compile(
        r"^(#{1,6})\s?note[s]?", re.IGNORECASE | re.MULTILINE
    )
    markdown_content = re.sub(notes_header_regex, "Notes:", markdown_content)

    notes_header_regex = re.compile(
        r"^(#{1,6})\s?figure[s]?", re.IGNORECASE | re.MULTILINE
    )
    markdown_content = re.sub(notes_header_regex, "Figure:", markdown_content)

    notes_header_regex = re.compile(
        r"^(#{1,6})\s?table[s]?", re.IGNORECASE | re.MULTILINE
    )
    markdown_content = re.sub(notes_header_regex, "Table:", markdown_content)

    return markdown_content


# %%

# Create directories if they don't exist
RAW_43_101_MD_DIR = "data/processed/43-101_reports"
REFINED_43_101_MD_DIR = "data/processed/43-101_reports_refined"
os.makedirs(RAW_43_101_MD_DIR, exist_ok=True)
os.makedirs(REFINED_43_101_MD_DIR, exist_ok=True)

# Refine the markdown files
for i, file_path in enumerate(Path(RAW_43_101_MD_DIR).glob("*.md")):
    logger.info(
        f"Refining {i + 1}/{len(list(Path(RAW_43_101_MD_DIR).glob('*.md')))}: {file_path}"
    )
    # Read the markdown file
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Preprocess the markdown content
    markdown_content = preprocess_markdown(markdown_content)

    # Save the refined markdown to the output directory
    with open(
        os.path.join(REFINED_43_101_MD_DIR, file_path.name),
        "w",
        encoding="utf-8",
    ) as f:
        logger.info(f"Writing to {os.path.join(REFINED_43_101_MD_DIR, file_path.name)}")
        f.write(markdown_content)

# %%
