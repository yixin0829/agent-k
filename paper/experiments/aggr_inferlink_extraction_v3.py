# %% [markdown]
# Prerequisite: Download "data/raw/inferlink_extraction_v3" from ta2 github repo: https://github.com/DARPA-CRITICALMAAS/ta2-minmod-data/tree/main/data/mineral-sites/inferlink/mining_report
#
# Process Inferlink's v3 extraction into a similar file as the ground truth file.

# %%
import json
import os

import pandas as pd

from agent_k.config.schemas import InferlinkEvalColumns
from agent_k.setup.construct_inferlink_eval import (
    normalize_grade_units,
    normalize_ore_units,
)


def get_nested(data, path, default=None):
    """Helper function to get a nested value from a dictionary."""
    try:
        for key in path:
            if isinstance(data, dict):
                data = data.get(key, default)
            elif isinstance(data, list) and isinstance(key, int):
                data = data[key]
            else:
                return default
        return data
    except (KeyError, IndexError, TypeError):
        return default


# Get all JSON files in the directory
json_dir = "data/raw/inferlink_extraction_v3/"
json_files = [
    os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")
]

df_gt = pd.read_csv("paper/data/processed/ground_truth/inferlink_ground_truth.csv")
# Get unique pairs of record_id and main_commodity
id_main_commodity = (
    df_gt[
        [
            InferlinkEvalColumns.CDR_RECORD_ID.value,
            InferlinkEvalColumns.MAIN_COMMODITY.value,
        ]
    ]
    .drop_duplicates()
    .values.tolist()
)
gt_record_ids = [id for id, _ in id_main_commodity]
record_id_main_commodity = {
    id: main_commodity for id, main_commodity in id_main_commodity
}

matched_data = {}
for file_path in json_files:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            for d in data:
                if d.get("record_id", "") in gt_record_ids:
                    matched_data[d["record_id"]] = d
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

rows = []
for record_id, d in matched_data.items():
    metadata = {}
    metadata["cdr_record_id"] = record_id
    metadata["mineral_site_name"] = get_nested(d, ["name"])
    metadata["country"] = get_nested(
        d, ["location_info", "country", 0, "observed_name"]
    )
    metadata["state_or_province"] = get_nested(
        d, ["location_info", "state_or_province", 0, "observed_name"]
    )
    metadata["main_commodity"] = record_id_main_commodity[record_id]

    if "mineral_inventory" in d:
        for mi in d["mineral_inventory"]:
            row = metadata.copy()
            row["commodity_observed_name"] = get_nested(
                mi, ["commodity", "observed_name"]
            )
            row["category_observed_name"] = get_nested(
                mi, ["category", 0, "observed_name"]
            )
            row["ore_unit_observed_name"] = get_nested(
                mi, ["ore", "unit", "observed_name"]
            )
            row["ore_value"] = get_nested(mi, ["ore", "value"])
            row["grade_unit_observed_name"] = get_nested(
                mi, ["grade", "unit", "observed_name"]
            )
            row["grade_value"] = get_nested(mi, ["grade", "value"])
            row["zone"] = get_nested(mi, ["zone"])
            rows.append(row)
    else:
        row = metadata.copy()
        rows.append(row)


# convert to csv
df_inferlink_extraction_v3 = pd.DataFrame(rows)
print(df_inferlink_extraction_v3.shape)
df_inferlink_extraction_v3.head()

# %%
# Assign default units if missing
df_inferlink_extraction_v3["ore_unit_observed_name"] = df_inferlink_extraction_v3[
    "ore_unit_observed_name"
].fillna("tonnes")
df_inferlink_extraction_v3["grade_unit_observed_name"] = df_inferlink_extraction_v3[
    "grade_unit_observed_name"
].fillna("percent")

# Assign default values if missing
df_inferlink_extraction_v3["ore_value"] = df_inferlink_extraction_v3[
    "ore_value"
].fillna(0)
df_inferlink_extraction_v3["grade_value"] = df_inferlink_extraction_v3[
    "grade_value"
].fillna(0)

master_metadata_df = df_inferlink_extraction_v3.loc[
    :, ["cdr_record_id", "mineral_site_name", "country", "state_or_province"]
].drop_duplicates()
master_inventory_df = df_inferlink_extraction_v3.loc[
    :,
    [
        "cdr_record_id",
        "main_commodity",
        "commodity_observed_name",
        "category_observed_name",
        "ore_unit_observed_name",
        "ore_value",
        "grade_unit_observed_name",
        "grade_value",
        "zone",
    ],
]
master_inventory_df = normalize_ore_units(master_inventory_df)
master_inventory_df = normalize_grade_units(master_inventory_df)

# %%
cols_to_drop = [
    "ore_unit_observed_name",
    "grade_unit_observed_name",
    "ore_value",
    "grade_value",
    "zone",
]
master_inventory_df = master_inventory_df.drop(columns=cols_to_drop)

category_to_resource_or_reserve = {
    "inferred": "resource",
    "measured": "resource",
    "indicated": "resource",
    "mineral resource": "resource",
    "measured+indicated": "resource",  # Exception
    "proven+probable": "reserve",  # Exception
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
# Assert for all rows have "category_observed_name" it has a corresponding "resource_or_reserve" value
assert (
    not master_inventory_df[
        master_inventory_df["category_observed_name"].notna()
        & master_inventory_df["resource_or_reserve"].isna()
    ]
    .any()
    .any()
)
master_inventory_df.drop(columns=["category_observed_name"], inplace=True)

master_inventory_df["contained_metal"] = (
    master_inventory_df["normalized_ore_value"]
    * master_inventory_df["normalized_grade_value"]
    / 100
)

# Note: mineral site with no inventory data will have a NaN value in the "category_observed_name" column
# and thus a NaN value in the "resource_or_reserve" column. These rows will be dropped during the groupby operation.

# Group by cdr_record_id, resource_or_reserve and sum the normalized_ore_value
resource_n_reserve_total_tonnage = (
    master_inventory_df.groupby(
        [
            InferlinkEvalColumns.CDR_RECORD_ID.value,
            InferlinkEvalColumns.MAIN_COMMODITY.value,
            InferlinkEvalColumns.COMMODITY_OBSERVED_NAME.value,
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

# pivot the resource_n_reserve column to wide format
resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage.pivot(
    index=[
        InferlinkEvalColumns.CDR_RECORD_ID.value,
        InferlinkEvalColumns.MAIN_COMMODITY.value,
        InferlinkEvalColumns.COMMODITY_OBSERVED_NAME.value,
    ],
    columns="resource_or_reserve",
    values=["normalized_ore_value", "contained_metal"],
)
# expand the index to separate columns
resource_n_reserve_total_tonnage.columns = [
    f"{col[0]}_{col[1]}" for col in resource_n_reserve_total_tonnage.columns
]
resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage.reset_index()

# Left join master_metadata_df (mineral site name, country, state or province) to
# master_inventory_df (resource or reserve, total tonnage, contained metal) on cdr_record_id
master_df = pd.merge(
    resource_n_reserve_total_tonnage,
    master_metadata_df,
    on=InferlinkEvalColumns.CDR_RECORD_ID.value,
    how="left",
)

cols_to_rename = {
    "normalized_ore_value_resource": InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_TONNAGE.value,
    "normalized_ore_value_reserve": InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_TONNAGE.value,
    "contained_metal_resource": InferlinkEvalColumns.TOTAL_MINERAL_RESOURCE_CONTAINED_METAL.value,
    "contained_metal_reserve": InferlinkEvalColumns.TOTAL_MINERAL_RESERVE_CONTAINED_METAL.value,
    "country_observed_name": InferlinkEvalColumns.COUNTRY.value,
    "state_or_province_observed_name": InferlinkEvalColumns.STATE_OR_PROVINCE.value,
    "mining_name": InferlinkEvalColumns.MINERAL_SITE_NAME.value,
}
master_df = master_df.rename(columns=cols_to_rename)
master_df.head()

# %%
master_df = master_df.merge(
    df_gt[["id", "cdr_record_id", "main_commodity", "commodity_observed_name"]],
    on=[
        InferlinkEvalColumns.CDR_RECORD_ID.value,
        InferlinkEvalColumns.MAIN_COMMODITY.value,
        InferlinkEvalColumns.COMMODITY_OBSERVED_NAME.value,
    ],
    how="inner",
)
# Move id to the first column
master_df = master_df[["id", *master_df.columns[:-1]]]
print(master_df.shape)
master_df.head()

# %%
master_df.to_csv(
    "paper/data/experiments/250628_inferlink_extraction_v3/inferlink_extraction_v3.csv",
    index=False,
)
