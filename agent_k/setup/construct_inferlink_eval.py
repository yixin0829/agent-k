"""
The goal is to construct a ground truth table using Inferlink's extraction results.
"""

import glob
import os

import pandas as pd

from agent_k.config.schemas import InferlinkEvalColumns


def read_metadata_files(base_dir="data/raw/ground_truth/inferlink"):
    """
    Read all metadata.csv files in the specified directory. Enrich the dataframe with the cdr_record_id.

    Args:
        base_dir (str): Base directory containing the metadata.csv files

    Returns:
        pd.DataFrame: Master metadata dataframe
    """
    # Find all metadata.csv files recursively
    metadata_files = glob.glob(f"{base_dir}/**/metadata.csv", recursive=True)

    print(f"Found {len(metadata_files)} metadata.csv files")

    master_metadata_df = pd.DataFrame()

    # Process each file
    for file_path in metadata_files:
        try:
            # Get the parent directory name (which will be the cdr_record_id)
            parent_dir = os.path.basename(os.path.dirname(file_path))

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Insert the new column at the beginning
            df.insert(0, InferlinkEvalColumns.CDR_RECORD_ID.value, parent_dir)

            # Append to master metadata dataframe
            master_metadata_df = pd.concat([master_metadata_df, df], ignore_index=True)

            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("Processing complete!")
    print(master_metadata_df.head())
    return master_metadata_df


def read_mineral_inventory_files(base_dir="data/raw/ground_truth/inferlink"):
    """
    Read all mineral_inventory_minimal.csv files in the specified directory.
    Enrich the dataframe with the cdr_record_id.

    Args:
        base_dir (str): Base directory containing the mineral_inventory_minimal.csv files

    Returns:
        pd.DataFrame: Master mineral inventory dataframe
    """
    # Find all mineral_inventory_minimal.csv files recursively
    inventory_files = glob.glob(
        f"{base_dir}/**/mineral_inventory_minimal.csv", recursive=True
    )

    print(f"Found {len(inventory_files)} mineral_inventory_minimal.csv files")

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
            df = pd.read_csv(file_path)

            # Insert the new column at the beginning
            df.insert(0, InferlinkEvalColumns.CDR_RECORD_ID.value, parent_dir)
            df.insert(1, InferlinkEvalColumns.MAIN_COMMODITY.value, parent_parent_dir)

            # If the df is empty, append a row with cdr_record_id
            if df.empty:
                df = pd.DataFrame(
                    {
                        InferlinkEvalColumns.CDR_RECORD_ID.value: [parent_dir],
                        InferlinkEvalColumns.MAIN_COMMODITY.value: [parent_parent_dir],
                    }
                )

            # Append to master inventory dataframe
            master_inventory_df = pd.concat(
                [master_inventory_df, df], ignore_index=True
            )

            print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    master_inventory_df[InferlinkEvalColumns.COMMODITY.value] = (
        master_inventory_df[InferlinkEvalColumns.COMMODITY.value]
        .str.lower()
        .str.strip()
    )

    print("Processing complete!")
    print(master_inventory_df.head())
    return master_inventory_df


def normalize_ore_units(df):
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
            print(f"Error normalizing row {idx}: {e}")
            # Keep original values if there's an error

    return df


def normalize_grade_units(df):
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
        "oz/t": 0.00343,  # Troy ounces per short ton (1 oz/t ≈ 0.00343%)
        "oz/ton": 0.00343,  # Troy ounces per ton
        "opt": 0.00343,  # Ounces per ton abbreviation
        "kg/t": 0.1,  # Kilograms per tonne (1 kg/t = 0.1%)
        "ppb": 0.0000001,  # Parts per billion (1 ppb = 0.0000001%)
        "wt%": 1.0,  # Weight percent
        "wt.%": 1.0,  # Weight percent (alternative notation)
        "grams per tonne": 0.0001,  # Grams per tonne (spelled out)
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
            print(f"Error normalizing grade in row {idx}: {e}")
            # Keep original values if there's an error

    return df


if __name__ == "__main__":
    # When run as a script, process all metadata.csv files
    master_metadata_df = read_metadata_files()
    master_metadata_df.to_csv("data/processed/inferlink_metadata.csv", index=False)
    master_inventory_df = read_mineral_inventory_files()
    master_inventory_df.to_csv("data/processed/inferlink_inventory.csv", index=False)

    master_inventory_df = normalize_ore_units(master_inventory_df)
    master_inventory_df = normalize_grade_units(master_inventory_df)

    # Assert there are no values in the "zone" column contain "total" to avoid double counting
    # assert not master_inventory_df["zone"].str.lower().str.contains("total").any()

    cols_to_drop = [
        "ore_unit_observed_name",
        "grade_unit_observed_name",
        "ore_value",
        "grade_value",
        "cutoff_grade_unit_observed_name",
        "cutoff_grade_value",
        "contained_metal",
        "zone",
    ]
    master_inventory_df = master_inventory_df.drop(columns=cols_to_drop)

    category_to_resource_or_reserve = {
        "inferred": "resource",
        "measured": "resource",
        "indicated": "resource",
        "mineral resource": "resource",
        "measured+indicated": "resource",  # Exception
        "proved": "reserve",
        "probable": "reserve",
        "proven": "reserve",
    }
    # remove any whitespace from category (e.g. "proven + probable" -> "proven+probable")
    master_inventory_df["category_observed_name"] = master_inventory_df[
        "category_observed_name"
    ].str.replace(" ", "")

    master_inventory_df.insert(
        4,
        "resource_or_reserve",
        master_inventory_df["category_observed_name"].map(
            category_to_resource_or_reserve
        ),
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
                InferlinkEvalColumns.COMMODITY.value,
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
            InferlinkEvalColumns.COMMODITY.value,
        ],
        columns="resource_or_reserve",
        values=["normalized_ore_value", "contained_metal"],
    )
    # expand the index to separate columns
    resource_n_reserve_total_tonnage.columns = [
        f"{col[0]}_{col[1]}" for col in resource_n_reserve_total_tonnage.columns
    ]
    resource_n_reserve_total_tonnage = resource_n_reserve_total_tonnage.reset_index()

    # Left join master_metadata_df and master_inventory_df on cdr_record_id
    master_df = pd.merge(
        resource_n_reserve_total_tonnage,
        master_metadata_df,
        on=InferlinkEvalColumns.CDR_RECORD_ID.value,
        how="left",
    )

    cols_to_drop = [
        "authors",
        "year",
        "month",
    ]
    master_df = master_df.drop(columns=cols_to_drop)

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

    master_df.to_csv("data/processed/inferlink_ground_truth.csv", index=False)

    # Filter for rows where commodity_observed_name appears in the main_commodity column
    for idx, row in master_df.iterrows():
        if (
            row[InferlinkEvalColumns.MAIN_COMMODITY.value]
            in row[InferlinkEvalColumns.COMMODITY.value]
        ):
            master_df.at[idx, "is_main_commodity"] = True
        else:
            master_df.at[idx, "is_main_commodity"] = False

    master_df = master_df[master_df["is_main_commodity"]]
    master_df.drop(columns=["is_main_commodity"], inplace=True)
    master_df.to_csv("data/processed/inferlink_ground_truth_filtered.csv", index=False)
