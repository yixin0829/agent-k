"""
https://github.com/DARPA-CRITICALMAAS/ta2-minmod-dashboard/blob/main/models/ms.py
"""

import asyncio

import pandas as pd

from agent_k.utils import dataservice_utils


class MineralSite:
    """A class for holding the mineral site data"""

    def __init__(self, commodity):
        self.commodity = commodity.lower()
        self.deposit_types = []
        self.country = []
        self.data_cache = {
            "countries": {},
            "deposit-types": {},
            "states-or-provinces": {},
            "commodities": {},
        }

    def init(self):
        """Initialize and load data from query path using the function reference"""

        self.load_data_cache()

        # https://github.com/DARPA-CRITICALMAAS/ta2-minmod-kg/blob/0bdb5482e1bce393672721f64d8426a86c642366/minmodkg/api/routers/dedup_mineral_site.py#L30
        self.df = pd.DataFrame(
            self.clean_and_fix(
                dataservice_utils.fetch_api_data(
                    "/dedup-mineral-sites",
                    params={"commodity": self.commodity},
                    ssl_flag=False,
                )
            )
        )
        if self.df.empty:
            raise Exception("No Data Available")

        self.df = self.clean_df(self.df)
        self.deposit_types = self.df["top_1_deposit_type"].drop_duplicates().to_list()
        self.country = self.df["country"].drop_duplicates().to_list()

    def load_data_cache(self):
        data_list = sorted(self.data_cache.keys())

        data_results = asyncio.run(
            dataservice_utils.fetch_all([("/" + url, None) for url in data_list])
        )

        for i in range(len(data_list)):
            for data in data_results[i]:
                q_key = data["uri"].split("/")[-1]
                self.data_cache[data_list[i]][q_key] = data

    def update_commodity(self, selected_commodity):
        """sets new commodity"""
        self.commodity = selected_commodity.lower()

    def clean_and_fix(self, raw_data):
        results = []
        for data in raw_data:
            if len(data["deposit_types"]) == 0:
                continue

            combined_data = {}
            combined_data["ms"] = "/".join(
                ["https://minmod.isi.edu/derived", "resource", data["id"]]
            )
            combined_data["ms_name"] = data["name"]
            combined_data["ms_type"] = data["type"]
            combined_data["ms_rank"] = data["rank"]
            combined_data["ms_sites"] = [site["id"] for site in data["sites"]]

            # Location details
            if (
                "location" in data
                and "country" in data["location"]
                and data["location"]["country"]
                and data["location"]["country"][0] in self.data_cache["countries"]
            ):
                combined_data["country"] = self.data_cache["countries"][
                    data["location"]["country"][0]
                ]["name"]
            else:
                combined_data["country"] = None

            if (
                "location" in data
                and "state_or_province" in data["location"]
                and data["location"]["state_or_province"]
                and data["location"]["state_or_province"][0]
                in self.data_cache["states-or-provinces"]
            ):
                combined_data["state_or_province"] = self.data_cache[
                    "states-or-provinces"
                ][data["location"]["state_or_province"][0]]["name"]
            else:
                combined_data["state_or_province"] = None

            if "location" in data:
                combined_data["lat"] = data["location"].get("lat", None)
                combined_data["lon"] = data["location"].get("lon", None)

            # Deposit Type details
            highest_confidence_deposit = max(
                data["deposit_types"], key=lambda x: x["confidence"]
            )

            deposit_details = self.data_cache["deposit-types"].get(
                highest_confidence_deposit["id"], None
            )

            if not deposit_details:
                continue
            combined_data["top1_deposit_name"] = deposit_details["name"]
            combined_data["top1_deposit_group"] = deposit_details["group"]
            combined_data["top1_deposit_environment"] = deposit_details["environment"]
            combined_data["top1_deposit_confidence"] = highest_confidence_deposit[
                "confidence"
            ]
            combined_data["top1_deposit_source"] = highest_confidence_deposit["source"]

            # Commodity details
            combined_data["commodity"] = data["grade_tonnage"]["commodity"]

            # GT details
            if "total_grade" in data["grade_tonnage"]:
                combined_data["total_grade"] = data["grade_tonnage"]["total_grade"]
                combined_data["total_tonnage"] = data["grade_tonnage"]["total_tonnage"]
                combined_data["total_contained_metal"] = data["grade_tonnage"][
                    "total_contained_metal"
                ]

            # Setting Unkown Deposit Types
            if not combined_data.get("total_tonnage") or not combined_data.get(
                "total_grade"
            ):
                combined_data["top1_deposit_name"] = "Unknown"

            results.append(combined_data)
        return results

    def clean_df(self, df):
        """A cleaner method to clean the raw data obtained from the SPARQL endpoint"""
        drop_columns = [
            "commodity",
            "total_contained_metal",
        ]
        df_selected = df.drop(drop_columns, axis=1)

        # rename columns
        col_names = {
            "ms": "mineral_site_uri",
            "ms_name": "mineral_site_name",
            "ms_type": "mineral_site_type",
            "ms_rank": "mineral_site_rank",
            "ms_sites": "sites",
            "country": "country",
            "state_or_province": "state_or_province",
            "lat": "latitude",
            "lon": "longitude",
            "total_tonnage": "total_tonnage",
            "total_grade": "total_grade",
            "top1_deposit_name": "top_1_deposit_type",
            "top1_deposit_group": "top1_deposit_group",
            "top1_deposit_environment": "top_1_deposit_environment",
            "top1_deposit_confidence": "top_1_deposit_classification_confidence",
            "top1_deposit_source": "top_1_deposit_classification_source",
        }

        df_selected = df_selected.rename(columns=col_names)

        # clean column ms name
        def clean_names(ms_name):
            if isinstance(ms_name, list):
                return ms_name[0]
            return ms_name

        df_selected["mineral_site_name"] = df_selected["mineral_site_name"].apply(
            clean_names
        )

        return df_selected
