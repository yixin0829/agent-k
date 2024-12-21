"""
Construct the evaluation set for the MinMod agent and save it to JSONL format.
Each line is a JSON object with the following fields:
- "qid" (str): the ID of the mineral site
- "question" (str): the question to ask the agent
- "answer" (list of lists): the ground truth answer to the question
- "metadata" (dict): the metadata of the mineral site
    - "question_category" (str): the category of the question
    - "sql" (str): the SQL query to answer the question
    - "selected_columns" (list of str): the columns that are selected
    - "filter_conditions" (list of dicts): the conditions to filter the data
"""

import os
import pandas as pd
from agent_k.config.schemas import MinModHyperCols, DataSource
from dataclasses import dataclass
import agent_k.config.general as config_general
import random
import uuid
from loguru import logger
import simplejson as json
from copy import deepcopy


@dataclass
class FilterCondition:
    column: str
    operator: str
    value: str

    def to_dict(self):
        return {"column": self.column, "operator": self.operator, "value": self.value}


MUST_HAVE_COLUMN = [MinModHyperCols.MINERAL_SITE_NAME.value]

# Evaluation set configuration
RELEVANT_COLUMN_CANDIDATES = [
    MinModHyperCols.STATE_OR_PROVINCE.value,
    MinModHyperCols.COUNTRY.value,
    MinModHyperCols.TOTAL_GRADE.value,
    MinModHyperCols.TOTAL_TONNAGE.value,
    MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
    MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
]

FILTER_COLUMN_CANDIDATES = [
    MinModHyperCols.STATE_OR_PROVINCE.value,
    MinModHyperCols.COUNTRY.value,
    MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
    MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
]


# QA template returns a tuple of (question, answer: list[list])
def single_state_or_province_qa_template(
    df_hyper: pd.DataFrame, state_or_province: str, selected_columns: list[str]
) -> tuple[str, list[list]]:
    country = df_hyper[
        df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value] == state_or_province
    ][MinModHyperCols.COUNTRY.value].values[0]
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value] == state_or_province
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites located in {state_or_province}, {country}? Report {selected_columns_str}.",
        answer,
    )


def single_country_qa_template(
    df_hyper: pd.DataFrame, country: str, selected_columns: list[str]
) -> tuple[str, list[list]]:
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[df_hyper[MinModHyperCols.COUNTRY.value] == country][
        selected_columns
    ].values.tolist()
    return (
        f"What are all the mineral sites located in {country}? Report {selected_columns_str}.",
        answer,
    )


def single_deposit_type_qa_template(
    df_hyper: pd.DataFrame, deposit_type: str, selected_columns: list[str]
) -> tuple[str, list[list]]:
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value] == deposit_type
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites with a deposit type of {deposit_type}? Report {selected_columns_str}.",
        answer,
    )


def single_deposit_environment_qa_template(
    df_hyper: pd.DataFrame, deposit_environment: str, selected_columns: list[str]
) -> tuple[str, list[list]]:
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value] == deposit_environment
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites with a deposit environment of {deposit_environment}? Report {selected_columns_str}.",
        answer,
    )


def multiple_state_or_province_qa_template(
    df_hyper: pd.DataFrame, state_or_provinces: list[str], selected_columns: list[str]
) -> tuple[str, list[list]]:
    state_or_provinces_str = ", ".join([sop for sop in state_or_provinces])
    # Match the last comma in state_or_provinces_str and replace it with "or"
    state_or_provinces_str_parts = state_or_provinces_str.rsplit(",", 1)
    if len(state_or_provinces_str_parts) > 1:
        state_or_provinces_str = (
            state_or_provinces_str_parts[0] + " or" + state_or_provinces_str_parts[1]
        )
    else:
        state_or_provinces_str = state_or_provinces_str_parts[0]
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isin(state_or_provinces)
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites located in {state_or_provinces_str}? Report {selected_columns_str}.",
        answer,
    )


def multiple_country_qa_template(
    df_hyper: pd.DataFrame, countries: list[str], selected_columns: list[str]
) -> tuple[str, list[list]]:
    countries_str = ", ".join([c for c in countries])
    # Match the last comma in countries_str and replace it with "or"
    countries_str_parts = countries_str.rsplit(",", 1)
    if len(countries_str_parts) > 1:
        countries_str = countries_str_parts[0] + " or" + countries_str_parts[1]
    else:
        countries_str = countries_str_parts[0]
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[df_hyper[MinModHyperCols.COUNTRY.value].isin(countries)][
        selected_columns
    ].values.tolist()
    return (
        f"What are all the mineral sites located in {countries_str}? Report {selected_columns_str}.",
        answer,
    )


def multiple_deposit_type_qa_template(
    df_hyper: pd.DataFrame, deposit_types: list[str], selected_columns: list[str]
) -> tuple[str, list[list]]:
    deposit_types_str = ", ".join([dt for dt in deposit_types])
    # Match the last comma in deposit_types_str and replace it with "or"
    deposit_types_str_parts = deposit_types_str.rsplit(",", 1)
    if len(deposit_types_str_parts) > 1:
        deposit_types_str = (
            deposit_types_str_parts[0] + " or" + deposit_types_str_parts[1]
        )
    else:
        deposit_types_str = deposit_types_str_parts[0]
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isin(deposit_types)
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites with a deposit type of {deposit_types_str}? Report {selected_columns_str}.",
        answer,
    )


def multiple_deposit_environment_qa_template(
    df_hyper: pd.DataFrame, deposit_environments: list[str], selected_columns: list[str]
) -> tuple[str, list[list]]:
    deposit_environments_str = ", ".join([de for de in deposit_environments])
    # Match the last comma in deposit_environments_str and replace it with "or"
    deposit_environments_str_parts = deposit_environments_str.rsplit(",", 1)
    if len(deposit_environments_str_parts) > 1:
        deposit_environments_str = (
            deposit_environments_str_parts[0]
            + " or"
            + deposit_environments_str_parts[1]
        )
    else:
        deposit_environments_str = deposit_environments_str_parts[0]
    selected_columns_str = ", ".join([rc.replace("_", " ") for rc in selected_columns])
    # Match the last comma in selected_columns_str and replace it with "and"
    selected_columns_str_parts = selected_columns_str.rsplit(",", 1)
    if len(selected_columns_str_parts) > 1:
        selected_columns_str = (
            selected_columns_str_parts[0] + " and" + selected_columns_str_parts[1]
        )
    else:
        selected_columns_str = selected_columns_str_parts[0]
    answer = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isin(
            deposit_environments
        )
    ][selected_columns].values.tolist()
    return (
        f"What are all the mineral sites with a deposit environment of {deposit_environments_str}? Report {selected_columns_str}.",
        answer,
    )


def construct_eval_set():
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.MINMOD_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )
    # Select 43-101 and MRDS data sources
    df_hyper = df_hyper[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isin(
            [DataSource.MRDATA_USGS_GOV_MRDS.value, DataSource.REPORTS_43_101.value]
        )
    ]

    qa_templates = {
        "single_state_or_province": single_state_or_province_qa_template,
        "single_country": single_country_qa_template,
        "single_deposit_type": single_deposit_type_qa_template,
        "single_deposit_environment": single_deposit_environment_qa_template,
        "multiple_state_or_province": multiple_state_or_province_qa_template,
        "multiple_country": multiple_country_qa_template,
        "multiple_deposit_type": multiple_deposit_type_qa_template,
        "multiple_deposit_environment": multiple_deposit_environment_qa_template,
    }

    json_template = {
        "qid": "",
        "question": "",
        "answer": [],
        "metadata": {
            "question_category": "match-based",
            "sql": "",
            "selected_columns": [],
            "filter_conditions": [],
        },
    }

    qa_pairs = []
    # Construct match-based questions on single and multiple values
    for selected_columns_count in range(len(RELEVANT_COLUMN_CANDIDATES)):
        # Select columns to report in the question
        if selected_columns_count == 0:
            selected_columns = MUST_HAVE_COLUMN
        else:
            selected_columns = MUST_HAVE_COLUMN + random.sample(
                RELEVANT_COLUMN_CANDIDATES, random.randint(1, selected_columns_count)
            )

        # Sample single values exclude "Unknown" and nan
        state_or_province = (
            df_hyper[
                ~df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isna()
            ][MinModHyperCols.STATE_OR_PROVINCE.value]
            .sample(n=1)
            .iloc[0]
        )
        country = (
            df_hyper[
                ~df_hyper[MinModHyperCols.COUNTRY.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.COUNTRY.value].isna()
            ][MinModHyperCols.COUNTRY.value]
            .sample(n=1)
            .iloc[0]
        )
        deposit_type = (
            df_hyper[
                ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isna()
            ][MinModHyperCols.TOP_1_DEPOSIT_TYPE.value]
            .sample(n=1)
            .iloc[0]
        )
        deposit_environment = (
            df_hyper[
                ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isin(
                    ["Unknown"]
                )
                & ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isna()
            ][MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value]
            .sample(n=1)
            .iloc[0]
        )

        # Sample multiple unique values (2-3 random values) with no repetition
        state_or_provinces = (
            df_hyper[
                ~df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isna()
            ][MinModHyperCols.STATE_OR_PROVINCE.value]
            .unique()
            .tolist()
        )
        countries = (
            df_hyper[
                ~df_hyper[MinModHyperCols.COUNTRY.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.COUNTRY.value].isna()
            ][MinModHyperCols.COUNTRY.value]
            .unique()
            .tolist()
        )
        deposit_types = (
            df_hyper[
                ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isin(["Unknown"])
                & ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isna()
            ][MinModHyperCols.TOP_1_DEPOSIT_TYPE.value]
            .unique()
            .tolist()
        )
        deposit_environments = (
            df_hyper[
                ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isin(
                    ["Unknown"]
                )
                & ~df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isna()
            ][MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value]
            .unique()
            .tolist()
        )
        state_or_provinces = random.sample(state_or_provinces, random.randint(2, 3))
        countries = random.sample(countries, random.randint(2, 3))
        deposit_types = random.sample(deposit_types, random.randint(2, 3))
        deposit_environments = random.sample(deposit_environments, random.randint(2, 3))

        # Generate questions using templates
        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "single_state_or_province"
        ](df_hyper, state_or_province, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (single-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.STATE_OR_PROVINCE.value} = '{state_or_province}'"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.STATE_OR_PROVINCE.value,
                operator="=",
                value=state_or_province,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "single_country"
        ](df_hyper, country, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (single-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.COUNTRY.value} = '{country}'"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.COUNTRY.value, operator="=", value=country
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "single_deposit_type"
        ](df_hyper, deposit_type, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (single-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.TOP_1_DEPOSIT_TYPE.value} = '{deposit_type}'"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
                operator="=",
                value=deposit_type,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "single_deposit_environment"
        ](df_hyper, deposit_environment, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (single-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value} = '{deposit_environment}'"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
                operator="=",
                value=deposit_environment,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "multiple_state_or_province"
        ](df_hyper, state_or_provinces, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (multiple-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.STATE_OR_PROVINCE.value} IN ({', '.join(f"'{sop}'" for sop in state_or_provinces)})"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.STATE_OR_PROVINCE.value,
                operator="IN",
                value=state_or_provinces,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "multiple_country"
        ](df_hyper, countries, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (multiple-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.COUNTRY.value} IN ({', '.join(f"'{c}'" for c in countries)})"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.COUNTRY.value, operator="IN", value=countries
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "multiple_deposit_type"
        ](df_hyper, deposit_types, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (multiple-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.TOP_1_DEPOSIT_TYPE.value} IN ({', '.join(f"'{dt}'" for dt in deposit_types)})"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
                operator="IN",
                value=deposit_types,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

        json_qa_pair = deepcopy(json_template)
        json_qa_pair["qid"] = str(uuid.uuid4())
        json_qa_pair["question"], json_qa_pair["answer"] = qa_templates[
            "multiple_deposit_environment"
        ](df_hyper, deposit_environments, selected_columns)
        json_qa_pair["metadata"]["question_category"] = "match-based (multiple-value)"
        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value} IN ({', '.join(f"'{de}'" for de in deposit_environments)})"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
                operator="IN",
                value=deposit_environments,
            ).to_dict()
        ]
        qa_pairs.append(json_qa_pair)

    # Save questions to eval directory
    os.makedirs(config_general.EVAL_DIR, exist_ok=True)
    with open(
        os.path.join(config_general.EVAL_DIR, "eval_set_matched_based.jsonl"), "w"
    ) as f:
        for qa_pair in qa_pairs:
            logger.info(f"Question: {qa_pair['question']}")
            f.write(json.dumps(qa_pair, ignore_nan=True) + "\n")


if __name__ == "__main__":
    construct_eval_set()
