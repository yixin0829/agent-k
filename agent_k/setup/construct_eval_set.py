"""
Construct the evaluation set for the MinMod agent and save it to JSONL format.

Currently, the evaluation set is constructed by sampling columns to report and filter values.
    We sample 1 to all_relevant_columns_count columns to report + record value column.
        For each columns_to_report, there are 8 question templates with 4 different filter conditions.
            For each template, we sample 1-3 filter values.
                In total, there are all_relevant_columns_count * 8 QA pairs in the evaluation set.

We validate the QA pairs to ensure quality:
    1. Answer must have more than 1 record
    2. Answer must have more than 1 unique data source (i.e. both 43-101 and MRDS data sources)
"""

import os
import random
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import simplejson as json

import agent_k.config.general as config_general
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, MinModHyperCols, QATemplateType
from agent_k.utils.general import sample_values_from_df

# Set random seed for reproducibility
# random.seed(42)

MUST_HAVE_COLUMN = [MinModHyperCols.RECORD_VALUE.value]

# Evaluation set configuration
RELEVANT_COLUMN_CANDIDATES = [
    MinModHyperCols.STATE_OR_PROVINCE.value,
    MinModHyperCols.COUNTRY.value,
    MinModHyperCols.TOTAL_GRADE.value,
    MinModHyperCols.TOTAL_TONNAGE.value,
    MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
    # Ignore these columns for now as if they can be inferred from deposit type
    # MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
    # MinModHyperCols.TOP_1_DEPOSIT_GROUP.value,
]

QA_JSONL_TEMPLATE = {
    "qid": "",
    "question": "",
    "answer": [],
    "metadata": {
        "question_category": "",
        "sql": "",
        "selected_columns": [],
        "filter_conditions": [],
        "data_source": [],
    },
}


# QA template returns a tuple of (question, answer: list[list], data_source: list[str])
def single_state_or_province_qa_template(
    df_hyper: pd.DataFrame, state_or_province: str, selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value] == state_or_province
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites located in {state_or_province}, {country}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def single_country_qa_template(
    df_hyper: pd.DataFrame, country: str, selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[df_hyper[MinModHyperCols.COUNTRY.value] == country][
        MinModHyperCols.DATA_SOURCE.value
    ].values.tolist()
    return (
        f"What are all the mineral sites located in {country}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def single_deposit_type_qa_template(
    df_hyper: pd.DataFrame, deposit_type: str, selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value] == deposit_type
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites with a deposit type of {deposit_type}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def single_deposit_environment_qa_template(
    df_hyper: pd.DataFrame, deposit_environment: str, selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value] == deposit_environment
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites with a deposit environment of {deposit_environment}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def multiple_state_or_province_qa_template(
    df_hyper: pd.DataFrame, state_or_provinces: list[str], selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.STATE_OR_PROVINCE.value].isin(state_or_provinces)
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites located in {state_or_provinces_str}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def multiple_country_qa_template(
    df_hyper: pd.DataFrame, countries: list[str], selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[df_hyper[MinModHyperCols.COUNTRY.value].isin(countries)][
        MinModHyperCols.DATA_SOURCE.value
    ].values.tolist()
    return (
        f"What are all the mineral sites located in {countries_str}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def multiple_deposit_type_qa_template(
    df_hyper: pd.DataFrame, deposit_types: list[str], selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_TYPE.value].isin(deposit_types)
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites with a deposit type of {deposit_types_str}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


def multiple_deposit_environment_qa_template(
    df_hyper: pd.DataFrame, deposit_environments: list[str], selected_columns: list[str]
) -> tuple[str, list[list], list[str]]:
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
    data_source = df_hyper[
        df_hyper[MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value].isin(
            deposit_environments
        )
    ][MinModHyperCols.DATA_SOURCE.value].values.tolist()
    return (
        f"What are all the mineral sites with a deposit environment of {deposit_environments_str}? Report {selected_columns_str}.",
        answer,
        data_source,
    )  # type: ignore[return-value]


@dataclass
class FilterCondition:
    column: str
    operator: str
    value: str | list[str]

    def to_dict(self):
        return {"column": self.column, "operator": self.operator, "value": self.value}


@dataclass
class SampleValuesArgs:
    df: pd.DataFrame
    column: str
    n: int | tuple[int, int]

    def to_dict(self):
        return {"df": self.df, "column": self.column, "n": self.n}


def construct_eval_set_matched_based():
    """
    Construct the evaluation set for the MinMod agent and save it to JSONL format.
    Focus on match-based questions with varying filter values and varying selected columns.
    """
    df_hyper = pd.read_csv(
        os.path.join(
            config_general.GROUND_TRUTH_DIR,
            config_general.enriched_hyper_reponse_file(config_general.COMMODITY),
        )
    )
    # Select 43-101 and MRDS data sources
    df_hyper = df_hyper[
        df_hyper[MinModHyperCols.DATA_SOURCE.value].isin(
            [DataSource.MRDATA_USGS_GOV_MRDS.value, DataSource.API_CDR_LAND.value]
        )
    ]

    qa_templates_to_func_mapping = {
        QATemplateType.SINGLE_STATE_OR_PROVINCE.value: single_state_or_province_qa_template,
        QATemplateType.SINGLE_COUNTRY.value: single_country_qa_template,
        QATemplateType.SINGLE_DEPOSIT_TYPE.value: single_deposit_type_qa_template,
        QATemplateType.SINGLE_DEPOSIT_ENVIRONMENT.value: single_deposit_environment_qa_template,
        QATemplateType.MULTIPLE_STATE_OR_PROVINCE.value: multiple_state_or_province_qa_template,
        QATemplateType.MULTIPLE_COUNTRY.value: multiple_country_qa_template,
        QATemplateType.MULTIPLE_DEPOSIT_TYPE.value: multiple_deposit_type_qa_template,
        QATemplateType.MULTIPLE_DEPOSIT_ENVIRONMENT.value: multiple_deposit_environment_qa_template,
    }

    qa_templates_to_question_cateories_mapping = {
        QATemplateType.SINGLE_STATE_OR_PROVINCE.value: "match-based (single-value)",
        QATemplateType.SINGLE_COUNTRY.value: "match-based (single-value)",
        QATemplateType.SINGLE_DEPOSIT_TYPE.value: "match-based (single-value)",
        QATemplateType.SINGLE_DEPOSIT_ENVIRONMENT.value: "match-based (single-value)",
        QATemplateType.MULTIPLE_STATE_OR_PROVINCE.value: "match-based (multiple-value)",
        QATemplateType.MULTIPLE_COUNTRY.value: "match-based (multiple-value)",
        QATemplateType.MULTIPLE_DEPOSIT_TYPE.value: "match-based (multiple-value)",
        QATemplateType.MULTIPLE_DEPOSIT_ENVIRONMENT.value: "match-based (multiple-value)",
    }

    qa_templates_to_filter_column_mapping = {
        QATemplateType.SINGLE_STATE_OR_PROVINCE.value: MinModHyperCols.STATE_OR_PROVINCE.value,
        QATemplateType.SINGLE_COUNTRY.value: MinModHyperCols.COUNTRY.value,
        QATemplateType.SINGLE_DEPOSIT_TYPE.value: MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
        QATemplateType.SINGLE_DEPOSIT_ENVIRONMENT.value: MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
        QATemplateType.MULTIPLE_STATE_OR_PROVINCE.value: MinModHyperCols.STATE_OR_PROVINCE.value,
        QATemplateType.MULTIPLE_COUNTRY.value: MinModHyperCols.COUNTRY.value,
        QATemplateType.MULTIPLE_DEPOSIT_TYPE.value: MinModHyperCols.TOP_1_DEPOSIT_TYPE.value,
        QATemplateType.MULTIPLE_DEPOSIT_ENVIRONMENT.value: MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
    }

    # Sample filter value arguments for each template type
    qa_to_sample_values_args_mapping = {
        # Sample single filter value arguments
        QATemplateType.SINGLE_STATE_OR_PROVINCE.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.STATE_OR_PROVINCE.value, n=1
        ),
        QATemplateType.SINGLE_COUNTRY.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.COUNTRY.value, n=1
        ),
        QATemplateType.SINGLE_DEPOSIT_TYPE.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.TOP_1_DEPOSIT_TYPE.value, n=1
        ),
        QATemplateType.SINGLE_DEPOSIT_ENVIRONMENT.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value, n=1
        ),
        # Sample multiple filter value arguments (2-3 values)
        QATemplateType.MULTIPLE_STATE_OR_PROVINCE.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.STATE_OR_PROVINCE.value, n=(2, 3)
        ),
        QATemplateType.MULTIPLE_COUNTRY.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.COUNTRY.value, n=(2, 3)
        ),
        QATemplateType.MULTIPLE_DEPOSIT_TYPE.value: SampleValuesArgs(
            df=df_hyper, column=MinModHyperCols.TOP_1_DEPOSIT_TYPE.value, n=(2, 3)
        ),
        QATemplateType.MULTIPLE_DEPOSIT_ENVIRONMENT.value: SampleValuesArgs(
            df=df_hyper,
            column=MinModHyperCols.TOP_1_DEPOSIT_ENVIRONMENT.value,
            n=(2, 3),
        ),
    }

    # Helper function to create a QA pair for a given QA template type
    def create_qa_pair(
        qa_template_key: str,
        qa_template: Callable,
        question_category: str,
        filter_column: str,
        filter_value: str | list[str],
        selected_columns: list[str],
    ) -> dict:
        json_qa_pair = deepcopy(QA_JSONL_TEMPLATE)
        json_qa_pair["qid"] = str(uuid.uuid4())
        (
            json_qa_pair["question"],
            json_qa_pair["answer"],
            json_qa_pair["metadata"]["data_source"],
        ) = qa_template(df_hyper, filter_value, selected_columns)
        json_qa_pair["metadata"]["question_category"] = question_category

        # Handle single vs multiple value cases
        is_multiple = qa_template_key.startswith("multiple_")
        operator = "IN" if is_multiple else "="

        # Format SQL WHERE clause based on single/multiple values
        if is_multiple:
            filter_values_str = ", ".join(f"'{v}'" for v in filter_value)
            where_clause = f"{filter_column} IN ({filter_values_str})"
        else:
            where_clause = f"{filter_column} = '{filter_value}'"

        json_qa_pair["metadata"]["sql"] = (
            f"SELECT {', '.join(selected_columns)} FROM df_hyper WHERE {where_clause}"
        )
        json_qa_pair["metadata"]["selected_columns"] = selected_columns
        json_qa_pair["metadata"]["filter_conditions"] = [
            FilterCondition(
                column=filter_column, operator=operator, value=filter_value
            ).to_dict()
        ]
        return json_qa_pair

    def validate_qa_pair(qa_pair: dict) -> bool:
        # 1. Answer must have more than 1 record
        if len(qa_pair["answer"]) == 1:
            return False
        # 2. Answer must have more than 1 unique data source (i.e. different data sources)
        if len(set(qa_pair["metadata"]["data_source"])) == 1:
            return False
        return True

    qa_pairs: list[dict] = []
    for relevant_columns_count in range(1, len(RELEVANT_COLUMN_CANDIDATES) + 1):
        for template_type in QATemplateType:
            # Sample columns to report in the question. Must have at least 2 columns.
            reported_columns = MUST_HAVE_COLUMN + random.sample(
                RELEVANT_COLUMN_CANDIDATES, relevant_columns_count
            )

            validated = False
            while not validated:
                # Re-sample filter value(s) if the QA pair is not valid
                sample_filter_values = sample_values_from_df(
                    **qa_to_sample_values_args_mapping[template_type.value].to_dict()
                )

                # Create a QA pair (i.e. one JSON object in the JSONL file)
                qa_pair = create_qa_pair(
                    qa_template_key=template_type.value,
                    qa_template=qa_templates_to_func_mapping[template_type.value],
                    question_category=qa_templates_to_question_cateories_mapping[
                        template_type.value
                    ],
                    filter_column=qa_templates_to_filter_column_mapping[
                        template_type.value
                    ],
                    filter_value=sample_filter_values,
                    selected_columns=reported_columns,
                )

                # Validate the QA pair to ensure quality
                validated = validate_qa_pair(qa_pair)
                if not validated:
                    logger.warning(
                        f"QA pair {qa_pair['qid']} is not valid. Re-sampling..."
                    )

            qa_pairs.append(qa_pair)

    # Save questions to eval directory
    os.makedirs(config_general.EVAL_DIR, exist_ok=True)
    with open(
        os.path.join(
            config_general.EVAL_DIR,
            config_general.eval_set_matched_based_file(config_general.COMMODITY),
        ),
        "w",
    ) as f:
        for qa_pair in qa_pairs:
            logger.info(f"Question: {qa_pair['question']}")
            f.write(json.dumps(qa_pair, ignore_nan=True) + "\n")


if __name__ == "__main__":
    construct_eval_set_matched_based()
