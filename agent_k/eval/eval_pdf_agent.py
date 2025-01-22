import asyncio
import json
import os

import pandas as pd
from autogen_agentchat.ui import Console
from openai import OpenAI

import agent_k.config.general as config_general
from agent_k.agents.db_agent import construct_db_agent_team
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource, EvalReport, MinModHyperCols
from agent_k.utils.db_utils import DuckDBWrapper
from agent_k.utils.eval_helper import load_eval_set
from agent_k.utils.general import load_list_to_df

client = OpenAI()


def load_latest_extraction_to_duckdb():
    """
    Load the latest extraction results to duckdb
    """

    # Read the latest extraction results based on the file creation time
    for file in os.listdir(config_general.PDF_AGENT_CACHE_DIR):
        if file.startswith("pdf_agent_extraction_") and file.endswith(".csv"):
            latest_file = os.path.join(config_general.PDF_AGENT_CACHE_DIR, file)
            break
    if not os.path.exists(latest_file):
        raise FileNotFoundError(
            f"No extraction results found in {config_general.PDF_AGENT_CACHE_DIR}"
        )

    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} records from {latest_file}")

    with DuckDBWrapper(database=config_general.DUCKDB_DB_PATH) as db:
        db.create_table_from_df("ni_43_101", df)
        logger.info(f"Loaded {len(df)} rows into 43_101_extraction table")


async def eval_pdf_agent(full_eval: bool = False, eval_set_version: str = "v3"):
    """
    Evaluate the PDF agent with the eval set.
    """

    team = construct_db_agent_team()

    # Note: Below is the same code for eval_db_agent.py. Differences:
    # 1. Ground truth data source is filtered by 43-101 instead of MRDS
    # 2. Replace "record value" with "CDR record id" for matching 43-101 data schema
    # 3. Use minmod.duckdb which only contains 43-101 table
    eval_set = load_eval_set(eval_set_version)

    eval_results = []
    for i, qa_pair in enumerate(eval_set):
        if i > 0 and not full_eval:
            break

        logger.info(f"Evaluating question {i+1} of {len(eval_set)}")
        qid, question, answer, selected_cols, data_source = (
            qa_pair["qid"],
            qa_pair["question"],
            qa_pair["answer"],
            qa_pair["metadata"]["selected_columns"],
            qa_pair["metadata"]["data_source"],
        )

        logger.info(f"{qid=}")
        logger.info(f"{question=}")

        logger.info(
            "Hacky replacing 'record value' with 'CDR record id' for matching 43-101 data schema"
        )
        question = question.replace("record value", "CDR record id")

        # Get initial file count
        if not os.path.exists(config_general.DB_AGENT_CACHE_DIR):
            logger.warning(
                f"Agent cache directory not found: {config_general.DB_AGENT_CACHE_DIR}"
            )
            logger.info("Create a new agent cache directory")
            os.makedirs(config_general.DB_AGENT_CACHE_DIR, exist_ok=True)
            initial_files = set()
        else:
            initial_files = set(os.listdir(config_general.DB_AGENT_CACHE_DIR))

        try:
            await team.reset()  # Reset the team for a new task.
        except RuntimeError:
            pass
        await Console(team.run_stream(task=question))

        # Get new files created
        current_files = set(os.listdir(config_general.DB_AGENT_CACHE_DIR))
        new_files = current_files - initial_files

        if not new_files:
            logger.warning(f"No result file generated for question {qid}")
            eval_results.append(EvalReport())
            continue

        # Get the latest created file name
        latest_file = max(
            new_files,
            key=lambda f: os.path.getctime(
                os.path.join(config_general.DB_AGENT_CACHE_DIR, f)
            ),
        )
        # Rename the file by appending qid to the filename
        # TODO: Figure out a better way to pass the pid from the agent
        result_path = os.path.join(
            config_general.DB_AGENT_CACHE_DIR, f"{latest_file.split('.')[0]}_{qid}.json"
        )
        os.rename(
            os.path.join(config_general.DB_AGENT_CACHE_DIR, latest_file), result_path
        )

        # Read the agent generated result from the cache
        with open(result_path) as f:
            agent_result = json.load(f)
        agent_df = load_list_to_df(agent_result, selected_cols=selected_cols)

        # Prepare answer DataFrame for comparison (filter by 43-101 data source)
        answer_df = load_list_to_df(answer, selected_cols=selected_cols)
        answer_df["data_source"] = data_source
        answer_df = answer_df[
            answer_df["data_source"].eq(DataSource.API_CDR_LAND.value)
        ].drop(columns=["data_source"])

        # Calculate eval metrics
        try:
            common_rows = pd.merge(agent_df, answer_df, how="inner", on=selected_cols)
        except Exception as e:
            logger.error(f"Error merging dataframes: {e}.")
            logger.debug(f"Debugging info:\n{agent_df}\n{answer_df}")
            common_rows = pd.DataFrame()

        precision = len(common_rows) / len(agent_df) if len(agent_df) > 0 else 0
        recall = len(common_rows) / len(answer_df) if len(answer_df) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        exact_match = 1.0 if agent_df.equals(answer_df) else 0.0

        # Calculate site-level metrics (we use record value for unique site identification)
        agent_df_ms = (
            agent_df[MinModHyperCols.RECORD_VALUE.value].to_list()
            if MinModHyperCols.RECORD_VALUE.value in agent_df.columns
            else []
        )
        answer_df_ms = (
            answer_df[MinModHyperCols.RECORD_VALUE.value].to_list()
            if MinModHyperCols.RECORD_VALUE.value in answer_df.columns
            else []
        )
        common_ms = set(agent_df_ms) & set(answer_df_ms)
        ms_precision = len(common_ms) / len(agent_df_ms) if len(agent_df_ms) > 0 else 0
        ms_recall = len(common_ms) / len(answer_df_ms) if len(answer_df_ms) > 0 else 0
        ms_f1 = (
            (2 * (ms_precision * ms_recall) / (ms_precision + ms_recall))
            if (ms_precision + ms_recall) > 0
            else 0
        )
        ms_em_score = 1.0 if agent_df_ms == answer_df_ms else 0.0

        eval_report = EvalReport(
            qid=qid,
            question=question,
            row_em_score=exact_match,
            row_precision=precision,
            row_recall=recall,
            row_f1=f1,
            ms_em_score=ms_em_score,
            ms_precision=ms_precision,
            ms_recall=ms_recall,
            ms_f1=ms_f1,
        )
        eval_results.append(eval_report)

        logger.info(eval_report)
        logger.info("=" * 100)

    # Calculate average metrics
    avg_em = sum(r.row_em_score for r in eval_results) / len(eval_results)
    avg_precision = sum(r.row_precision for r in eval_results) / len(eval_results)
    avg_recall = sum(r.row_recall for r in eval_results) / len(eval_results)
    avg_f1 = sum(r.row_f1 for r in eval_results) / len(eval_results)
    avg_ms_em = sum(r.ms_em_score for r in eval_results) / len(eval_results)
    avg_ms_precision = sum(r.ms_precision for r in eval_results) / len(eval_results)
    avg_ms_recall = sum(r.ms_recall for r in eval_results) / len(eval_results)
    avg_ms_f1 = sum(r.ms_f1 for r in eval_results) / len(eval_results)

    logger.info("Overall evaluation metrics:")
    logger.info(f"Average Exact Match: {avg_em:.2f}")
    logger.info(f"Average Precision: {avg_precision:.2f}")
    logger.info(f"Average Recall: {avg_recall:.2f}")
    logger.info(f"Average F1: {avg_f1:.2f}")
    logger.info(f"Average MS EM: {avg_ms_em:.2f}")
    logger.info(f"Average MS Precision: {avg_ms_precision:.2f}")
    logger.info(f"Average MS Recall: {avg_ms_recall:.2f}")
    logger.info(f"Average MS F1: {avg_ms_f1:.2f}")
    logger.info("=" * 100)

    if full_eval:
        # Save the evaluation results to a CSV file
        eval_df = pd.DataFrame([r.to_dict() for r in eval_results])
        eval_df.to_csv(
            os.path.join(
                config_general.EVAL_DIR,
                config_general.eval_results_file(config_general.COMMODITY),
            ),
            index=False,
        )


if __name__ == "__main__":
    load_latest_extraction_to_duckdb()
    asyncio.run(eval_pdf_agent(full_eval=False, eval_set_version="v3"))
