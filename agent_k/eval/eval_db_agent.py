import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import autogen
import autogen.runtime_logging
import pandas as pd

import agent_k.config.general as config_general
from agent_k.agents.db_agent import construct_db_agent
from agent_k.config.logger import logger
from agent_k.config.schemas import DataSource
from agent_k.utils.general import load_list_to_df


@dataclass
class EvalReport:
    qid: str = field(default="Unknown", metadata={"description": "Question ID"})
    question: str = field(default="Unknown", metadata={"description": "Question"})
    row_em_score: float = field(
        default=0, metadata={"description": "Exact match score for all rows"}
    )
    row_precision: float = field(
        default=0, metadata={"description": "Precision score for all rows"}
    )
    row_recall: float = field(
        default=0, metadata={"description": "Recall score for all rows"}
    )
    row_f1: float = field(default=0, metadata={"description": "F1 score for all rows"})

    def __str__(self):
        return f"EM: {self.row_em_score:.2f}, Precision: {self.row_precision:.2f}, Recall: {self.row_recall:.2f}, F1: {self.row_f1:.2f}"

    def to_dict(self):
        return {
            "qid": self.qid,
            "question": self.question,
            "row_em_score": self.row_em_score,
            "row_precision": self.row_precision,
            "row_recall": self.row_recall,
            "row_f1": self.row_f1,
        }


def eval_db_agent(full_eval: bool = False):
    """
    Evaluate the DB agent with the eval set.
    If full_eval is True, evaluate all questions.
    """
    # Construct the DB agent and user proxy agent
    db_agent, user_proxy = construct_db_agent()

    # Read the eval dataset
    with open(
        os.path.join(
            config_general.EVAL_DIR,
            config_general.eval_set_matched_based_file(config_general.COMMODITY),
        ),
        "r",
    ) as f:
        eval_set = [json.loads(line) for line in f]
        logger.info(f"Eval set loaded: {len(eval_set)} questions")

    eval_results = []
    for i, qa_pair in enumerate(eval_set):
        if i > 1 and not full_eval:
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

        # Get initial file count
        if not os.path.exists(config_general.AGENT_CACHE_DIR):
            logger.warning(
                f"Agent cache directory not found: {config_general.AGENT_CACHE_DIR}"
            )
            logger.info("Create a new agent cache directory")
            os.makedirs(config_general.AGENT_CACHE_DIR, exist_ok=True)
            initial_files = set()
        else:
            initial_files = set(os.listdir(config_general.AGENT_CACHE_DIR))

        # Start autogen logging
        autogen_logging_session_id = autogen.runtime_logging.start(
            logger_type="file",
            config={"filename": f"runtime_{datetime.now().strftime('%Y-%m-%d')}.log"},
        )
        logger.info(f"{autogen_logging_session_id=}")
        # Run the agent
        user_proxy.initiate_chat(db_agent, message=question, max_turns=10)
        # Stop autogen logging
        autogen.runtime_logging.stop()

        # Get new files created
        current_files = set(os.listdir(config_general.AGENT_CACHE_DIR))
        new_files = current_files - initial_files

        if not new_files:
            logger.warning(f"No result file generated for question {qid}")
            eval_results.append(EvalReport())
            continue

        # Get the latest created file name
        latest_file = max(
            new_files,
            key=lambda f: os.path.getctime(
                os.path.join(config_general.AGENT_CACHE_DIR, f)
            ),
        )
        # Rename the file by appending qid to the filename
        # TODO: Figure out a better way to pass the pid from the agent
        result_path = os.path.join(
            config_general.AGENT_CACHE_DIR, f"{latest_file.split('.')[0]}_{qid}.json"
        )
        os.rename(
            os.path.join(config_general.AGENT_CACHE_DIR, latest_file), result_path
        )

        # Read the agent generated result from the cache
        with open(result_path) as f:
            agent_result = json.load(f)
        agent_df = load_list_to_df(agent_result, selected_cols=selected_cols)
        # Note: Deduplicate the agent result from MRDS data otherwise see > 1 recall
        agent_df.drop_duplicates(inplace=True)

        # Prepare answer DataFrame for comparison (filter by MRDS data source)
        answer_df = load_list_to_df(answer, selected_cols=selected_cols)
        answer_df["data_source"] = data_source
        answer_df = answer_df[
            answer_df["data_source"].eq(DataSource.MRDATA_USGS_GOV.value)
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

        eval_report = EvalReport(
            qid=qid,
            question=question,
            row_em_score=exact_match,
            row_precision=precision,
            row_recall=recall,
            row_f1=f1,
        )
        eval_results.append(eval_report)

        logger.info(eval_report)
        logger.info("=" * 100)

    # Calculate average metrics
    avg_em = sum(r.row_em_score for r in eval_results) / len(eval_results)
    avg_precision = sum(r.row_precision for r in eval_results) / len(eval_results)
    avg_recall = sum(r.row_recall for r in eval_results) / len(eval_results)
    avg_f1 = sum(r.row_f1 for r in eval_results) / len(eval_results)

    logger.info("Overall evaluation metrics:")
    logger.info(f"Average Exact Match: {avg_em:.2f}")
    logger.info(f"Average Precision: {avg_precision:.2f}")
    logger.info(f"Average Recall: {avg_recall:.2f}")
    logger.info(f"Average F1: {avg_f1:.2f}")
    logger.info("=" * 100)

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
    eval_db_agent(full_eval=False)
