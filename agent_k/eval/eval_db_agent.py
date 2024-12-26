import json
import os
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

import agent_k.config.general as config_general
from agent_k.agents.db_agent import construct_db_agent
from agent_k.config.schemas import DataSource


@dataclass
class EvalReport:
    qid: str = field(default="", metadata={"description": "Question ID"})
    question: str = field(default="", metadata={"description": "Question"})
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
        indent = "\t" * 8
        return (
            f"QID: {self.qid}\n"
            f"{indent}Question: {self.question}\n"
            f"{indent}EM: {self.row_em_score:.2f}, Precision: {self.row_precision:.2f}, Recall: {self.row_recall:.2f}, F1: {self.row_f1:.2f}"
        )

    def to_dict(self):
        return {
            "qid": self.qid,
            "question": self.question,
            "row_em_score": self.row_em_score,
            "row_precision": self.row_precision,
            "row_recall": self.row_recall,
            "row_f1": self.row_f1,
        }


def eval_db_agent(dev_mode: bool = False):
    """
    Evaluate the DB agent with the eval set.
    If dev_mode is True, only evaluate the first 2 questions.
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
        if i > 1 and dev_mode:
            break
        logger.info(f"Evaluating question {i+1} of {len(eval_set)}")
        qid, question, answer, selected_cols, data_source = (
            qa_pair["qid"],
            qa_pair["question"],
            qa_pair["answer"],
            qa_pair["metadata"]["selected_columns"],
            qa_pair["metadata"]["data_source"],
        )
        logger.info(f"Sampling question: {qid}, {question}")

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

        # Run the agent
        user_proxy.initiate_chat(db_agent, message=question, max_turns=10)

        # Get new files created
        current_files = set(os.listdir(config_general.AGENT_CACHE_DIR))
        new_files = current_files - initial_files

        if not new_files:
            logger.warning(f"No result file generated for question {qid}")
            eval_results.append(EvalReport())
            continue

        # Get the latest created file
        latest_file = max(
            new_files,
            key=lambda f: os.path.getctime(
                os.path.join(config_general.AGENT_CACHE_DIR, f)
            ),
        )
        result_path = os.path.join(config_general.AGENT_CACHE_DIR, latest_file)

        # Read the agent generated result
        with open(result_path) as f:
            agent_result = json.load(f)
        agent_df = pd.DataFrame(agent_result, columns=selected_cols)

        # Convert answer to DataFrame for comparison (filter by data source)
        answer_df = pd.DataFrame(answer, columns=selected_cols)
        answer_df["data_source"] = data_source
        answer_df = answer_df[
            answer_df["data_source"] == DataSource.MRDATA_USGS_GOV.value
        ]
        answer_df.drop(columns=["data_source"], inplace=True)

        # Calculate metrics
        try:
            common_rows = pd.merge(agent_df, answer_df, how="inner", on=selected_cols)
        except Exception as e:
            logger.error(f"Error merging dataframes: {e}")
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
    # eval_db_agent(dev_mode=True)
    eval_db_agent()
