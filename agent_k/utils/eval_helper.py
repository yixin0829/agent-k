import json
import os

import agent_k.config.general as config_general
from agent_k.config.logger import logger


def load_eval_set(eval_set_version: str = "v3"):
    # Read the eval dataset
    with open(
        os.path.join(
            config_general.EVAL_DIR,
            config_general.eval_set_matched_based_file(
                config_general.COMMODITY, eval_set_version
            ),
        ),
        "r",
    ) as f:
        eval_set = [json.loads(line) for line in f]
        logger.info(f"Eval set loaded: {len(eval_set)} questions")

    return eval_set
