# %%
# Evaluate exact match with the ground truth using numpy isclose
import json
import re

import numpy as np

from src.config.logger import logger

# Configuration variables
PRED_OUTPUT_PATH = "data/experiments/FinQA/test_pred_gpt-3.5-turbo-0125.json"
# PRED_OUTPUT_PATH = "data/experiments/FinQA/test_pred_gpt-4-0613.json"

# Load prediction data
with open(PRED_OUTPUT_PATH, "r") as f:
    pred_data = json.load(f)

# Initialize counters
correct = 0
total = 0

for d in pred_data:
    pred = d["pred"]
    gold = d["gold"]

    # Process string gold answer to float
    # Step 1: remove all non-numeric characters except 0-9 and .
    gold = re.sub(r"[^0-9.]", "", gold)

    # Step 2: count how many digits after the decimal point
    # If only one decimal digit and it's 0, then set num_digits to 0
    num_digits = len(gold.split(".")[1]) if "." in gold else 0
    if num_digits == 1 and gold.split(".")[1] == "0":
        num_digits = 0

    # Step 3: round the pred to the same number of digits as the gold
    pred_rounded = round(pred, num_digits)

    # Step 4: cast gold to float and turn to positive if negative
    gold = float(gold) if gold else -1

    # Step 5: check if the pred is close to the gold
    is_close = np.isclose(abs(pred_rounded), abs(gold), rtol=1e-2)
    logger.info(
        f"ID: {d['id']}, Pred: {pred}, Pred Rounded: {pred_rounded}, Gold: {gold}, Is Close: {is_close}"
    )

    if is_close:
        correct += 1
    total += 1

# Print out eval metric
logger.info(
    f"Exact Match (isclose) Accuracy: {correct}/{total} = {correct / total:.4f}"
)
