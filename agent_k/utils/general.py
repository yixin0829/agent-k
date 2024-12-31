import random

import pandas as pd

from agent_k.config.schemas import MinModHyperCols


def sample_values(df: pd.DataFrame, column: str, n: int | tuple = 1) -> str | list[str]:
    """Sample non-Unknown, non-null values from a column in a dataframe.

    Args:
        df: DataFrame to sample from
        column: Column name to sample from
        n: If int, sample exactly n values. If tuple of (min_n, max_n), sample random number of values in that range.

    Returns:
        If n=1, returns single sampled value as string.
        Otherwise returns list of sampled values.
    """
    unique_values = (
        df[~df[column].isin(["Unknown"]) & ~df[column].isna()][column].unique().tolist()
    )

    if isinstance(n, tuple):
        min_n, max_n = n
        n = random.randint(min_n, max_n)

    sampled = random.sample(unique_values, n)
    return sampled[0] if n == 1 else sampled


def load_list_to_df(data: list[list[str]], selected_cols: list[str]) -> pd.DataFrame:
    """Load a list of lists into a DataFrame with selected columns.

    Args:
        data: List of lists to load into DataFrame
        selected_columns: List of columns to select from the data

    Returns:
        DataFrame with selected columns and type conversion.
    """
    df = pd.DataFrame(data, columns=selected_cols, dtype="object")

    # Convert columns to float
    for col in selected_cols:
        if col in [MinModHyperCols.TOTAL_GRADE, MinModHyperCols.TOTAL_TONNAGE]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
