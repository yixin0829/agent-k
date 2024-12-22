import random

import pandas as pd


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
