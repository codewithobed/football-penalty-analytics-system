import pandas as pd
import numpy as np

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the football penalty dataset."""
    df = df.copy()
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False, regex=True)]
    df = df.loc[:, ~df.columns.astype(str).str.endswith(".1")]
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    df = df.loc[:, df.isna().mean() < 0.95]

    for col in df.columns:
        if col.lower() in ["minute", "home_goals", "away_goals", "player_id", "goalkeeper_id", "goalkeer_id"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    return df

def find_target_column(df: pd.DataFrame):
    """Find the target column."""
    for col in ["Outcome", "outcome", "Result", "result", "Target", "target"]:
        if col in df.columns:
            return col
    return None

def normalise_outcome(value):
    """Convert outcome to 1 = Goal and 0 = Miss."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()

    if text in ["goal", "scored", "score", "success", "successful", "1", "1.0"]:
        return 1
    if text in ["miss", "missed", "saved", "save", "failed", "unsuccessful", "0", "0.0"]:
        return 0

    try:
        numeric = float(text)
        return 1 if numeric >= 0.5 else 0
    except Exception:
        return np.nan

def add_outcome_label(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add binary and readable outcome labels."""
    df = df.copy()
    df["Outcome_Binary"] = df[target_col].apply(normalise_outcome)
    df["Outcome_Label"] = df["Outcome_Binary"].map({1: "Goal", 0: "Miss"})
    return df
