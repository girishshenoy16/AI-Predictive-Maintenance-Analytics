import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/FD001")
PROCESSED_DATA_PATH = Path("data/processed")


COLUMNS = (
    ["engine_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

def load_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Load raw NASA turbofan data file and assign proper column names.
    """
    df = pd.read_csv(
        file_path,
        sep=" ",
        header=None
    )

    # Remove empty columns caused by space delimiter
    df = df.dropna(axis=1)

    df.columns = COLUMNS
    return df


def load_fd001_data():
    train_df = load_raw_data(RAW_DATA_PATH / "train_FD001.txt")
    test_df = load_raw_data(RAW_DATA_PATH / "test_FD001.txt")

    train_df["source"] = "train"
    test_df["source"] = "test"

    return train_df, test_df


def combine_datasets(train_df, test_df):
    return pd.concat([train_df, test_df], ignore_index=True)


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = (
        df.groupby("engine_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )

    df = df.merge(max_cycle, on="engine_id", how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def create_failure_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """
    Binary label indicating whether failure will occur within given cycles.
    """
    df["failure_within_30"] = (df["RUL"] <= threshold).astype(int)
    return df


def drop_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    constant_cols = [
        col for col in sensor_cols
        if df[col].std() == 0
    ]

    return df.drop(columns=constant_cols)


# --------------------------------------------------
# NEW CLEANING FUNCTION (Integrated Cleanly)
# --------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    print("Removing duplicates...")
    df = df.drop_duplicates()

    print("Handling missing values...")
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method="ffill")

    print("Removing negative RUL values...")
    df = df[df["RUL"] >= 0]

    print("Clipping extreme outliers (1%-99%)...")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    for col in sensor_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    return df


def main():
    print("Loading FD001 data...")
    train_df, test_df = load_fd001_data()

    print("Combining datasets...")
    full_df = combine_datasets(train_df, test_df)

    print("Initial Shape: - ", full_df.shape)

    print("Computing RUL...")
    full_df = compute_rul(full_df)

    print("Creating failure labels...")
    full_df = create_failure_label(full_df, threshold=30)

    print("Dropping constant sensors...")
    full_df = drop_constant_sensors(full_df)

    print("Cleaning dataset...")
    full_df = clean_dataset(full_df)

    output_path = PROCESSED_DATA_PATH / "fd001_master.csv"
    full_df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to {output_path}")
    print("Dataset shape:", full_df.shape)
    print("Failure rate:")
    print(full_df["failure_within_30"].value_counts(normalize=True))


if __name__ == "__main__":
    main()