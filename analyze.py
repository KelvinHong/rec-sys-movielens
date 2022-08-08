import pandas as pd
import os
from termcolor import colored

filenames = [
    # "genome-scores.csv",
    # "genome-tags.csv",
    "links.csv",
    "movies.csv",
    "ratings.csv",
    "tags.csv",
]
files = [
    os.path.join("./ml-latest-small/", csv_file) for csv_file in filenames
]

for file in files:
    print("=" * 50)
    df = pd.read_csv(file)
    print(f"Metadata of the file {file} is:")
    print(f"    Number of rows: {len(df)}.")
    print(f"    Column names: {df.columns.tolist()}")
    # Detect NAs
    for colname in df.columns:
        NA_count = df[colname].isna().sum()
        if NA_count > 0:
            print(
                colored(
                    f"        Column '{colname}' has {df[colname].isna().sum()} missing values;",
                    "red"))
    # Detect categorical columns
    categorical_cols = [
        colname for colname in df.columns if df[colname].dtype == "object"
    ]
    if categorical_cols:
        print(
            colored(f"    Categorical columns are {categorical_cols}.",
                    "green"))
    # calculate statistical summary for numeric and non-ID columns
    numeric_non_id_cols = [colname for colname in df.columns \
            if ("id" not in colname.lower() \
                and df[colname].dtype != "object"\
                and colname != "timestamp")]
    if numeric_non_id_cols:
        print(
            "    Statistics of some columns are given:\n",
            df[numeric_non_id_cols].describe().apply(
                lambda s: s.apply('{0:.5f}'.format)))
