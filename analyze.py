import pandas as pd
import os
from termcolor import colored


filenames = [
    "genome-scores.csv",
    "genome-tags.csv",
    "links.csv",
    "movies.csv",
    "ratings.csv",
    "tags.csv",
]
files = [os.path.join("./ml-25m/", csv_file) for csv_file in filenames]

# genome-scores.csv
for file in files:
    df = pd.read_csv(file)
    print(f"Metadata of the file {file} is:")
    print(f"    Number of rows: {len(df)}.")
    print(f"    Column names: {df.columns.tolist()}")
    # Detect NAs
    for colname in df.columns:
        NA_count = df[colname].isna().sum()
        if NA_count > 0:
            print(colored(f"        Column {colname} has {df[colname].isna().sum()} missing values;", "red"))
            





