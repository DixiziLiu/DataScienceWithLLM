#!/usr/bin/env python3

import pandas as pd
import os
import sys

def read_and_process_data(directory):
    try:
        # Read csvs
        # First, get the number of lines in the CSV file
        dfs = []
        for i in [0,1,2,3]:
            filename = os.path.join(directory, f'stratified_data_splits/{str(i+1)}/augments.csv')
            with open(filename) as f:
                total_lines = sum(1 for line in f)

            # Then, load the CSV into a DataFrame, excluding the last 5 lines
            df = pd.read_csv(filename, on_bad_lines='skip')

            dfs.append(df)

        # Drop empty rows and rows where text is just a bunch of # signs
        dfs = [df[df.text.notna() & (~df.text.str.contains("^#+$", na=False, regex=True))] for df in dfs]

        for i, df in enumerate(dfs):
            # Group by model_id and temperature and count the size of each group
            counts = df.groupby(['model_id', 'temperature']).size().unstack().fillna(0).astype(int)

            # Print the result
            print(f'Stratified training fold {i+1}')
            print(counts)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Optionally accept directory as argument
    directory = './data' if len(sys.argv) < 2 else sys.argv[1]
    read_and_process_data(directory)
