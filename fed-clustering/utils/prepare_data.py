import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_hash_experiment

HASH_EXPERIMENT = get_hash_experiment()


def prepare_data(
    input_csv: str,
    output_dir: str,
    randomize: bool = False,
    filename: Optional[str] = None,
    file_format="csv",
):
    # Load data from CSV
    df = pd.read_csv(input_csv, header=0, low_memory=False)
    df = df.fillna(-1)
    df = df.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN

    df = df.dropna(axis=1)  # Drop columns with NaN

    ids = df.iloc[:, 0].values
    x = df.iloc[:, 3:].values

    if randomize:
        np.random.seed(0)
        idx_random = np.random.permutation(len(ids))
        x = x[idx_random, :]
        ids = ids[idx_random]

    # Combine back into a DataFrame
    data = np.column_stack((ids, x))
    # remove header
    x = x[1:]
    # convert boolean to int
    x = x.astype(float)

    df = pd.DataFrame(data=x)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to csv file
    filename = filename if filename else "processed_data.csv"
    if file_format == "csv":
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, sep=",", index=False, header=False)
    else:
        raise NotImplementedError("Only CSV format is supported.")


def main():
    parser = argparse.ArgumentParser(description="Read CSV data and process it")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--randomize", type=int, help="Whether to randomize data sequence"
    )
    parser.add_argument("--out_path", type=str, help="Path to output data file")
    args = parser.parse_args()

    from time import perf_counter
    from dfa_lib_python.task import Task
    from dfa_lib_python.dataset import DataSet
    from dfa_lib_python.element import Element

    dataflow_tag = "nvidiaflare-df"

    t1 = Task(1, dataflow_tag, "PrepareData")
    t1.begin()
    start = perf_counter()

    output_dir = os.path.dirname(args.out_path)
    filename = os.path.basename(args.out_path)

    prepare_data(args.input_csv, output_dir, args.randomize, filename)

    duration = perf_counter() - start

    to_dfanalyzer = [
        HASH_EXPERIMENT,
        args.input_csv,
        args.randomize,
        args.out_path,
        duration,
    ]
    t1_input = DataSet("iPrepareData", [Element(to_dfanalyzer)])
    t1.add_dataset(t1_input)
    t1_output = DataSet("oPrepareData", [Element([])])
    t1.add_dataset(t1_output)
    t1.end()


if __name__ == "__main__":
    main()
