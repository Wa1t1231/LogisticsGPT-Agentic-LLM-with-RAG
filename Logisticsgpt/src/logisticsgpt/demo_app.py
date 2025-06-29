"""Command‑line demo of LogisticsGPT capabilities."""

import argparse
import json
from pprint import pprint

import pandas as pd

from logisticsgpt import ask_rag, sql_query, sql_write, plot_dataframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["rag", "read", "write", "plot"], required=True)
    parser.add_argument("--arg", help="Question / SQL / CSV path depending on task", required=True)
    args = parser.parse_args()

    if args.task == "rag":
        ans = ask_rag(args.arg)
        print(ans)
    elif args.task == "read":
        rows = sql_query(args.arg)
        pprint(rows)
    elif args.task == "write":
        res = sql_write(args.arg)
        print(res)
    elif args.task == "plot":
        df = pd.read_csv(args.arg)
        img = plot_dataframe(df, x=df.columns[0], y=df.columns[1], title="Auto chart")
        print(f"Chart saved → {img}")


if __name__ == "__main__":
    main()
