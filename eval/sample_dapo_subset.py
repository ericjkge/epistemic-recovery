import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample a reproducible subset from a parquet dataset and save both indices and subset parquet."
    )
    parser.add_argument("--input_parquet", type=str, required=True, help="Path to source parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--sample_size", type=int, default=1500, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    parser.add_argument(
        "--prefix",
        type=str,
        default="train_random1500_seed0",
        help="Output filename prefix for the indices json and subset parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.input_parquet)
    total_rows = len(df)
    if args.sample_size > total_rows:
        raise ValueError(f"Requested sample_size={args.sample_size}, but dataset only has {total_rows} rows")

    rng = np.random.default_rng(args.seed)
    sampled_indices = np.sort(rng.choice(total_rows, size=args.sample_size, replace=False))
    sampled_df = df.iloc[sampled_indices].copy()
    sampled_df["original_index"] = sampled_indices

    os.makedirs(args.output_dir, exist_ok=True)

    indices_path = os.path.join(args.output_dir, f"{args.prefix}_indices.json")
    subset_path = os.path.join(args.output_dir, f"{args.prefix}.parquet")

    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_parquet": os.path.abspath(args.input_parquet),
                "sample_size": args.sample_size,
                "seed": args.seed,
                "indices": sampled_indices.tolist(),
            },
            f,
            indent=2,
        )

    sampled_df.to_parquet(subset_path, index=False)

    print(f"Loaded rows: {total_rows}")
    print(f"Sampled rows: {len(sampled_df)}")
    print(f"Saved indices: {indices_path}")
    print(f"Saved subset: {subset_path}")


if __name__ == "__main__":
    main()
