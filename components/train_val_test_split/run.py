#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import os
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    logger.info(f"Fetching artifact {args.input}")
    artifact = run.use_artifact(args.input)
    artifact_dir = artifact.download()

    files = [f for f in os.listdir(artifact_dir) if f.endswith('.csv')]
    artifact_local_path = os.path.join(artifact_dir, files[0])

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files - Windows compatible fix
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")

        # Use delete=False for Windows compatibility
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False
        ) as fp:
            tmp_path = fp.name
            df.to_csv(fp.name, index=False)

        try:
            log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k}_split_of_dataset",
                tmp_path,
                run,
            )
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False
    )

    args = parser.parse_args()
    go(args)
