import argparse
from pathlib import Path

import numpy as np
from Bio import SeqIO


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/validation/holdout FASTA splits.")
    parser.add_argument("split_name", type=str, help="Name of split directory under data/splits/")
    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=None,
        help="Path to source FASTA. Defaults to data/raw/uniref50_<split_name>.fasta",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        required=True,
        help="Number of train sequences",
    )
    parser.add_argument(
        "--validation-count",
        type=int,
        required=True,
        help="Number of validation sequences",
    )
    parser.add_argument(
        "--holdout-count",
        type=int,
        required=True,
        help="Number of holdout sequences",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_name = args.split_name
    input_fasta = args.input_fasta or Path(f"data/raw/uniref50_{split_name}.fasta")
    output_dir = Path("data/splits") / split_name
    train_count = args.train_count
    validation_count = args.validation_count
    holdout_count = args.holdout_count

    if train_count < 0 or validation_count < 0 or holdout_count < 0:
        raise SystemExit("train/validation/holdout counts must be >= 0.")
    if train_count + validation_count + holdout_count == 0:
        raise SystemExit("At least one split count must be > 0.")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(input_fasta, "fasta"))
    target_total = train_count + validation_count + holdout_count

    if len(records) < target_total:
        raise SystemExit(
            f"Not enough sequences in {input_fasta}. "
            f"Need {target_total}, found {len(records)}."
        )

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(records))[:target_total]
    selected = [records[i] for i in indices]

    train_end = train_count
    validation_end = train_end + validation_count

    train_records = selected[:train_end]
    validation_records = selected[train_end:validation_end]
    holdout_records = selected[validation_end:]

    train_path = output_dir / "train.fasta"
    validation_path = output_dir / "validation.fasta"
    holdout_path = output_dir / "holdout.fasta"

    SeqIO.write(train_records, train_path, "fasta")
    SeqIO.write(validation_records, validation_path, "fasta")
    SeqIO.write(holdout_records, holdout_path, "fasta")

    print(f"Wrote {len(train_records)} sequences to {train_path}")
    print(f"Wrote {len(validation_records)} sequences to {validation_path}")
    print(f"Wrote {len(holdout_records)} sequences to {holdout_path}")


if __name__ == "__main__":
    main()