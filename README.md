# Tiny pLM: Training tiny protein language models

Tiny masked language model (MLM) pretraining experiments on UniRef50 subsets.

## Setup

```bash
uv sync
```
To install uv, `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Download and split UniRef50 data

Download the first `N` UniRef50 sequences:

```bash
bash scripts/download_uniref50.sh <num_sequences>
```

This writes `data/raw/uniref50_<num_sequences>.fasta`.

Create train/validation/holdout FASTA splits under `data/splits/<split_name>/`:

```bash
uv run python scripts/split_fasta.py <split_name> \
  --train-count <train_count> \
  --validation-count <validation_count> \
  --holdout-count <holdout_count>
```

By default this reads `data/raw/uniref50_<split_name>.fasta`.
The split count flags are required.

## Train

Training are specified in TOML files in `configs/`.
Run training with
```bash
train-tiny --config configs/<config_name>.toml
train-tiny --config configs/40k.toml
train-tiny --config configs/800k.toml
```

## Outputs

Each run writes:
- checkpoint (`.pt`)
- run metrics (`.json`)
- train loss curve (`.svg`)

under the paths defined in the selected TOML config.
