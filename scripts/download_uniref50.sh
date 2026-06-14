#!/usr/bin/env bash

set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <num_sequences>" >&2
  exit 1
fi

NUM_SEQS="$1"
if ! [[ "$NUM_SEQS" =~ ^[0-9]+$ ]] || [ "$NUM_SEQS" -le 0 ]; then
  echo "Error: <num_sequences> must be a positive integer." >&2
  exit 1
fi

OUTPUT_FASTA="data/raw/uniref50_${NUM_SEQS}.fasta"
SOURCE_URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"

mkdir -p "$(dirname "$OUTPUT_FASTA")"

curl -L "$SOURCE_URL" \
  | gunzip -c \
  | awk -v n="$NUM_SEQS" '/^>/{c++; if(c>n) exit} {print}' \
  > "$OUTPUT_FASTA"

echo "Saved up to ${NUM_SEQS} sequences to ${OUTPUT_FASTA}"
