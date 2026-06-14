from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXBZUO"
SPECIAL_TOKENS = ("<pad>", "<mask>", "<unk>")

IGNORE_INDEX = -100

def build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    tokens = list(SPECIAL_TOKENS) + list(AMINO_ACIDS)
    stoi = {token: i for i, token in enumerate(tokens)}
    itos = {i: token for token, i in stoi.items()}
    return stoi, itos


@dataclass(frozen=True)
class TrainConfig:
    max_len: int = 256
    batch_size: int = 16
    mask_prob: float = 0.15
    lr: float = 2e-4
    weight_decay: float = 0.01 # L2 regularizer, shrinks all weights by this factor each step


class MLMDataset(Dataset):
    def __init__(
        self,
        fasta_path: str | Path,
        stoi: dict[str, int],
        max_len: int = 256,
        mask_prob: float = 0.15,
    ) -> None:
        self.records = list(SeqIO.parse(str(fasta_path), "fasta"))
        self.stoi = stoi
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_id = stoi["<pad>"]
        self.mask_id = stoi["<mask>"]
        self.unk_id = stoi["<unk>"]
        self.vocab_size = len(stoi)

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in sequence[: self.max_len]]
        n = len(ids)
        pad = self.max_len - n
        input_ids = torch.tensor(ids + [self.pad_id] * pad, dtype=torch.long)
        attention_mask = torch.tensor([1] * n + [0] * pad, dtype=torch.long)
        return input_ids, attention_mask

    def _apply_mlm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT-style masked language modeling to one encoded sequence.

        Maskable positions are selected with probability ``self.mask_prob`` where
        ``attention_mask == 1`` (non-padding tokens). For selected positions:
        80% are replaced with ``<mask>``, 10% with a random token, and 10%
        are left unchanged (this strategy was used for ESM2). The returned labels 
        keep original token IDs at masked positions and use IGNORE_INDEX elsewhere 
        for loss ignore.
        """
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        maskable = attention_mask.bool()
        masked_positions = (rand < self.mask_prob) & maskable

        # The ~ operator inverts the boolean mask, so here it selects positions that are NOT masked.
        # For all non-masked positions, set the label to IGNORE_INDEX so they are ignored in 
        # loss computation.
        labels[~masked_positions] = IGNORE_INDEX
        if masked_positions.sum() == 0:
            return input_ids, labels

        probs = torch.rand(input_ids.shape)

        replace_with_mask = (probs < 0.8) & masked_positions
        replace_with_random = (probs >= 0.8) & (probs < 0.9) & masked_positions

        masked_input_ids = input_ids.clone()
        masked_input_ids[replace_with_mask] = self.mask_id
        random_ids = torch.randint(0, self.vocab_size, input_ids.shape, dtype=torch.long)
        masked_input_ids[replace_with_random] = random_ids[replace_with_random]
        return masked_input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = str(self.records[idx].seq)
        input_ids, attention_mask = self._encode(sequence)
        input_ids, labels = self._apply_mlm(input_ids, attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloader(
    fasta_path: str | Path,
    stoi: dict[str, int],
    batch_size: int = 16,
    max_len: int = 256,
    mask_prob: float = 0.15,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MLMDataset(
        fasta_path=fasta_path,
        stoi=stoi,
        max_len=max_len,
        mask_prob=mask_prob,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_desc: str = "train",
) -> tuple[float, list[float]]:
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    total_steps = 0
    batch_losses: list[float] = []

    iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask=attention_mask)
        # Compute the cross-entropy loss by flattening logits and labels so each token is a prediction;
        # logits are reshaped to (batch_size * seq_len, vocab_size), labels to (batch_size * seq_len)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        total_steps += 1
        batch_losses.append(batch_loss)
        iterator.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / max(total_steps, 1), batch_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    progress_desc: str = "val",
) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    total_steps = 0

    iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        total_steps += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_steps, 1)
