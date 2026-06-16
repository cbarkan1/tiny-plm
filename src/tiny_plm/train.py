from pathlib import Path

import torch
import torch.nn as nn
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYXBZUO"
SPECIAL_TOKENS = ("<pad>", "<mask>", "<unk>")
TOKENS = SPECIAL_TOKENS + tuple(AMINO_ACIDS)
STOI = {token: i for i, token in enumerate(TOKENS)}
ITOS = {i: token for token, i in STOI.items()}

IGNORE_INDEX = -100


class MLMDataset(Dataset):
    def __init__(
        self,
        fasta_path: str | Path,
        max_len: int = 256,
        mask_prob: float = 0.15,
    ) -> None:
        self.records = list(SeqIO.parse(str(fasta_path), "fasta"))
        self.stoi = STOI
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_id = STOI["<pad>"]
        self.mask_id = STOI["<mask>"]
        self.unk_id = STOI["<unk>"]
        self.vocab_size = len(STOI)

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in sequence[: self.max_len]]
        n = len(ids)
        pad = self.max_len - n
        
        # We don't need .to(device) here because Dataset __getitem__/_encode methods are routinely called on the CPU,
        # before data is handed off to the DataLoader/collate_fn and then transferred to the device (e.g., GPU) as a batch.
        # This keeps data loading and preprocessing efficient and avoids unnecessary device transfers for each item.
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

        # labels will become the ground truth for training. All non-masked positions will be ignored,
        # as training is only over masked positions
        labels = input_ids.clone()

        rand = torch.rand(input_ids.shape)
        maskable = attention_mask.bool() # Don't mask <pad> tokens

        # The '&' operator performs elementwise logical AND on boolean tensors; 
        # Python's built-in 'and' cannot be used because it only works on single booleans, not tensors.
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
        """
        Returns dict with datapoint (input_ids) and ground truth label from which loss will be computed.
        In this case, the datapoint is the sequence with mlm applied (some positions 
        masked, others mutated randomly, others unchanged) and the ground truth label
        is the original sequence with non-maskable positions ignored (because those aren't
        used for computing loss).
        """
        sequence = str(self.records[idx].seq)
        input_ids, attention_mask = self._encode(sequence)
        input_ids, labels = self._apply_mlm(input_ids, attention_mask)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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

        # .to(device) is needed for tensors that will be processed by the model or used in computations on a specific device.
        # In general, use .to(device) for all input data before passing it to the model or performing calculations involving tensors.
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Using set_to_none=True isn't strictly required but can be more efficient
        optimizer.zero_grad(set_to_none=True)

        # Forward pass to build the computation graph needed to calculate gradients
        # Recall that attention_mask here is just to prevent <pad> tokens from being
        # used in the attention calculations
        logits = model(input_ids, attention_mask=attention_mask)

        # Flatten logits and labels and compute loss
        batch_size, seq_len, vocab_size = logits.shape
        loss = loss_fn(logits.view(batch_size*seq_len, vocab_size), labels.view(batch_size*seq_len))
        
        loss.backward()

        # model.parameters() is an iterator over a series of objects containing the gradient info
        # It mutates each of those in place in order to scale down the norm of the gradinet if
        # the norm is over 1.
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
    model.eval() # tells model not to use dropout
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    total_steps = 0

    iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        batch_size, seq_len, vocab_size = logits.shape
        loss = loss_fn(logits.view(batch_size*seq_len, vocab_size), labels.view(batch_size*seq_len))
        total_loss += loss.item()
        total_steps += 1
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_steps, 1)
