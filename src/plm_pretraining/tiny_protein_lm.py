import torch
import torch.nn as nn


class TinyProteinLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # input_ids: [B, T]
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # src_key_padding_mask expects True where positions are PAD
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits