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
        
        # nn.Embedding is similar to a linear layer (if you one-hot encode the vocab, 
        # then it truly is a linear layer). It's a vector of dim d_model for each vocab
        # token, and the vector components are learnable. But the components are
        # initialized differently than for a linear layer.
        # The embedding of the padding token is not learnable, and we specify the
        # padding token index with padding_idx
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # The original transformers paper used sinusoidal functions for embedding
        # but now people often use learning embedding vectors.
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Performs dropout (during training only; automatically turns off for model.eval())
        self.dropout = nn.Dropout(dropout)

        # nn.TransformerEncoderLayer defaults to using an FFN with one hidden layer
        # dim_feedforward is the width for that hidden layer
        # batch_first=True puts batch dimension first, so indexing is [B, T] or [B, T, V]
        # norm_first=True puts the norm layer before attention and before FFN
        # Note that we have dropouts set within the transformer, and separate dropout
        # for the initial embeddings (self.dropout above).
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

        # self.norm will be to normalize the final hidden state, right before decoding
        self.norm = nn.LayerNorm(d_model)

        # lm_head is a linear layer to map the final hidden state into the logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        attention_mask masks <pad> tokens, it has nothing to do with <mask> tokens!!

        Note: the attention_mask terminology is HuggingFace standard, so we'll keep it here,
        but it's confusing in this masked LM project where <mask> has a separate meaning.
        """

        # input_ids: [B, T]
        # bsz is short for "batch size"
        bsz, seq_len = input_ids.shape

        # This builds a new torch.Tensor, so it needs to know the device in order to properly
        # build the tensor. This sets the device to whatever device was used to build input_ids
        # unsqueeze(0) adds a batch dimension at the 0th position (because we set batch_first=True)
        # expand(bsz, seq_len) copies position into bsz new identical rows, so we end up
        # with a (bsz, seq_len)-shaped tensor where each row is an identical list of integers
        # from 0 to seq_len-1.
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # Dropout zeros out a specified fraction of the activations, and scales up the remaining
        # activations so the norm of the resulting vector is unchanged.
        x = self.dropout(x)

        # add the attention mask for <pad> tokens (not for <mask> tokens!)
        # src_key_padding_mask expects True where positions are PAD
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits