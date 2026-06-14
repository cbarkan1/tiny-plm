import torch
from plm_pretraining.tiny_protein_lm import TinyProteinLM
torch.manual_seed(42)

# Toy vocab
tokens = ["<pad>", "<mask>"] + list("ACDEFGHIKLMNPQRSTVWY")
stoi = {t: i for i, t in enumerate(tokens)}
itos = {i: t for t, i in stoi.items()}

pad_id = stoi["<pad>"]

def encode(seq: str, max_len: int = 16):
    """returns encoded sequence, attention mask"""
    ids = list(map(stoi.get, seq[:max_len]))
    n = len(ids)
    pad = max_len - n # Add pad tokens so all sequences reach desired length
    return ids + [pad_id] * pad, [1] * n + [0] * pad

seqs = ["MKTW", "ACDEFGHIK"]
encoded = [encode(s, max_len=16) for s in seqs]
input_ids = torch.tensor([x[0] for x in encoded], dtype=torch.long)
attention_mask = torch.tensor([x[1] for x in encoded], dtype=torch.long)

# PyTorch initializes weights when model object is created
model = TinyProteinLM(
    vocab_size=len(tokens),
    max_len=16,
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=2,
    dropout=0.1,
    pad_token_id=pad_id,
)

model.eval() # switch to eval mode
with torch.no_grad():
    logits = model(input_ids, attention_mask=attention_mask)

print("logits shape:", logits.shape)  # [B, T, V]
pred_ids = logits.argmax(dim=-1) # Temp = 0
print("pred ids:", pred_ids)
print("first seq first 8 preds:", [itos[i.item()] for i in pred_ids[0, :8]])
