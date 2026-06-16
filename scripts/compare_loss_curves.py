import matplotlib.pyplot as plt

import json
from pathlib import Path

def load_train_loss(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["train_loss_steps"], data["split_name"]

json_paths = [
    Path("artifacts/800k/800k_1epoch_metrics.json"),
    Path("artifacts/40k/40k_1epoch_metrics.json"),
    Path("artifacts/40k_s2/1epoch_metrics.json"),
    Path("artifacts/5k/1epoch_metrics.json"),
]

plt.figure(figsize=(10, 6))
for jp in json_paths:
    curve, label = load_train_loss(jp)
    plt.plot(curve, label=label)

plt.xlabel("Training step")
plt.ylabel("Train loss")
plt.title("Comparison of train loss curves")
plt.legend()
plt.tight_layout()
plt.show()