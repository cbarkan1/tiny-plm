from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_loss_plot(losses: list[float], output_path: Path) -> None:
    if not losses:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Train loss")
    plt.title(f"Train loss over steps (n={len(losses)})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
