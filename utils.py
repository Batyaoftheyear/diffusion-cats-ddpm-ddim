import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision.utils import make_grid


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_config_and_args(config, args):
    merged = dict(config)
    for key, value in vars(args).items():
        if value is not None and key in merged:
            merged[key] = value
    return merged


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def tensor_to_uint8(images: torch.Tensor) -> torch.Tensor:
    if images.min() < 0 or images.max() > 1:
        images = (images.clamp(-1, 1) + 1) / 2
    else:
        images = images.clamp(0, 1)
    images = (images * 255).round().to(torch.uint8)
    return images


def save_image_grid(images: torch.Tensor, path, nrow=None):
    images = tensor_to_uint8(images).float() / 255.0
    if nrow is None:
        nrow = int(math.sqrt(len(images))) or 1
    grid = make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    Image.fromarray(grid).save(path)


def save_individual_images(images: torch.Tensor, output_dir, prefix="sample"):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    images = tensor_to_uint8(images).permute(0, 2, 3, 1).cpu().numpy()
    for index, image in enumerate(images):
        Image.fromarray(image).save(output_dir / f"{prefix}_{index:03d}.png")


def save_checkpoint(path, model, config, epoch, best_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model.config),
        "train_config": config,
        "epoch": epoch,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, path)


def append_train_log(csv_path, rows):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "global_step", "step", "loss", "learning_rate"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def plot_loss_curve(csv_path, output_path):
    data = pd.read_csv(csv_path)
    if data.empty:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(data["global_step"], data["loss"], linewidth=1.2)
    plt.xlabel("Global step")
    plt.ylabel("Train loss")
    plt.title("Train loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def append_sampling_metrics(csv_path, record):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    columns = ["method", "steps", "num_images", "batch_size", "seed", "eta", "time_sec", "grid_path"]
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
