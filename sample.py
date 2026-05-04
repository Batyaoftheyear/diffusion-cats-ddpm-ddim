import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from tqdm.auto import tqdm

from utils import (
    append_sampling_metrics,
    ensure_dir,
    save_image_grid,
    save_individual_images,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Генерация изображений из обученной модели")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["ddpm", "ddim"], required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--save_individual", action="store_true")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet2DModel.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def build_scheduler(method, checkpoint, steps):
    train_config = checkpoint["train_config"]
    default_steps = 1000 if method == "ddpm" else 100
    num_inference_steps = steps or default_steps
    scheduler_cls = DDPMScheduler if method == "ddpm" else DDIMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=train_config["num_train_timesteps"],
        beta_schedule=train_config["beta_schedule"],
    )
    scheduler.set_timesteps(num_inference_steps)
    return scheduler, num_inference_steps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    train_config = checkpoint["train_config"]
    scheduler, num_inference_steps = build_scheduler(args.method, checkpoint, args.steps)

    if args.output_dir is None:
        output_dir = Path(train_config["output_dir"]) / "sampling" / f"{args.method}_{num_inference_steps}"
    else:
        output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "images")
    ensure_dir(output_dir.parent / "metrics")

    all_images = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.perf_counter()
    remaining = args.num_images
    while remaining > 0:
        current_batch_size = min(args.batch_size, remaining)
        batch_scheduler, _ = build_scheduler(args.method, checkpoint, num_inference_steps)
        if generator.device.type == "cuda":
            latents = torch.randn(
                (current_batch_size, 3, train_config["image_size"], train_config["image_size"]),
                generator=generator,
                device=device,
            )
        else:
            latents = torch.randn(
                (current_batch_size, 3, train_config["image_size"], train_config["image_size"]),
                generator=generator,
                device=device,
            )
        for timestep in tqdm(batch_scheduler.timesteps, desc=f"{args.method.upper()} sampling", leave=False):
            model_output = model(latents, timestep).sample
            step_kwargs = {"eta": args.eta} if isinstance(batch_scheduler, DDIMScheduler) else {}
            latents = batch_scheduler.step(model_output, timestep, latents, **step_kwargs).prev_sample
        all_images.append(latents.cpu())
        remaining -= current_batch_size

    total_time = time.perf_counter() - start_time
    images = torch.cat(all_images, dim=0)

    grid_path = output_dir / "grid.png"
    save_image_grid(images, grid_path)

    if args.save_individual:
        save_individual_images(images, output_dir / "images")

    metrics_path = output_dir.parent / "metrics" / "sampling_comparison.csv"
    append_sampling_metrics(
        metrics_path,
        {
            "method": args.method,
            "steps": num_inference_steps,
            "num_images": args.num_images,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "eta": args.eta if args.method == "ddim" else "",
            "time_sec": round(total_time, 4),
            "grid_path": str(grid_path).replace("\\", "/"),
        },
    )

    print(f"Method: {args.method}")
    print(f"Steps: {num_inference_steps}")
    print(f"Images: {args.num_images}")
    print(f"Time: {total_time:.4f} sec")
    print(f"Grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
