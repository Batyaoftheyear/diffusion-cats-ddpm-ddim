import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from tqdm.auto import tqdm

from models import build_model
from utils import (
    append_sampling_metrics,
    ensure_dir,
    load_config,
    save_image_grid,
    save_individual_images,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Генерация изображений из обученной модели")
    parser.add_argument("--config", type=str, default="config.yaml")
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
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = build_model(model_config=checkpoint["model_config"])
    load_result = model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"Missing keys: {load_result.missing_keys}")
        print(f"Unexpected keys: {load_result.unexpected_keys}")
        raise RuntimeError("Checkpoint несовместим с архитектурой модели.")

    print("Checkpoint loaded with strict key match.")

    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device)
    model.eval()
    return model, checkpoint


def build_scheduler(method, effective_config, steps):
    default_steps = (
        effective_config["ddpm_default_steps"]
        if method == "ddpm"
        else effective_config["ddim_default_steps"]
    )
    num_inference_steps = steps or default_steps
    scheduler_cls = DDPMScheduler if method == "ddpm" else DDIMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=effective_config["num_train_timesteps"],
        beta_schedule=effective_config["beta_schedule"],
        prediction_type=effective_config["prediction_type"],
    )
    scheduler.set_timesteps(num_inference_steps)
    return scheduler, num_inference_steps


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def main():
    args = parse_args()
    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    effective_config = dict(base_config)
    effective_config.update(checkpoint["train_config"])
    _, num_inference_steps = build_scheduler(args.method, effective_config, args.steps)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if args.output_dir is None:
        output_dir = Path(effective_config["output_dir"]) / "sampling" / f"{args.method}_{num_inference_steps}"
    else:
        output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "images")
    ensure_dir(output_dir.parent / "metrics")

    print(f"Device: {device}")
    print(f"Model parameter count: {parameter_count}")
    print(f"Sampling method: {args.method}")
    print(f"Sampling steps: {num_inference_steps}")
    print(f"Prediction type: {effective_config['prediction_type']}")

    all_images = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    start_time = time.perf_counter()
    remaining = args.num_images
    while remaining > 0:
        current_batch_size = min(args.batch_size, remaining)
        batch_scheduler, _ = build_scheduler(args.method, effective_config, num_inference_steps)
        latent_dtype = torch.float16 if device.type == "cuda" else torch.float32
        sample = torch.randn(
            (current_batch_size, effective_config["in_channels"], effective_config["image_size"], effective_config["image_size"]),
            generator=generator,
            device=device,
            dtype=latent_dtype,
        )

        with torch.inference_mode():
            with get_autocast_context(device):
                for timestep in tqdm(batch_scheduler.timesteps, desc=f"{args.method.upper()} sampling", leave=False):
                    model_output = model(sample, timestep).sample
                    step_kwargs = {"eta": args.eta} if isinstance(batch_scheduler, DDIMScheduler) else {}
                    sample = batch_scheduler.step(model_output, timestep, sample, **step_kwargs).prev_sample

        all_images.append(sample.float().cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
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

    print(f"Images: {args.num_images}")
    print(f"Time: {total_time:.4f} sec")
    print(f"Grid saved to: {grid_path}")


if __name__ == "__main__":
    main()
