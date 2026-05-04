import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import get_dataloader
from utils import (
    append_train_log,
    ensure_dir,
    load_config,
    merge_config_and_args,
    plot_loss_curve,
    save_checkpoint,
    save_image_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение DDPM на датасете котов")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def build_model(image_size: int):
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def create_preview(unwrapped_model, config, output_dir, epoch, device):
    preview_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"],
    )
    pipeline = DDPMPipeline(unet=unwrapped_model, scheduler=preview_scheduler)
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(config["seed"] + epoch)
    images = pipeline(
        batch_size=config["preview_num_images"],
        generator=generator,
        num_inference_steps=config["num_train_timesteps"],
        output_type="pt",
    ).images
    sample_path = Path(output_dir) / "samples" / f"epoch_{epoch:03d}.png"
    ensure_dir(sample_path.parent)
    save_image_grid(images, sample_path)


def main():
    args = parse_args()
    base_config = load_config(args.config)
    config = merge_config_and_args(base_config, args)
    output_dir = Path(config["output_dir"])

    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "samples")

    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
    )

    if accelerator.is_main_process:
        with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    accelerator.wait_for_everyone()

    if config.get("seed") is not None:
        set_seed(config["seed"])

    dataloader = get_dataloader(
        dataset_name=config["dataset_name"],
        image_size=config["image_size"],
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )

    model = build_model(config["image_size"])
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule=config["beta_schedule"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
        eps=config["adam_epsilon"],
    )

    total_train_steps = len(dataloader) * config["num_epochs"]
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=total_train_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    best_loss = float("inf")
    global_step = 0
    log_path = output_dir / "logs" / "train_log.csv"
    progress_bar = tqdm(
        total=total_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        epoch_rows = []

        for step, batch in enumerate(dataloader, start=1):
            clean_images = batch["pixel_values"]
            noise = torch.randn_like(clean_images)
            batch_size = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=clean_images.device,
                dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_value = loss.detach().item()
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss_value:.4f}")

            epoch_rows.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step": step,
                    "loss": loss_value,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                }
            )

        if accelerator.is_main_process:
            append_train_log(log_path, epoch_rows)
            plot_loss_curve(log_path, output_dir / "loss_curve.png")

            mean_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                unwrapped_model,
                config,
                epoch,
                best_loss=min(best_loss, mean_epoch_loss),
            )

            if mean_epoch_loss < best_loss:
                best_loss = mean_epoch_loss
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    unwrapped_model,
                    config,
                    epoch,
                    best_loss=best_loss,
                )

            if epoch % config["save_model_epochs"] == 0:
                save_checkpoint(
                    output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                    unwrapped_model,
                    config,
                    epoch,
                    best_loss=best_loss,
                )

            if epoch % config["save_image_epochs"] == 0 or epoch == config["num_epochs"]:
                create_preview(
                    unwrapped_model=unwrapped_model,
                    config=config,
                    output_dir=output_dir,
                    epoch=epoch,
                    device=accelerator.device,
                )

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
