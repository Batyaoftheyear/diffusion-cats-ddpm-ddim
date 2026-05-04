from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class ImageTransform:
    def __init__(self, image_size: int):
        self.transform = build_transforms(image_size)

    def __call__(self, example):
        images = example["image"]
        if isinstance(images, list):
            example["pixel_values"] = [self.transform(image.convert("RGB")) for image in images]
        else:
            example["pixel_values"] = self.transform(images.convert("RGB"))
        return example


def build_transforms(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def get_dataset(dataset_name: str, image_size: int, split: str = "train"):
    dataset = load_dataset(dataset_name, split=split)
    return dataset.with_transform(ImageTransform(image_size))


def collate_fn(examples):
    pixel_values = [example["pixel_values"] for example in examples]
    return {"pixel_values": torch.stack(pixel_values)}


def get_dataloader(
    dataset_name: str,
    image_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
):
    dataset = get_dataset(dataset_name=dataset_name, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
