from diffusers import UNet2DModel


def build_model(config=None, model_config=None):
    if model_config is not None:
        return UNet2DModel.from_config(model_config)

    if config is None:
        raise ValueError("Нужен config или model_config для создания модели.")

    return UNet2DModel(
        sample_size=config["image_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        layers_per_block=config["layers_per_block"],
        block_out_channels=tuple(config["block_out_channels"]),
        down_block_types=tuple(config["down_block_types"]),
        up_block_types=tuple(config["up_block_types"]),
    )
