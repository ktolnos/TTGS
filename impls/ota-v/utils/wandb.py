import ml_collections


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            project="OTA",
            group="Debug",
            mode="online",
        )
    )
    return config
