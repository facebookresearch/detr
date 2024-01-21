import wandb


def init(config: dict[str, any]):
    # you will be prompted to enter your API key
    wandb.login()

    return wandb.init(project="detrmae", entity="dermae", config=config)
