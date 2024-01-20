import wandb


def init(config: dict[str, any]):
    # you will be prompted to enter your API key
    wandb.login()

    wandb.init(project="detrmae", config=config)
    return wandb
