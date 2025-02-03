import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="config",
    config_name="train_config",
    version_base=None,
)
def main(cfg: DictConfig):
    print(cfg.prott5)


if __name__ == "__main__":
    main()
