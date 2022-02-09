import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import lht_utils

    OmegaConf.resolve(config)
    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    lht_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        lht_utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
