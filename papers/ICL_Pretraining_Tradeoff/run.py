import jax
import hydra
from omegaconf import DictConfig
import logging
from ml_collections import ConfigDict

import icl.utils as u
from icl.train import train

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
# Reduce verbosity of third-party libraries
logging.getLogger('orbax').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('flax').setLevel(logging.WARNING)

# Also configure absl logging (used by Orbax)
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.WARNING)


def omega_to_ml_collections(cfg: DictConfig) -> ConfigDict:
    """Convert OmegaConf DictConfig to ml_collections ConfigDict."""
    def _convert(obj):
        if isinstance(obj, DictConfig):
            config = ConfigDict()
            for key, value in obj.items():
                config[key] = _convert(value)
            return config
        else:
            return obj
    
    return _convert(cfg)


@hydra.main(version_base=None, config_path="icl/configs", config_name="plain")
def main(cfg: DictConfig) -> None:
    logging.info(f"Process: {jax.process_index() } / {jax.process_count()}")
    logging.info("Local Devices:\n" + "\n".join([str(x) for x in jax.local_devices()]) + "\n")

    config = omega_to_ml_collections(cfg)
    config = u.filter_config(config)
    train(config)


if __name__ == "__main__":
    main()
