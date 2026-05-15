from functools import partial
import logging
import hydra
import omegaconf

from execution_guard import execution_guard
from train import Trainer


@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
@partial(execution_guard, force_overwrite=True)
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger("execution_guard")
    
    num_pruning_steps = int(round(1/cfg.per_step_pruning_ratio))
    if num_pruning_steps * cfg.per_step_pruning_ratio < 1.0:
        num_pruning_steps += 1
    num_pruning_steps += 1  # extra step to reach zero pruned model

    trainer = Trainer(cfg, logger)

    for i in range(0, num_pruning_steps):
        logger.info("\n"*10)
        logger.info(f"Pruning step {i+1}/{num_pruning_steps}")
        
        trainer.train(pruning_step=i)


if __name__ == "__main__":
    main()
