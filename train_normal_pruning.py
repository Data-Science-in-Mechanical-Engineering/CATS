from functools import partial
import logging
import hydra
import omegaconf

from execution_guard import execution_guard
from train import train

@hydra.main(config_path="parameters", config_name="main", version_base="1.1")
@partial(execution_guard, force_overwrite=True)
def main(cfg: omegaconf.DictConfig):
    logger = logging.getLogger("execution_guard")
    
    num_pruning_steps = int(round(1/cfg.per_step_pruning_ratio))
    if num_pruning_steps * cfg.per_step_pruning_ratio < 1.0:
        num_pruning_steps += 1
    num_pruning_steps += 1  # extra step to reach zero pruned model
        
    num_features_attention_orig = cfg.model.num_features_attention
    num_features_residual_orig = cfg.model.num_features_residual
    num_features_head_orig = cfg.model.num_features_head

    for i in range(num_pruning_steps):
        logger.info("\n"*10)
        logger.info(f"Pruning step {i+1}/{num_pruning_steps}")
        
        cfg.model.num_features_attention = max(1, int(num_features_attention_orig * (1 - cfg.per_step_pruning_ratio * i)))
        cfg.model.num_features_attention += (cfg.model.num_attention_heads - cfg.model.num_features_attention % cfg.model.num_attention_heads) % cfg.model.num_attention_heads  # make divisible by num heads
        cfg.model.num_features_residual = max(1, int(num_features_residual_orig * (1 - cfg.per_step_pruning_ratio * i)))
        cfg.model.num_features_head = max(1, int(num_features_head_orig * (1 - cfg.per_step_pruning_ratio * i)))
        print(f"  num_features_attention: {cfg.model.num_features_attention}, {cfg.model.num_features_attention%cfg.model.num_attention_heads}")

        train(cfg, logger)


if __name__ == "__main__":
    main()
