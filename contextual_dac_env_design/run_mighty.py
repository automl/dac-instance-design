from __future__ import annotations

import time

import hydra
import numpy as np
from mighty.mighty_runners.factory import get_runner_class


@hydra.main("./configs", "base", version_base=None)
def run_mighty(cfg):
    # Make runner
    print(cfg)
    runner_cls = get_runner_class(cfg.runner)
    runner = runner_cls(cfg)

    # Execute run
    start = time.time()
    train_result, eval_result = runner.run()
    end = time.time()

    # Print stats
    print("Training finished!")
    print(
        f"Reached a reward of {np.round(eval_result['mean_eval_reward'], decimals=2)} in {train_result['step']} steps and {np.round(end-start,decimals=2)}s."
    )


if __name__ == "__main__":
    run_mighty()
