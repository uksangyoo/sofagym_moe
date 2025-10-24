import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from typing import Optional
import argparse
import copy
import numpy as np


def main(config: Optional[dict] = None, max_steps: int = 20, visualize: bool = False, action_scale: float = 0.5):
    try:
        from sofagym_moe.envs.CrawlingMOE.CrawlingMOEEnv import CrawlingMOEEnv
    except Exception as e:
        print("Could not import CrawlingMOEEnv:", e)
        print("Ensure this package is installed and that 'sofagym' is available.")
        return

    try:
        cfg = copy.deepcopy(config) if config is not None else {}
        if visualize:
            cfg.update({"render": 1})

        env = CrawlingMOEEnv(config=cfg, use_server=False)
    except Exception as e:
        print("Failed to create CrawlingMOEEnv instance:", e)
        return

    try:
        action_space = env.env.action_space
    except Exception:
        action_space = getattr(env, "action_space", None)

    if action_space is None:
        print("Environment has no accessible action_space. Aborting example.")
        return

    print("Resetting environment...")
    state = env.reset()
    print("Initial state shape:", None if state is None else getattr(state, 'shape', None))

    if visualize:
        try:
            env.render('human')
        except Exception as e:
            print("Warning: render failed:", e)

    for step in range(max_steps):
        raw_action = action_space.sample()
        if hasattr(action_space, 'shape'):
            action = np.array(raw_action, dtype=np.float32) * float(action_scale)
            try:
                action = np.clip(action, action_space.low, action_space.high)
            except Exception:
                pass
        else:
            action = raw_action
            if float(action_scale) != 1.0:
                try:
                    action = int(round(action * float(action_scale)))
                except Exception:
                    pass
        try:
            result = env.step(action)
        except Exception as e:
            print(f"Step failed at step {step}:", e)
            break

        if isinstance(result, tuple) and len(result) >= 3:
            state, reward, done = result[0], result[1], result[2]
            info = result[3] if len(result) > 3 else {}
            print(f"Step {step}: reward={reward}, done={done}")
            if done:
                print("Episode finished after", step + 1, "steps")
                break
        else:
            print(f"Step {step}: result={result}")

        if visualize:
            try:
                env.render('human')
            except Exception as e:
                print("Warning: render failed during step:", e)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run random actions in CrawlingMOE with incremental force control")
    parser.add_argument("--steps", type=int, default=20, help="number of steps to run")
    parser.add_argument("--visualize", action="store_true", help="enable visualization")
    parser.add_argument("--action-scale", type=float, default=0.5, help="scale for random actions")
    args = parser.parse_args()
    main(max_steps=args.steps, visualize=args.visualize, action_scale=args.action_scale)
