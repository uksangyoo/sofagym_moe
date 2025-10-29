# sofagym_moe
 MOE-family (https://proceedings.mlr.press/v305/yoo25a.html) soft robotics environments for reinforcement learning.


## Environments

> **Primary (recommended): MultiFingerMOE**
> 
> Actively supported — a multi-finger gripper with 3 soft fingers (6D action space).

- **MultiFingerMOE**: Multi-finger gripper with 3 soft fingers (6D action space)
- **CrawlingMOE**: Single soft finger crawling robot (2D action space)
- **MOE**: Discrete action gripper manipulation
- **MOEGripper**: Gripper with discrete actions

## Prerequisites

Before installation, ensure you have:

1. **SOFA Framework** (v23 or later): https://www.sofa-framework.org/
2. **SofaPython3** plugin: https://github.com/sofa-framework/SofaPython3
3. **SoftRobots** plugin: https://github.com/SofaDefrost/SoftRobots
4. **Python 3.8+** with pip

## Installation

### Step 1: Install base sofagym framework

```bash
# Clone and install sofagym (required dependency)
git clone https://github.com/SofaDefrost/SofaGym.git
cd SofaGym
pip install -e .
```

### Step 2: Install sofagym_moe package

Extract or clone the `sofagym_moe` package, then install from the **parent directory**:

```bash
# If the package is at ~/sofagym_moe/, run from ~/
cd ~  # or wherever the parent of sofagym_moe is
pip install -e .
```

**Important**: The `pyproject.toml` file should be in the parent directory of `sofagym_moe/`, not inside it.

### Verify installation

```bash
python3 -c "import sofagym_moe; print('Success! Version:', sofagym_moe.__version__)"
```## Quick Start

```python
from sofagym_moe.envs.MultiFingerMOE.MultiFingerMOEEnv import MultiFingerMOEEnv
import numpy as np

env = MultiFingerMOEEnv(config={"render": 0})
state = env.reset()

for _ in range(100):
    action = np.random.uniform(-0.5, 0.5, size=6)
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

## Examples

Run included examples from the `examples/` directory:

```bash
cd sofagym_moe/examples

# Run multi-finger gripper with visualization
python3 run_random_multifinger.py --visualize --steps 50

# Run crawling robot
python3 run_random_crawling.py --visualize --steps 50

# Quick test without visualization
python3 run_random_multifinger.py --steps 10
```

## Package Structure

```
sofagym_moe/
├── __init__.py
├── register.py
├── README.md
├── INCREMENTAL_ACTIONS.md
├── pyproject.toml          # Should be in parent directory!
├── envs/
│   ├── MultiFingerMOE/
│   │   ├── __init__.py
│   │   ├── MultiFingerMOEEnv.py
│   │   ├── MultiFingerMOEScene.py
│   │   ├── MultiFingerMOEToolbox.py
│   │   └── mesh/
│   ├── CrawlingMOE/
│   │   ├── __init__.py
│   │   └── ...
│   ├── MOE/
│   └── MOEGripper/
├── examples/
│   ├── run_random_multifinger.py
│   ├── run_random_crawling.py
│   └── quick_start.py
└── tests/
```

## Incremental Action Control

Actions represent **force deltas**, not absolute forces:
- Actions are in range `[-1, 1]` 
- Scaled by `max_delta` (40000 for MultiFingerMOE, 8000 for CrawlingMOE)
- Forces accumulate: `new_force = current_force + (action * max_delta)`
- Forces are clamped to `[0, max_force]`
- Forces reset to zero on `env.reset()`

See `INCREMENTAL_ACTIONS.md` for details.

## Troubleshooting

**Import Error: "No module named 'sofagym_moe'"**
- Ensure you installed from the parent directory of `sofagym_moe/`
- The `pyproject.toml` should be one level up from the `sofagym_moe/` folder
- Check: `pip show sofagym-moe` to verify installation

**Import Error: "No module named 'sofagym'"**
- Install the base sofagym framework first (see Installation Step 1)

**SOFA errors on startup**
- Verify SOFA, SofaPython3, and SoftRobots are properly installed
- Check SOFA can be imported: `python3 -c "import Sofa"`

## License

This package extends SofaGym with MOE-specific environments. See original SofaGym license.
