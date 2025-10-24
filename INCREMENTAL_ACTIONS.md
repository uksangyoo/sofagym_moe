# Incremental Force Control

All cable-actuated MOE environments use **incremental force control** for smooth, velocity-like manipulation.

## What Changed

### Before
Actions represented absolute cable forces:
```python
action = [0.5, -0.3]  # Sets cable forces to +20000N and -12000N directly
```

### After  
Actions represent force **deltas** (changes):
```python
action = [0.5, -0.3]  # Increases cable 1 by +4000N, decreases cable 2 by -2400N
```

## Implementation Details

### ApplyAction Controller
- **`current_forces`**: Tracks accumulated force for each cable (initialized to zeros)
- **`max_delta`**: Maximum force change per step
- **`max_force`**: Absolute force limits (clamping bounds)
- **`reset()`**: Resets `current_forces` to zeros

### Force Update Algorithm
```python
for each cable:
    delta = action[i] * max_delta
    new_force = current_forces[i] + delta
    new_force = clip(new_force, -max_force, +max_force)
    current_forces[i] = new_force
```

## Environment Parameters

| Environment | Action Dim | max_delta | max_force |
|------------|-----------|-----------|-----------|
| MultiFingerMOE | 6 (2 per finger × 3 fingers) | 40000 | 200000 |
| CrawlingMOE | 2 (antagonistic pair) | 8000 | 40000 |

## Usage Example

```python
from sofagym_moe.envs.MultiFingerMOE.MultiFingerMOEEnv import MultiFingerMOEEnv
import numpy as np

env = MultiFingerMOEEnv(config={"render": 0})
env.reset()

# Apply constant small action - forces accumulate
action = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
for _ in range(10):
    obs, reward, done, info = env.step(action)
    forces = env.env.root.ApplyAction.current_forces
    print(f"Accumulated forces: {forces[:4]}")

# Forces grow: [8000, -8000, ...] -> [16000, -16000, ...] -> ...
```

## Benefits

1. **Smoother control**: Velocity-like action space
2. **No discontinuities**: Forces change gradually
3. **Better exploration**: Small actions maintain current state
4. **Stable learning**: Reduces action noise impact

## Migration Guide

### Old Code
```python
action = action_space.sample()  # Random absolute forces
env.step(action)
```

### New Code
```python
# Scale random actions down for incremental control
action = action_space.sample() * 0.5  # Recommended: 0.3-0.5
env.step(action)
```

### Training Tips
- Lower action scale (0.3-0.5) for random exploration
- Actions near zero maintain current forces
- Reset clears force accumulation
