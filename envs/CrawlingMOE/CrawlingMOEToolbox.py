# CrawlingMOEToolbox.py
import sys
from typing import List

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

# allow relative imports like CartPole does
sys.path.insert(0, __file__.rsplit("/", 2)[0])  # parent/..
sys.path.insert(0, __file__.rsplit("/", 1)[0])  # this dir

SofaRuntime.importPlugin("Sofa.Component")

# ----------------------------
# Helpers
# ----------------------------
_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def _get_finger_positions(rootNode):
    """Return an (N,>=3) array of vertex positions for the deformable finger."""
    try:
        pos = rootNode.Modeling.finger.finger.dofs.position.value  # [[x,y,z,...], ...]
        arr = np.asarray(pos, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return np.zeros((1, 3), dtype=float)


def _xyz_from_any(arr: np.ndarray) -> np.ndarray:
    """Ensure we return an (N,3) xyz array from an (N,>=3) array."""
    cols = min(arr.shape[1], 3)
    xyz = arr[:, :cols]
    if xyz.shape[1] < 3:
        # pad missing columns on the right
        pad = np.zeros((xyz.shape[0], 3 - xyz.shape[1]), dtype=float)
        xyz = np.concatenate([xyz, pad], axis=1)
    return xyz


class ApplyAction(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = kwargs["root"]
        self.cables = kwargs["cables"]
        self.max_force = float(kwargs.get("max_force", 40000.0))
        self.max_delta = float(kwargs.get("max_delta", 8000.0))
        self.current_forces = [0.0] * len(self.cables)

    def compute_action(self, action) -> List[float]:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2:
            raise ValueError(f"Expected action of size 2, got {a.size}")

        d0 = +a[0] * self.max_delta
        d1 = -a[0] * self.max_delta
        d2 = +a[1] * self.max_delta
        d3 = -a[1] * self.max_delta

        new_forces = []
        for i, delta in enumerate([d0, d1, d2, d3]):
            new_force = self.current_forces[i] + delta
            new_force = float(np.clip(new_force, -self.max_force, self.max_force))
            new_forces.append(new_force)
            self.current_forces[i] = new_force
        
        return new_forces

    def apply_action(self, forces: List[float]):
        for cable, force in zip(self.cables, forces):
            cable.CableConstraint.value = [float(force)]
    
    def reset(self):
        self.current_forces = [0.0] * len(self.cables)


class StateInitializer(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rootNode = kwargs["rootNode"]
        self.cables = kwargs["cables"]
        self.init_states = kwargs.get("init_states", [0.0, 0.0, 0.0, 0.0])

    def init_state(self, init_states):
        vals = np.asarray(init_states, dtype=float).reshape(-1)
        if vals.size == 2:
            vals = np.array([+vals[0], -vals[0], +vals[1], -vals[1]], dtype=float)
        elif vals.size != 4:
            raise ValueError(f"init_states must be length 2 or 4, got {vals.size}")

        for cable, val in zip(self.cables, vals.tolist()):
            cable.CableConstraint.value = [float(val)]
        
        if hasattr(self.rootNode, 'ApplyAction'):
            self.rootNode.ApplyAction.reset()

        # Force the Reward to re-lock the tip index on the next use (lazy, so it uses
        # the *current* geometry, not the half-initialized one).
        if hasattr(self.rootNode, "Reward"):
            self.rootNode.Reward.tip_index = None


# ----------------------------
# Reward (tip = highest-at-initialization, but computed lazily)
# ----------------------------
class RewardShaper(Sofa.Core.Controller):
    """
    On first use (or after reset), pick the vertex with the highest coordinate
    along `up_axis`, and then keep using that *same index* for the episode.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rootNode = kwargs["rootNode"]
        self.target_pos = np.asarray(kwargs.get("target_pos", [35.0, 35.0, 35.0]), dtype=float)
        self.up_axis = kwargs.get("up_axis", "y")
        self.tip_index = None  # lazy-lock later when positions are valid

    def _ensure_tip_index(self):
        if self.tip_index is None:
            arr = _get_finger_positions(self.rootNode)
            xyz = _xyz_from_any(arr)
            axis = _AXIS_TO_INDEX.get(self.up_axis, 1)  # default y
            self.tip_index = int(np.argmax(xyz[:, axis]))
            print(f"[INFO]   >>> Tip index locked (highest {self.up_axis}) = {self.tip_index}, pos={xyz[self.tip_index]}")

    def _tip_pos(self):
        self._ensure_tip_index()
        arr = _get_finger_positions(self.rootNode)
        xyz = _xyz_from_any(arr)
        return xyz[self.tip_index, :3]

    def getReward(self):
        tip = self._tip_pos()
        print(f"[INFO]   >>> Tip position: {tip}, target: {self.target_pos}")
        dist = float(np.linalg.norm(tip - self.target_pos))
        reward = -dist
        return reward, dist, self.target_pos.tolist()

    def update(self, goal=None):
        pass


# ----------------------------
# Gym <-> SOFA glue
# ----------------------------
def getState(rootNode):
    """Return observation. Currently: the 4 cable lengths (floats)."""
    cables = rootNode.ApplyAction.cables
    lengths = []
    for c in cables:
        try:
            v = c.CableConstraint.cableLength.value
            if isinstance(v, (list, tuple, np.ndarray)):
                lengths.append(float(v[0]))
            else:
                lengths.append(float(v))
        except Exception:
            lengths.append(0.0)

    # If you want to include the tip xyz too (obs size 7), you can do:
    tip = rootNode.Reward._tip_pos() if hasattr(rootNode, "Reward") else np.zeros(3)
    return lengths + tip.tolist()

    # return lengths  # shape (4,)


def getReward(rootNode):
    reward, dist, _ = rootNode.Reward.getReward()
    done = dist < 5.0  # tweak threshold to taste
    return done, reward


def getPos(root):
    """Persist just the tip position for convenience."""
    if hasattr(root, "Reward"):
        tip = root.Reward._tip_pos()
        return [float(tip[0]), float(tip[1]), float(tip[2])]
    tip = _xyz_from_any(_get_finger_positions(root))[-1, :3]
    return [float(tip[0]), float(tip[1]), float(tip[2])]


def setPos(root, pos):
    """
    No-op placeholder.
    For deformables, forcing only one vertex is not meaningful.
    If you need a true restore, save & set the full `dofs.position.value`.
    """
    pass


def action_to_command(action, root, nb_steps):
    return root.ApplyAction.compute_action(action)


def startCmd(root, action, duration):
    forces = root.ApplyAction.compute_action(action)

    def executeAnimation(root, forces, factor):
        root.ApplyAction.apply_action(forces)

    root.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"root": root, "forces": forces},
            duration=duration,
            mode="once",
        )
    )
