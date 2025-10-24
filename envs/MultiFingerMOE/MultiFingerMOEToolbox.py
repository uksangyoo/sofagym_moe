# MultiFingerMOEToolbox.py  — per-finger antagonistic control, reward = -dist(cube,target)

import sys
from typing import List, Tuple, Optional

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

# allow relative imports like CartPole
sys.path.insert(0, __file__.rsplit("/", 2)[0])  # parent/..
sys.path.insert(0, __file__.rsplit("/", 1)[0])  # this dir

SofaRuntime.importPlugin("Sofa.Component")


# ----------------------------
# Helpers
# ----------------------------
def _get_cable_lengths(cables) -> List[float]:
    """Read one scalar length from each cable."""
    out = []
    for c in cables:
        try:
            v = c.CableConstraint.cableLength.value
            if isinstance(v, (list, tuple, np.ndarray)):
                out.append(float(v[0]))
            else:
                out.append(float(v))
        except Exception:
            out.append(0.0)
    return out


def _infer_finger_spans(cables: List, cables_per_finger: int = 4) -> List[Tuple[int, int]]:
    """
    If the scene didn't publish rootNode.finger_cable_spans, infer spans by grouping
    every `cables_per_finger` cables as one finger.
    """
    n = len(cables) // cables_per_finger
    return [(i * cables_per_finger, (i + 1) * cables_per_finger) for i in range(n)]


def _cube_positions(rootNode) -> Optional[np.ndarray]:
    """
    Return Nx3 array of the cube vertices (Vec3) if found, else None.
    Expects a node named 'Cube' with MechanicalObject 'mstate' (matches your scene).
    """
    try:
        cube = getattr(rootNode, "Cube")
        mo = getattr(cube, "mstate")
        arr = np.asarray(mo.position.value, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # ensure (N,3)
        cols = min(arr.shape[1], 3)
        xyz = arr[:, :cols]
        if xyz.shape[1] < 3:
            pad = np.zeros((xyz.shape[0], 3 - xyz.shape[1]), dtype=float)
            xyz = np.concatenate([xyz, pad], axis=1)
        return xyz
    except Exception:
        return None


def _cube_center(rootNode) -> Optional[np.ndarray]:
    """Return 3-vector (center of mass approx) of cube by averaging vertices."""
    xyz = _cube_positions(rootNode)
    if xyz is None or xyz.size == 0:
        return None
    return np.mean(xyz, axis=0)


# ----------------------------
# Actions (Gym -> SOFA command)
# ----------------------------
class ApplyAction(Sofa.Core.Controller):
    """
    STRICT per-finger antagonistic control with INCREMENTAL force changes:
      - action size MUST be 2 * n_fingers
      - for each finger i: [a0, a1] -> 4 cable force DELTAS [+a0, -a0, +a1, -a1] * max_delta
      - Current forces are tracked and updated incrementally
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = kwargs["root"]
        self.cables: List = kwargs["cables"]  # flattened list across fingers
        self.max_force = float(kwargs.get("max_force", 200000.0))
        self.max_delta = float(kwargs.get("max_delta", 40000.0))  # Max force change per step

        # Spans published by the scene, else infer
        self.spans: List[Tuple[int, int]] = getattr(self.root, "finger_cable_spans", None)
        if not self.spans:
            self.spans = _infer_finger_spans(self.cables, cables_per_finger=4)

        self.n_fingers = len(self.spans)
        
        # Track current forces for each cable (initialized to zero)
        self.current_forces = [0.0] * len(self.cables)

    def compute_action(self, action) -> List[float]:
        """Compute new forces by adding action deltas to current forces."""
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2 * self.n_fingers:
            raise ValueError(
                f"ApplyAction: expected action size 2*n_fingers ({2*self.n_fingers}), got {a.size}"
            )
        
        # Compute force deltas from action
        force_deltas = []
        for i in range(self.n_fingers):
            a0, a1 = a[2 * i: 2 * i + 2]
            # Each action component creates antagonistic deltas scaled by max_delta
            d0 = +a0 * self.max_delta
            d1 = -a0 * self.max_delta
            d2 = +a1 * self.max_delta
            d3 = -a1 * self.max_delta
            force_deltas.extend([d0, d1, d2, d3])
        
        # Update current forces incrementally and clamp to limits
        new_forces = []
        for i, delta in enumerate(force_deltas):
            new_force = self.current_forces[i] + delta
            new_force = float(np.clip(new_force, -self.max_force, self.max_force))
            new_forces.append(new_force)
            self.current_forces[i] = new_force
        
        return new_forces

    def apply_action(self, forces: List[float]):
        if len(forces) != len(self.cables):
            raise ValueError(f"apply_action: got {len(forces)} forces for {len(self.cables)} cables")
        for cable, force in zip(self.cables, forces):
            cable.CableConstraint.value = [float(force)]
    
    def reset(self):
        """Reset all tracked forces to zero."""
        self.current_forces = [0.0] * len(self.cables)


# ----------------------------
# Initial state
# ----------------------------
class StateInitializer(Sofa.Core.Controller):
    """
    STRICT per-finger antagonistic initializer:
      - init_states MUST be length 2*n_fingers
      - expands per-finger [a0, a1] -> [+a0, -a0, +a1, -a1] (absolute forces)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rootNode = kwargs["rootNode"]
        self.cables: List = kwargs["cables"]
        self.init_states = kwargs.get("init_states", [0.0, 0.0])  # validated/expanded below

    def _spans(self):
        spans = getattr(self.rootNode, "finger_cable_spans", None)
        return spans if spans else _infer_finger_spans(self.cables, cables_per_finger=4)

    def init_state(self, init_states):
        vals = np.asarray(init_states, dtype=float).reshape(-1)
        spans = self._spans()
        n = len(spans)

        if vals.size != 2 * n:
            raise ValueError(f"StateInitializer: init_states must be length 2*n_fingers ({2*n}), got {vals.size}")

        expanded = []
        for i in range(n):
            a0, a1 = vals[2 * i: 2 * i + 2]
            expanded.extend([+a0, -a0, +a1, -a1])  # absolute forces

        for cable, val in zip(self.cables, expanded):
            cable.CableConstraint.value = [float(val)]
        
        # Reset the ApplyAction force tracker when initializing state
        if hasattr(self.rootNode, 'ApplyAction'):
            self.rootNode.ApplyAction.reset()


# ----------------------------
# Reward: -‖cube_center - target_pos‖
# ----------------------------
class RewardShaper(Sofa.Core.Controller):
    """
    Reward = negative distance between cube center and target_pos.
    Done when distance < done_threshold (default 5.0).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rootNode = kwargs["rootNode"]
        self.target_pos = np.asarray(kwargs.get("target_pos", [0.0, 80.0, 0.0]), dtype=float)
        self.done_threshold = float(kwargs.get("done_threshold", 5.0))


    def getReward(self):
        c = _cube_center(self.rootNode)
        if c is None:
            return 0.0, float("inf"), self.target_pos.tolist()

        # simple distance reward
        diff = c - self.target_pos
        dist = (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
        reward = -0.5 * dist

        # finger proximity bonus (incremental, memory-efficient)
        try:
            min_finger_dist_sq = float("inf")
            for fp in _finger_positions_gen(self.rootNode):
                dx = fp[0] - c[0]
                dy = fp[1] - c[1]
                dz = fp[2] - c[2]
                d_sq = dx*dx + dy*dy + dz*dz
                if d_sq < min_finger_dist_sq:
                    min_finger_dist_sq = d_sq
            reward += 0.2 / (1.0 + min_finger_dist_sq**0.5)
        except Exception:
            pass

        # optional: cheap contact bonus
        try:
            if _any_finger_in_contact(self.rootNode):
                reward += 1.0
        except Exception:
            pass

        # optional: simple orientation reward
        try:
            cube_up = _cube_up_vector(self.rootNode)  # 3-vector
            target_up = np.array([0,0,1], dtype=float)
            ori_error = 1.0 - np.dot(cube_up, target_up)
            reward += -0.3 * ori_error
        except Exception:
            pass

        return reward, dist, self.target_pos.tolist()




    def update(self, goal=None):
        pass


# ----------------------------
# Gym <-> SOFA glue
# ----------------------------
def getState(rootNode):
    """
    Observation = cable lengths ONLY (size = 4 * n_fingers).
    """
    return _get_cable_lengths(rootNode.ApplyAction.cables)


def getReward(rootNode):
    if hasattr(rootNode, "Reward"):
        reward, dist, _ = rootNode.Reward.getReward()
        done = dist < getattr(rootNode.Reward, "done_threshold", 5.0)
        return done, reward
    return False, 0.0


def getPos(root):
    """
    Convenience: return cube center for logging/visualization; not in observation.
    """
    c = _cube_center(root)
    if c is None:
        return [0.0, 0.0, 0.0]
    return [float(c[0]), float(c[1]), float(c[2])]


def setPos(root, pos):
    """No-op placeholder for compatibility."""
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
