import numpy as np

class DummyRewardShaper:
    def __init__(self):
        self._target_position = type('', (), {})()
        self._target_position.value = [np.array([10.0, 0.0, 0.0])]

    def _cube_center(self, _): 
        return np.random.normal(10, 3, 3)

    def _finger_positions_gen(self, _):
        return [np.random.normal(10, 3, 3) for _ in range(3)]

    def _any_finger_in_contact(self, _):
        return np.random.rand() > 0.8  # 20% chance of contact

    def _cube_up_vector(self, _):
        v = np.random.normal(0, 1, 3)
        return v / np.linalg.norm(v)

# Inject dummy into reward function
def test_getReward():
    rsh = DummyRewardShaper()
    prev_dist = None
    for t in range(10):
        cube_center = rsh._cube_center(None)
        target_pos = rsh._target_position.value[0]
        dist = np.linalg.norm(cube_center - target_pos)

        reward = 1.0 / (1.0 + dist)
        if prev_dist is not None:
            delta = prev_dist - dist
            reward += 0.5 * delta
        prev_dist = dist

        min_finger_dist_sq = min(np.sum((cube_center - fpos)**2) for fpos in rsh._finger_positions_gen(None))
        reward += 0.3 / (1 + np.sqrt(min_finger_dist_sq))
        if rsh._any_finger_in_contact(None):
            reward += 1.0
        z_axis = np.array([0, 0, 1])
        cube_up = rsh._cube_up_vector(None)
        reward += 0.2 * np.dot(z_axis, cube_up)
        print(f"Step {t}: reward={reward:.3f}, dist={dist:.2f}")

if __name__ == "__main__":
    test_getReward()
