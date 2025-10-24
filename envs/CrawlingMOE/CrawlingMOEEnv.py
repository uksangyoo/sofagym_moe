import os, sys
from typing import Optional
import numpy as np
from gym import spaces
from sofagym.AbstractEnv import AbstractEnv
from sofagym.ServerEnv import ServerEnv
from sofagym.rpc_server import start_scene

from . import CrawlingMOEScene
from . import CrawlingMOEToolbox

class CrawlingMOEEnv:
    path = os.path.dirname(os.path.abspath(__file__))
    metadata = {'render.modes': ['human', 'rgb_array']}
    dim_state = 7
    DEFAULT_CONFIG = {
        "scene": "CrawlingMOE",
        "scene_module_path": __module__.rsplit('.', 1)[0] if '.' in __module__ else __module__,
        "deterministic": True,
        "source": [0, 0, 400],
        "target": [0, 0, 0],
        "goal": False,
        "start_node": None,
        "scale_factor": 10,
        "dt": 0.001,
        "timer_limit": 80,
        "timeout": 50,
        "display_size": (1600, 800),
        "render": 0,
        "save_data": False,
        "save_image": False,
        "save_path": path + "/Results" + "/CrawlingMOE",
        "planning": False,
        "discrete": False,
        "start_from_history": None,
        "python_version": sys.version,
        "zFar": 4000,
        "time_before_start": 0,
        "seed": None,
        "nb_actions": 2,
        "dim_state": dim_state,
        "randomize_states": False,
        "init_states": [0.0]*4,
        "xi": 0, "yi": 0, "zi": 0,
        "ytheta": 0.0, "xtheta": 0.0,
        "target_pos": [80.0, 80.0, 35.0],
        "use_server": False,
    }

    def __init__(self, config=None, root=None, use_server: Optional[bool]=None):
        if use_server is not None:
            self.DEFAULT_CONFIG.update({'use_server': use_server})
        self.use_server = self.DEFAULT_CONFIG["use_server"]

        # CartPole pattern: wrap AbstractEnv/ServerEnv with DEFAULT_CONFIG
        self.env = ServerEnv(self.DEFAULT_CONFIG, config, root=root) if self.use_server \
                   else AbstractEnv(self.DEFAULT_CONFIG, config, root=root)

        self.initialize_states()
        if self.env.config["goal"]:
            self.init_goal()

        # spaces (continuous actions: 2 antagonistic channels)
        self.env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.nb_actions = str(self.env.nb_actions)  # CartPole passes string to start_scene

        high = np.full((self.env.config["dim_state"],), np.finfo(np.float32).max, dtype=np.float32)
        self.env.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Build local SOFA root if not using server
        if self.env.root is None and not self.use_server:
            self.env.init_root()

    def __getattr__(self, name):
        return self.env.__getattribute__(name)

    def initialize_states(self):
        if self.env.config["randomize_states"]:
            self.init_states = self.randomize_init_states()
            self.env.config.update({'init_states': list(self.init_states)})
        else:
            self.init_states = self.env.config["init_states"]

    def randomize_init_states(self):
        return self.env.np_random.uniform(low=-0.05, high=0.05, size=(len(self.env.config["init_states"]),))

    def reset(self):
        self.initialize_states()
        if self.env.config["goal"]:
            self.init_goal()
        self.env.reset()

        if self.use_server:
            obs = start_scene(self.env.config, self.nb_actions)
            state = np.array(obs['observation'], dtype=np.float32)
        else:
            state = np.array(self.env._getState(self.env.root), dtype=np.float32)
        return state

    def step(self, action):
        return self.env.step(action)
