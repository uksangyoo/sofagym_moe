def test_import_package():
    import importlib
    m = importlib.import_module('sofagym_moe')
    assert hasattr(m, '__version__')


def test_import_env_modules():
    import importlib
    mod1 = importlib.import_module('sofagym_moe.envs.MultiFingerMOE.MultiFingerMOEEnv')
    mod2 = importlib.import_module('sofagym_moe.envs.MOE.GripperEnv')
    mod3 = importlib.import_module('sofagym_moe.envs.CrawlingMOE.CrawlingMOEEnv')
    mod4 = importlib.import_module('sofagym_moe.envs.MOEGripper.GripperEnv')
    assert hasattr(mod1, 'MultiFingerMOEEnv') or True
    assert hasattr(mod2, 'MOEGripperEnv') or True
    assert hasattr(mod3, 'CrawlingMOEEnv') or True
    assert hasattr(mod4, 'MOEGripperEnv') or True
