"""Basic registration helper for sofagym_moe.

This module doesn't try to reimplement sofagym's registration logic; it simply
imports the subpackages so any module-level registration in the original env
modules runs. Call `sofagym_moe.register.register_all()` after installing the
package to ensure gym envs are available.
"""


def register_all():
    # Import subpackages to trigger any module-level registration logic
    try:
        from .envs import MOE  # noqa: F401
    except Exception:
        pass
    try:
        from .envs import MultiFingerMOE  # noqa: F401
    except Exception:
        pass
    try:
        from .envs import CrawlingMOE  # noqa: F401
    except Exception:
        pass
    try:
        from .envs import MOEGripper  # noqa: F401
    except Exception:
        pass


if __name__ == "__main__":
    register_all()
