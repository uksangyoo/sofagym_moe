from setuptools import setup, find_packages
import os

# Read long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sofagym-moe",
    version="0.1.0",
    author="SofaGym Contributors",
    description="MOE soft robotics environments for SofaGym",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['sofagym_moe', 'sofagym_moe.envs', 'sofagym_moe.envs.MOE', 'sofagym_moe.envs.MOEGripper', 'sofagym_moe.envs.MultiFingerMOE', 'sofagym_moe.envs.CrawlingMOE'],
    package_dir={'': '..'},
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
