#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="project_theia",
    version="0.1",
    authors=[],
    description="Ultrasound tracking helper",
    packages=find_packages(
        include=["compute_environment", "compatibility", "project_theia", "project_theia.*"]
    ),
    install_requires=[
        "lightning==2.5.5"
        "matplotlib==3.10.7"
        "torch==2.8.0"
        "wandb-0.22.2", 'lightning', 'torchvision'
    ],
    python_requires=">=3.12, <3.14",
    setup_requires=[
        "setuptools_scm",
    ],
    extras_require={
        "test": ["pytest==6.2.2"],
        "formatting": ["black==22.10.0"],
        "dev": ["flake8==3.8.0"],
    },
)
