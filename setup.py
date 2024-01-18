from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="adaface",
    version="0.1.0",
    license="proprietary",
    description="hogehoge",
    author="naxa",
    url="https://github.com/naxa/AdaFace",
    packages=find_packages(),
    install_requires=[
        str(r) for r in pkg_resources.parse_requirements(Path(__file__).with_name("requirements.txt").open())
    ],
)
