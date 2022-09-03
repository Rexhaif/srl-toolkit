from setuptools import setup, find_packages
from pkg_resources import parse_requirements
from pathlib import Path

with Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in parse_requirements(requirements_txt)
    ]

setup(
    name='srl_toolkit',
    version='0.1',
    packages=["srl_toolkit"],
    install_requires=install_requires,
)