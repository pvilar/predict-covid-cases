""" Setup script for package """

from setuptools import setup, find_packages

with open("requirements.txt") as reqs:
    requirements = reqs.read().splitlines()

setup(
    name="corona",
    version="0.0.1",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.5"
)
