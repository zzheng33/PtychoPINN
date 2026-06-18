from setuptools import find_packages, setup


setup(
    name="ptychopinn-torch",
    version="0.1.0",
    packages=find_packages(include=["ptychopinn_torch", "ptychopinn_torch.*"]),
)
