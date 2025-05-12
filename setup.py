"""default setup.py file for compatability with older dependency management systems"""
from setuptools import setup, find_packages

setup(
    name="pyjama",
    version="0.1a1",
    description="PyJama: Differentiable Jamming and Anti-Jamming with NVIDIA Sionna",
    author="Ulbricht, Fabian and Marti, Gian and Wiesmayr, Reinhard and Studer, Christoph",
    packages=find_packages()
)
