from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1",
    packages=find_packages(include=["assets", "robots", "tests", "utils", "assets.*", "robots.*", "tests.*", "utils.*"]),
    install_requires=[],  # Add your dependencies here if needed
)
