from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1",
    packages=find_packages(include=["my_assets", "my_robots", "my_tests", "my_utils", "my_assets.*", "my_robots.*", "my_tests.*", "my_utils.*"]),
    install_requires=[],  # Add your dependencies here if needed
)
