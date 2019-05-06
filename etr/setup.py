import re
from setuptools import setup, find_packages


version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('etr/etr.py').read(),
    re.M
    ).group(1)


with open("README.md", "rb") as f:
    long_descr = f.read().decode("utf-8")


setup(
    name="etr",
    entry_points={
        "console_scripts": ['etr = etr.etr:main']
        },
    version=version,
    description="Experiment Tools for Reinforcement learning.",
    long_description=long_descr,
    author="Xin Hong",
    author_email="silverhugh.77@gmail.com",
    packages=find_packages(exclude=('tests')),
    install_requires=['argparse', 'ruamel.yaml'],
    include_package_data=True,
    zip_safe=False
)
