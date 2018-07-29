from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_packge_data=True,
    description='Generic example trainer package.',
)