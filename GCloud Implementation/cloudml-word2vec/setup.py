from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==1.5', 'urllib3==1.26.5']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Word 2 vec',
    requires=['word2vec_ops1.so']
)
