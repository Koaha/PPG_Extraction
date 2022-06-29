from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

setup(
    name='RRest',
    version='1.0',
    packages=find_packages(include=['RRest.*']),
    url='https://github.com/Koaha/RRest',
    install_reqs = parse_requirements('requirements.txt', session='deploy'),
    python_requires='>=3.6',
    license='MIT',
    author='Koha',
    author_email='',
    description=''
)
