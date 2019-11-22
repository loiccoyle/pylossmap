#!/usr/bin/env python
# Filename: setup.py
"""
The pylossmap setup script.

"""
from setuptools import setup

with open('requirements.txt') as fobj:
    REQUIREMENTS = [l.strip() for l in fobj.readlines()]

try:
    with open("README.md") as fh:
        LONG_DESCRIPTION = fh.read()
except UnicodeDecodeError:
    LONG_DESCRIPTION = "pylossmap, library for handling LHC loss maps."

setup(
    name='pylossmap',
    url='',
    description='',
    long_description=LONG_DESCRIPTION,
    author='Loic Coyle',
    author_email='loic.thomas.coyle@cern.ch',
    packages=['pylossmap'],
    include_package_data=True,
    platforms='any',
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    install_requires=REQUIREMENTS,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
    ],
)

__author__ = 'Loic Coyle'
