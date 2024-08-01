#! /usr/bin/env python
"""
Set up for acfhod
"""
from setuptools import setup
import os

def get_requirements():
    """
    Read the requirements from a file
    """
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt') as req:
            for line in req:
                # skip commented lines
                if not line.startswith('#'):
                    requirements.append(line.strip())
    return requirements

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('acfhod/data')

setup(
    name='acfhod',
    version=0.1,
    description="Angular Correlation Function with Halo Occupation Distribution",
    author="Giovanni Ferrami",
    author_email="gferrami@student.unimelb.edu.au",
    url="https://github.com/Ferr013/ACFHOD",
    packages = ['acfhod', 'acfhod.HOD', 'acfhod.Plots', 'acfhod.Utils', 'acfhod.ClusteringAnalysis'],
    package_data={'': extra_files},
    install_requires=get_requirements(),
    python_requires='>=3.8',
    license="BSD-3"
)