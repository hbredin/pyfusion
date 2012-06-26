#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup, find_packages

setup(
    name='PyFusion',
    version='0.1.0',
    description='Python module for fusion',
    author='Hervé Bredin',
    author_email='bredin@limsi.fr',
    url='http://packages.python.org/PyFusion',
    # packages= find_packages(),
    packages=['pyfusion', 
              'pyfusion.normalization',
              'pyfusion.late'],
    install_requires=['numpy >=1.6.1', 
                      'scipy >=0.10.1',
                      'scikit-learn >=0.11'],
    classifiers=[ 
       "Development Status :: 4 - Beta", 
       "Intended Audience :: Science/Research", 
       "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)", 
       "Natural Language :: English", 
       "Programming Language :: Python :: 2.7", 
       "Topic :: Scientific/Engineering"]
)