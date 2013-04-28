# -*- coding: utf-8 -*-
"""
pySparseTransforms: Sparse transforms package writen in cython

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://bitbucket.org/amitibo/pySparseTransforms>
License: See attached license file
"""
from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import numpy as np
import os.path
import sys


NAME = 'pySparseTransforms'
PACKAGE_NAME = 'sparse_transforms'
VERSION = '0.1'
DESCRIPTION = 'Sparse transforms package writen in cython'
LONG_DESCRIPTION = """
"""
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
URL = "http://bitbucket.org/amitibo/pySparseTransforms"
KEYWORDS = ["sparse", "transforms"]
LICENSE = 'BSD'
CLASSIFIERS = [
    'License :: OSI Approved :: BSD License',
    'Development Status :: 3 - Alpha',
    'Topic :: Scientific/Engineering'
]
ICLUDE_DIRS = [np.get_include()]

def main():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        license=LICENSE,
        packages=[PACKAGE_NAME],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [
            Extension(
                PACKAGE_NAME + '.' + 'cytransforms',
                [
                    'src/cytransforms.pyx',
                ],
                include_dirs=ICLUDE_DIRS
            )
        ],
    )


if __name__ == '__main__':
    main()
