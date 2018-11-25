from setuptools import setup
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

setup(
    name = 'modulation',
    version = '0.1.0',
    author = 'Josh Karpel',
    author_email = 'josh.karpel@gmail.com',
    description = 'A Python library for Raman generation calculations.',
    long_description = open('README.md').read(),
    long_description_content_type = "text/markdown",
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    packages = [
        'modulation',
    ],
    package_data = {
        '': ['*.pyx'],
    },
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib',
        'cython',
        'simulacra',
    ],
)
