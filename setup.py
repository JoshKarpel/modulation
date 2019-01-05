from pathlib import Path

from setuptools import setup

THIS_DIR = Path(__file__).parent

setup(
    name = 'modulation',
    version = '0.1.0',
    author = 'Josh Karpel',
    author_email = 'josh.karpel@gmail.com',
    description = 'A Python library for Raman generation calculations.',
    long_description = Path('README.md').read_text(),
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
        'scripts.scans',
    ],
    package_data = {
        '': ['*.pyx'],
    },
    entry_points = {
        'console_scripts': [
            f'{x.stem}=scripts.scans.{x.stem}:main'
            for x in Path('scripts/scans/').iterdir()
            if x.stem.startswith('scan')
        ],
    },
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib',
        'cython',
        'simulacra',
    ],
)
