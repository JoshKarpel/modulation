from pathlib import Path
import re

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).parent


def find_version():
    """Grab the version out of modulation/version.py without importing it."""
    version_file_text = (THIS_DIR / "modulation" / "version.py").read_text()
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file_text, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="modulation",
    version=find_version(),
    author="Josh Karpel",
    author_email="josh.karpel@gmail.com",
    description="A Python library for Raman generation calculations.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    packages=find_packages(exclude=["dev", "docker", "sci", "tests"]),
    package_data={"": ["*.pyx"]},
    entry_points={
        "console_scripts": [
            f"{x.stem}=modulation_scans.{x.stem}:main"
            for x in (THIS_DIR / "modulation_scans").iterdir()
            if x.stem.startswith(("scan", "export"))
        ]
    },
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
