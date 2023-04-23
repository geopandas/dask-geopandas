import os
import sys

from setuptools import setup, find_packages

# ensure the current directory is on sys.path so versioneer can be imported
# when pip uses PEP 517/518 build rules.
# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.append(os.path.dirname(__file__))

import versioneer

install_requires = [
    "geopandas>=0.10",
    "dask>=2023.2.0",
    "distributed>=2023.2.0",
    "packaging",
]

setup(
    name="dask-geopandas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    maintainer="Julia Signell",
    maintainer_email="jsignell@gmail.com",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="dask geopandas spatial distributed cluster",
    description="Parallel GeoPandas with Dask",
    long_description=(
        open("README.rst").read() if os.path.exists("README.rst") else ""
    ),
    url="https://github.com/geopandas/dask-geopandas",
    project_urls={
        "Documentation": "https://github.com/geopandas/dask-geopandas",
        "Source": "https://github.com/geopandas/dask-geopandas/",
        "Issue Tracker": "https://github.com/geopandas/dask-geopandas/issues",
    },
    packages=find_packages(),
    package_data={"dask_geopandas": ["*.yaml"]},
    python_requires=">=3.7",
    install_requires=install_requires,
    tests_require=["pytest"],
    zip_safe=False,
)
