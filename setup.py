import os
from setuptools import setup, find_packages

import versioneer

install_requires = [
    "geopandas>=0.10",
    "dask>=2021.06.0",
    "distributed>=2021.06.0",
    "pygeos",
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
    ],
    keywords="dask geopandas spatial distributed cluster",
    description="GeoPandas objects backed with Dask",
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
    install_requires=install_requires,
    tests_require=["pytest"],
    zip_safe=False,
)
