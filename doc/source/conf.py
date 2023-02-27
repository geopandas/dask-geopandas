# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import dask_geopandas  # noqa

autodoc_mock_imports = [
    "pygeos",
    "dask",
]

# -- Project information -----------------------------------------------------

project = "dask-geopandas"
copyright = "2020-, GeoPandas development team"
author = "GeoPandas development team"

# The full version, including alpha/beta/rc tags
release = version = dask_geopandas.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinx_copybutton",
]

numpydoc_show_class_members = False
autosummary_generate = True
jupyter_execute_notebooks = "auto"
execution_excludepatterns = [
    "basic-intro.ipynb",
    "dissolve.ipynb",
    "spatial-partitioning.ipynb",
]


def setup(app):
    app.add_css_file("custom.css")  # may also be an URL


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/geopandas/dask-geopandas",
    "use_repository_button": True,
    "use_fullscreen_button": False,
}
html_title = "dask-geopandas"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
