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
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))


# -- Project information -----------------------------------------------------

project = 'rnet'
copyright = '2022, Kota Sakazaki'
author = 'Kota Sakazaki'

# The full version, including alpha/beta/rc tags
release = '0.0.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'numpydoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.githubpages',
              'sphinx.ext.intersphinx',
              'sphinx.ext.napoleon'
]
autodoc_default_options = {
    'show-inheritance': True
}
autodoc_mock_imports = ['osgeo', 'qgis']
autodoc_typehints = 'none'
numpydoc_attributes_as_param_list = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'qgis': ('https://qgis.org/pyqgis/master/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ksakazaki/rnet",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
   ]
}

html_context = {
    "github_repo": "https://github.com/ksakazaki/rnet"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
