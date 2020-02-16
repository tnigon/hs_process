# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

autodoc_mock_imports = ["geopandas", "osgeo", "seaborn"]

import sphinx_bootstrap_theme
import recommonmark
from recommonmark.transform import AutoStructify

# autodoc_default_options = {
#     'members': None
# }

autosummary_generate = True

# -- Project information -----------------------------------------------------

project = 'hs_process'
copyright = '2020, Tyler J. Nigon'
author = 'Tyler J. Nigon'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    # 'docfx_yaml.extension',  # creates configuration.yaml file
    'sphinx.ext.autosummary',  # Creates TOC sub-level for hs_process methods
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'recommonmark',
    'nbsphinx',  # converts Jupyter notebooks to html
    'sphinx.ext.napoleon',  # for parsing docstrings
    'sphinx_automodapi.automodapi',  # Generates individual pages for each function
    'sphinx_automodapi.smart_resolver',  # Tries to resolve errors that import classes from other files
    'autodocsumm'
]

automodapi_inheritance_diagram = False  # indicates whether to show inheritance diagrams by default
numpydoc_show_class_members = False  # needed to avoid having methods and attributes of classes being shown multiple times.
numpydoc_class_members_toctree = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

language = 'python'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Keeps sphinx from reading files in this folder. This does not affect
# sphinx-apidoc (must add it again when creating docs)
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'bootstrap'
# html_theme = 'sphinx_rtd_theme'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
# html_theme = 'cloud'
# html_theme_path = [cloud_sptheme.get_theme_dir()]
# def setup(app):
#     app.add_stylesheet('bootstrap.min.css')
#     app.add_javascript('jquery-1.11.0.min.js')
#     app.add_javascript('jquery-fix')
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    'navbar_title': "hs_process",
    'navbar_links': [
        ('Github', "https://github.com/tnigon/hs_process", True),
        ('Spectral Python', "http://www.spectralpython.net/", True)
    ],
    'navbar_site_name': "Contents",
    'globaltoc_depth': 3,
    'navbar_pagenav': False,  # sidebar is doing this
    # 'navbar_pagenav_name': "Page Menu",
    'navbar_fixed_top': "true",
    'bootswatch_theme': "flatly",  # DO NOT CAPTIALIZE
    # 'bootswatch_theme': "spacelab",
    'bootstrap_version': "3",
    }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
html_sidebars = {
   '**': ['localtoc.html', 'searchbox.html'],
   'using/windows': ['windowssidebar.html', 'searchbox.html'],
}
