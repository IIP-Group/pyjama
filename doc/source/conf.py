# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyJama'
copyright = '2024, Fabian Ulbricht, Gian Marti, Reinhard Wiesmayr'
author = 'Fabian Ulbricht, Gian Marti Reinhard Wiesmayr'
release = '0.1'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

sys.path.append(os.path.abspath("./_ext")) # load custom extensions

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': False,
    'display_version': True,
     'navigation_depth': 5,
    }
html_show_sourcelink = False
pygments_style = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/sionna.css']

intersphinx_mapping = {'sionna': ('https://nvlabs.github.io/sionna', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'tensorflow': (
                           'https://www.tensorflow.org/api_docs/python',
                           'https://github.com/GPflow/tensorflow-intersphinx/raw/master/tf2_py_objects.inv'),
                       }

napoleon_custom_sections = [("Input shape", "params_style"),
                            ("Output shape", "params_style"),
                            ("Attributes", "params_style"),
                            ("Input", "params_style"),
                            ("Output", "params_style")
                            ]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

numfig = True

# do not re-execute jupyter notebooks when building the docs
nbsphinx_execute = 'never'
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Make sure that nbsphinx picks the HTML output rather
# than trying to auto-expose the widgets (too complicated).
import nbsphinx
nbsphinx.DISPLAY_DATA_PRIORITY_HTML = tuple(
    m for m in nbsphinx.DISPLAY_DATA_PRIORITY_HTML
    if not m.startswith('application/')
)
# Avoid duplicate display of widgets, see: https://github.com/spatialaudio/nbsphinx/issues/378#issuecomment-573599835
nbsphinx_widgets_path = ''
