import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dymos import __version__

project = 'Dymos'
copyright = '2024, The Dymos Development Team'
author = 'The Dymos Development Team'
version = __version__
release = __version__

extensions = [
    'sphinx.ext.mathjax',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

root_doc = 'index'

exclude_patterns = [
    '_build',
    'dymos_book',
    '**/.ipynb_checkpoints',
    '**.ipynb',
]

myst_enable_extensions = [
    'colon_fence',
    'dollarmath',
]

html_theme = 'basic'
html_static_path = ['_static']
