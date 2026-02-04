"""
Main source package initialization.
"""

__version__ = "1.0.0"
__author__ = "Malaria Detection Team"

# Import key modules
from . import data
from . import models
from . import training
from . import evaluation
from . import visualization

__all__ = [
    'data',
    'models',
    'training',
    'evaluation',
    'visualization'
]
