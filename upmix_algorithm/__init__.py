"""
Upmix Algorithm - Conversion audio multicanal.

Ce package implémente un algorithme d'upmix pour convertir des signaux audio
d'un format multicanal à un autre (par exemple stéréo vers 5.1 ou 7.1).
"""

__version__ = "0.1.0"

from .upmix_processor import UpmixProcessor, create_default_params

__all__ = ["UpmixProcessor", "create_default_params", "__version__"]
