# domain_tools/entertainment_tools/__init__.py

import logging
# from typing import Any, Optional # Not strictly needed here if only importing the class

# Import the EntertainmentTools class directly from entertainment_tool.py
# This is the ONLY import needed from entertainment_tool.py for this package's __init__.py
from .entertainment_tool import EntertainmentTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.entertainment_tools import *`
__all__ = ["EntertainmentTools"]

logger.info("EntertainmentTools package initialized.")

# You should remove:
# - All individual function imports (search_movies, search_tv_shows, etc.)
# - The entire redundant 'class EntertainmentTools:' definition and its methods
#   because the actual tool methods are within the EntertainmentTools class in entertainment_tool.py.