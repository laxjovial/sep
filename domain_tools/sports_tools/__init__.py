# domain_tools/sports_tools/__init__.py

import logging
# from typing import Any, Optional # Not strictly needed here if only importing the class

# Import the SportsTools class directly from sports_tool.py
# This is the ONLY import needed from sports_tool.py for this package's __init__.py
from .sports_tool import SportsTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.sports_tools import *`
__all__ = ["SportsTools"]

logger.info("SportsTools package initialized.")

# You should remove:
# - All individual function imports (get_latest_scores, get_upcoming_events, etc.)
# - The entire redundant 'class SportsTools:' definition and its methods
#   because the actual tool methods are within the SportsTools class in sports_tool.py.