# domain_tools/travel_tools/__init__.py

import logging
# from typing import Any, Optional # Not strictly needed here if only importing the class

# Import the TravelTools class directly from travel_tool.py
# This is the ONLY import needed from travel_tool.py for this package's __init__.py
from .travel_tool import TravelTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.travel_tools import *`
__all__ = ["TravelTools"]

logger.info("TravelTools package initialized.")

# You should remove:
# - All individual function imports (search_flights, Google Hotels, etc.)
# - The entire redundant 'class TravelTools:' definition and its methods
#   because the actual tool methods are within the TravelTools class in travel_tool.py.