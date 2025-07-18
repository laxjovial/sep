# domain_tools/legal_tools/__init__.py

import logging
# from typing import Any, Optional # Not strictly needed here if only importing the class

# Import the LegalTools class directly from legal_tool.py
# This is the ONLY import needed from legal_tool.py for this package's __init__.py
from .legal_tool import LegalTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.legal_tools import *`
__all__ = ["LegalTools"]

logger.info("LegalTools package initialized.")

# You should remove:
# - All individual function imports (perform_legal_research, legal_search_web, etc.)
# - The entire redundant 'class LegalTools:' definition and its methods
#   because the actual tool methods are within the LegalTools class in legal_tool.py.