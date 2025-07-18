# domain_tools/education_tools/__init__.py

import logging
# from typing import Any, Optional # Not strictly needed here if only importing the class

# Import the EducationTools class directly from education_tool.py
# This is the ONLY import needed from education_tool.py for this package's __init__.py
from .education_tool import EducationTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.education_tools import *`
__all__ = ["EducationTools"]

logger.info("EducationTools package initialized.")

# You should remove:
# - All individual function imports (search_educational_resources, education_search_web, etc.)
# - The entire redundant 'class EducationTools:' definition and its methods
#   because the actual tool methods are within the EducationTools class in education_tool.py.