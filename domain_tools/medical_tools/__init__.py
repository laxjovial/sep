# domain_tools/medical_tools/__init__.py

import logging
# from typing import Any, Optional, List # No longer needed here if only importing class

# Import the MedicalTools class directly from medical_tool.py
# This is the ONLY import needed from medical_tool.py
from .medical_tool import MedicalTools

logger = logging.getLogger(__name__)

# This line ensures that `from domain_tools.medical_tools import MedicalTools` works
# If you use `from domain_tools.medical_tools import *`, it will only expose MedicalTools
__all__ = ["MedicalTools"]

logger.info("MedicalTools package initialized.")

# Remove all individual function imports (get_drug_info, check_symptoms, etc.)
# Remove the redundant MedicalTools class definition and its method wrappers.
# The actual @tool decorated methods are within the MedicalTools class in medical_tool.py.