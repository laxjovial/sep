# domain_tools/crypto_tools/__init__.py

import logging

# Import the CryptoTools class directly from crypto_tool.py
# This is the ONLY import needed from crypto_tool.py
from .crypto_tool import CryptoTools

logger = logging.getLogger(__name__)

# This line ensures that `from domain_tools.crypto_tools import CryptoTools` works
# If you use `from domain_tools.crypto_tools import *`, it will only expose CryptoTools
__all__ = ["CryptoTools"]

logger.info("CryptoTools package initialized.")

# You no longer need to define the CryptoTools class or its methods here.
# The actual tool methods are within the CryptoTools class in crypto_tool.py.