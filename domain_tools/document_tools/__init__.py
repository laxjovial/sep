# domain_tools/document_tools/__init__.py

import logging
from .document_tool import DocumentTools # Import the class

logger = logging.getLogger(__name__)

__all__ = ["DocumentTools"]

logger.info("DocumentTools package initialized.")
