# domain_tools/news_tools/__init__.py

import logging
# from typing import Any, Optional # No longer strictly needed here if only importing the class

# Import the NewsTools class directly from news_tool.py
# This is the ONLY import needed from news_tool.py for this package's __init__.py
from .news_tool import NewsTools

logger = logging.getLogger(__name__)

# This line is good practice to explicitly define what is exposed when
# someone does `from domain_tools.news_tools import *`
__all__ = ["NewsTools"]

logger.info("NewsTools package initialized.")

# You should remove:
# - All individual function imports (get_top_headlines, search_news, etc.)
# - The entire redundant 'class NewsTools:' definition and its methods
#   because the actual tool methods are within the NewsTools class in news_tool.py.