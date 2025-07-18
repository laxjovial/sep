# domain_tools/weather_tools/__init__.py

import logging
# from typing import Any, Optional, List # Not strictly needed here if only importing the class

# Import the WeatherTools class directly from weather_tool.py
# This is the ONLY import needed from weather_tool.py for this package's __init__.py
from .weather_tool import WeatherTools

logger = logging.getLogger(__name__)

# This line explicitly defines what is exposed when
# someone does `from domain_tools.weather_tools import *`
__all__ = ["WeatherTools"]

logger.info("WeatherTools package initialized.")

# You should remove:
# - All individual function imports (get_current_weather, get_weather_forecast, etc.)
# - The entire redundant 'class WeatherTools:' definition and its methods
#   because the actual tool methods are within the WeatherTools class in weather_tool.py.